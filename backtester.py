"""
Backtesting Engine v2 — NSE Scanner Pro
=========================================
CRITICAL FIX: Indicators are computed ONCE per stock, not once per bar.
Old approach: enrich_dataframe() called N times per stock → 250k ops for 500 stocks → 15min
New approach: enrich once → walk forward through pre-computed rows → 60 seconds total

Realistic Indian market cost model: slippage + STT + brokerage + GST
Walk-forward validation: optional 70/30 train/test split
No lookahead bias guaranteed.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from data_engine import Indicators


# ============================================================
# COST MODEL
# ============================================================

@dataclass
class CostModel:
    slippage_pct: float = 0.10
    brokerage_per_order: float = 20
    stt_pct: float = 0.025
    exchange_txn_pct: float = 0.00345
    sebi_pct: float = 0.0001
    gst_pct: float = 18.0

    def total_cost_pct(self, entry_price: float, exit_price: float, shares: int = None) -> float:
        # Default to ₹1,00,000 position size (realistic Indian retail trader)
        if shares is None:
            shares = max(1, int(100000 / entry_price)) if entry_price > 0 else 1
        position_value = entry_price * max(shares, 1)
        slippage       = self.slippage_pct * 2
        brokerage_pct  = (self.brokerage_per_order * 2) / position_value * 100 if position_value > 0 else 0
        stt            = self.stt_pct
        exchange       = self.exchange_txn_pct * 2
        sebi           = self.sebi_pct * 2
        taxable        = brokerage_pct + exchange
        gst            = taxable * (self.gst_pct / 100)
        return slippage + brokerage_pct + stt + exchange + sebi + gst

DEFAULT_COSTS = CostModel()


@dataclass
class BacktestTrade:
    symbol: str
    strategy: str
    signal: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    net_pnl_pct: float = 0.0
    pnl_abs: float = 0.0
    holding_days: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    cost_pct: float = 0.0


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    period: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_pct: float
    net_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    net_profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    expectancy_pct: float = 0.0
    net_expectancy_pct: float = 0.0
    avg_holding_days: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    total_costs_pct: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)
    is_walk_forward: bool = False
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    overfit_warning: bool = False


# ============================================================
# MAIN BACKTEST — THE FIXED VERSION
# ============================================================

def backtest_strategy(
    df: pd.DataFrame, symbol: str, strategy: str,
    lookback_days: int = 500, max_hold: int = 30,
    cost_model: CostModel = None
) -> Optional[BacktestResult]:
    """
    Walk-forward backtest. Indicators computed ONCE, then iterated.
    """
    if df is None or len(df) < 220:
        return None

    test_data = df.iloc[-min(lookback_days, len(df)):]

    # ── CRITICAL FIX: Enrich ONCE ──
    try:
        enriched = Indicators.enrich_dataframe(test_data)
    except Exception as e:
        return None

    trades: List[BacktestTrade] = []
    in_trade = False
    current_trade: Optional[BacktestTrade] = None
    start_idx = min(200, len(enriched) - 20)

    for i in range(start_idx, len(enriched)):
        latest    = enriched.iloc[i]
        date_str  = str(enriched.index[i].date())
        df_window = enriched.iloc[:i + 1]  # visible history up to bar i

        # ── Manage existing trade ──
        if in_trade and current_trade:
            current_trade.holding_days += 1
            low_i  = float(latest.get("low", 0))
            high_i = float(latest.get("high", 0))
            close_i= float(latest.get("close", 0))

            if current_trade.signal == "BUY":
                if low_i > 0 and low_i <= current_trade.stop_loss:
                    _close_trade(current_trade, date_str, current_trade.stop_loss,
                                 "STOP LOSS", "BUY")
                    trades.append(current_trade); in_trade = False; current_trade = None; continue
                if high_i >= current_trade.target_1 > 0:
                    _close_trade(current_trade, date_str, current_trade.target_1,
                                 "TARGET 1", "BUY")
                    trades.append(current_trade); in_trade = False; current_trade = None; continue
                if close_i > 0:
                    day_pnl = (close_i / current_trade.entry_price - 1) * 100
                    current_trade.max_favorable = max(current_trade.max_favorable, day_pnl)
                    current_trade.max_adverse   = min(current_trade.max_adverse, day_pnl)
            else:  # SHORT
                if high_i >= current_trade.stop_loss > 0:
                    _close_trade(current_trade, date_str, current_trade.stop_loss,
                                 "STOP LOSS", "SHORT")
                    trades.append(current_trade); in_trade = False; current_trade = None; continue
                if low_i > 0 and low_i <= current_trade.target_1:
                    _close_trade(current_trade, date_str, current_trade.target_1,
                                 "TARGET 1", "SHORT")
                    trades.append(current_trade); in_trade = False; current_trade = None; continue

            if current_trade.holding_days >= max_hold:
                _close_trade(current_trade, date_str, float(latest.get("close", current_trade.entry_price)),
                             f"MAX HOLD ({max_hold}d)", current_trade.signal)
                trades.append(current_trade); in_trade = False; current_trade = None
            continue

        # ── Check for new signal ──
        signal = _check_signal(df_window, latest, strategy, symbol)
        if signal:
            in_trade = True
            current_trade = signal

    # Close any open trade at end of data
    if in_trade and current_trade:
        last = enriched.iloc[-1]
        _close_trade(current_trade, str(enriched.index[-1].date()),
                     float(last.get("close", current_trade.entry_price)),
                     "END OF DATA", current_trade.signal)
        trades.append(current_trade)

    if not trades:
        return None

    return _compute_stats(trades, strategy, symbol, lookback_days, cost_model or DEFAULT_COSTS)


def _close_trade(t: BacktestTrade, exit_date: str, exit_price: float,
                 exit_reason: str, signal: str):
    """Mutate a trade object with exit info."""
    t.exit_date   = exit_date
    t.exit_price  = round(exit_price, 2)
    t.exit_reason = exit_reason
    if signal == "BUY":
        t.pnl_pct = round((exit_price / t.entry_price - 1) * 100, 2) if t.entry_price > 0 else 0
    else:
        t.pnl_pct = round((t.entry_price / exit_price - 1) * 100, 2) if exit_price > 0 else 0


# ============================================================
# SIGNAL CHECKERS — work with pre-enriched df slice
# ============================================================

def _check_signal(df: pd.DataFrame, latest, strategy: str,
                  symbol: str) -> Optional[BacktestTrade]:
    if strategy == "VCP":               return _check_vcp(df, latest, symbol)
    if strategy == "EMA21_Bounce":      return _check_ema21(df, latest, symbol)
    if strategy == "52WH_Breakout":     return _check_52wh(df, latest, symbol)
    if strategy == "Failed_Breakout_Short": return _check_short(df, latest, symbol)
    if strategy == "Last30Min_ATH":     return _check_ath(df, latest, symbol)
    return None


def _check_vcp(df, latest, symbol):
    if len(df) < 200: return None
    close = float(latest.get("close", 0))
    sma50 = float(latest.get("sma_50", 0))
    sma200= float(latest.get("sma_200", 0))
    if close < sma50 or sma50 < sma200: return None
    pct_52wh = float(latest.get("pct_from_52w_high", -50))
    if pct_52wh < -25: return None

    # Volatility compression: std_10 / std_50
    std10 = float(df["close"].iloc[-10:].std()) if len(df) >= 10 else 999
    std50 = float(df["close"].iloc[-50:].std()) if len(df) >= 50 else 999
    if std50 <= 0 or (std10 / std50) > 0.50: return None

    vol_sma20 = float(latest.get("vol_sma_20", 1) or 1)
    avg_vol5  = float(df["volume"].iloc[-5:].mean()) if len(df) >= 5 else vol_sma20
    if avg_vol5 / vol_sma20 > 0.70: return None

    pivot = round(float(df["high"].iloc[-25:].max()), 2)
    entry = close if close >= pivot * 0.998 else pivot
    atr   = float(latest.get("atr_14", close * 0.02) or close * 0.02)
    sl    = round(float(df["low"].iloc[-15:].min()) * 0.995, 2)
    risk  = entry - sl
    if risk <= 0 or (risk / entry * 100) > 8: return None

    return BacktestTrade(symbol=symbol, strategy="VCP", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=sl, target_1=round(entry + 2*risk, 2), target_2=round(entry + 3.5*risk, 2))


def _check_ema21(df, latest, symbol):
    if len(df) < 60: return None
    close = float(latest.get("close", 0))
    ema21 = float(latest.get("ema_21", 0))
    sma50 = float(latest.get("sma_50", 0))
    sma200= float(latest.get("sma_200", 0))
    if close < sma50 or sma50 < sma200: return None
    if ema21 <= 0: return None
    ema_dist = (close - ema21) / ema21 * 100
    if ema_dist > 3 or ema_dist < -2: return None
    if float(latest.get("low", close)) > ema21 * 1.01: return None
    if close < ema21 or close < float(latest.get("open", close)): return None

    atr  = float(latest.get("atr_14", close * 0.02) or close * 0.02)
    sl   = round(min(float(latest.get("low", close)), ema21) - 0.5 * atr, 2)
    risk = close - sl
    if risk <= 0: return None
    return BacktestTrade(symbol=symbol, strategy="EMA21_Bounce", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(close, 2),
        stop_loss=sl, target_1=round(close + 1.5*risk, 2), target_2=round(close + 2.5*risk, 2))


def _check_52wh(df, latest, symbol):
    if len(df) < 200: return None
    close = float(latest.get("close", 0))
    sma50 = float(latest.get("sma_50", 0))
    sma200= float(latest.get("sma_200", 0))
    if close < sma50 or sma50 < sma200: return None
    if float(latest.get("pct_from_52w_high", -50)) < -3: return None
    prev_high = float(df["high"].iloc[-20:-1].max()) if len(df) >= 21 else 0
    if close <= prev_high: return None
    vol_ratio = float(latest.get("volume", 0)) / (float(latest.get("vol_sma_20", 1) or 1))
    if vol_ratio < 1.5: return None

    atr  = float(latest.get("atr_14", close * 0.02) or close * 0.02)
    sl   = round(prev_high - 0.5 * atr, 2)
    risk = close - sl
    if risk <= 0: sl = round(close - 1.5*atr, 2); risk = close - sl
    if risk <= 0: return None
    return BacktestTrade(symbol=symbol, strategy="52WH_Breakout", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(close, 2),
        stop_loss=sl, target_1=round(close + 2*risk, 2), target_2=round(close + 3.5*risk, 2))


def _check_short(df, latest, symbol):
    if len(df) < 60: return None
    if len(df) < 2: return None
    prev  = df.iloc[-2]
    close = float(latest.get("close", 0))
    atr   = float(latest.get("atr_14", close * 0.02) or close * 0.02)
    resistance = float(df["high"].iloc[-20:-2].max()) if len(df) >= 22 else 0
    if float(prev.get("high", 0)) < resistance: return None
    if close >= resistance: return None
    if close >= float(latest.get("open", close)): return None
    if close >= float(prev.get("close", close)): return None

    sl   = round(max(float(latest.get("high", close)), float(prev.get("high", close))) + atr * 0.3, 2)
    risk = sl - close
    if risk <= 0: return None
    return BacktestTrade(symbol=symbol, strategy="Failed_Breakout_Short", signal="SHORT",
        entry_date=str(df.index[-1].date()), entry_price=round(close, 2),
        stop_loss=sl, target_1=round(close - 1.5*risk, 2), target_2=round(close - 2.5*risk, 2))


def _check_ath(df, latest, symbol):
    if len(df) < 50: return None
    close  = float(latest.get("close", 0))
    high_d = float(latest.get("high", 0))
    ema21  = float(latest.get("ema_21", 0))
    if high_d <= 0 or close < ema21: return None
    if (high_d - close) / high_d * 100 > 1.0: return None
    if float(latest.get("pct_from_52w_high", -50)) < -5: return None
    if close < float(latest.get("open", close)): return None
    vol_ratio = float(latest.get("volume", 0)) / (float(latest.get("vol_sma_20", 1) or 1))
    if vol_ratio < 1.0: return None

    atr  = float(latest.get("atr_14", close * 0.02) or close * 0.02)
    sl   = round(float(latest.get("low", close - atr)), 2)
    risk = close - sl
    if risk <= 0: sl = round(close - atr, 2); risk = atr
    if risk <= 0: return None
    return BacktestTrade(symbol=symbol, strategy="Last30Min_ATH", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(close, 2),
        stop_loss=sl, target_1=round(close + 1.5*risk, 2), target_2=round(close + 2.5*risk, 2))


# ============================================================
# STATS COMPUTATION
# ============================================================

def _compute_stats(trades: List[BacktestTrade], strategy: str, symbol: str,
                   lookback: int, cost_model: CostModel = DEFAULT_COSTS) -> BacktestResult:
    for t in trades:
        if t.net_pnl_pct == 0 and t.cost_pct == 0 and t.entry_price > 0:
            cost = cost_model.total_cost_pct(t.entry_price, max(t.exit_price, 0.01))
            t.cost_pct    = round(cost, 3)
            t.net_pnl_pct = round(t.pnl_pct - cost, 3)

    wins   = [t for t in trades if t.net_pnl_pct > 0]
    losses = [t for t in trades if t.net_pnl_pct <= 0]

    total_pnl    = sum(t.pnl_pct for t in trades)
    net_pnl      = sum(t.net_pnl_pct for t in trades)
    total_costs  = sum(t.cost_pct for t in trades)
    avg_win      = np.mean([t.pnl_pct for t in trades if t.pnl_pct > 0]) if any(t.pnl_pct > 0 for t in trades) else 0
    avg_loss_val = np.mean([abs(t.pnl_pct) for t in trades if t.pnl_pct <= 0]) if any(t.pnl_pct <= 0 for t in trades) else 0

    gross_wins   = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_losses = abs(sum(t.pnl_pct for t in trades if t.pnl_pct <= 0))
    pf           = gross_wins / gross_losses if gross_losses > 0 else 0

    net_wins     = sum(t.net_pnl_pct for t in wins)
    net_losses_v = abs(sum(t.net_pnl_pct for t in losses))
    net_pf       = net_wins / net_losses_v if net_losses_v > 0 else 0

    # Equity curve + drawdown
    equity  = []
    running = 0.0
    peak    = 0.0
    max_dd  = 0.0
    for t in trades:
        running += t.net_pnl_pct
        peak = max(peak, running)
        dd   = peak - running
        max_dd = max(max_dd, dd)
        equity.append({"date": t.exit_date, "equity": round(running, 2), "drawdown": round(dd, 2)})

    # Sharpe (annualised, using daily returns proxy)
    daily_returns = np.array([t.net_pnl_pct for t in trades])
    sharpe = 0.0
    if len(daily_returns) > 2 and daily_returns.std() > 0:
        sharpe = round(float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)), 2)

    return BacktestResult(
        strategy=strategy, symbol=symbol,
        period=f"Last {lookback} trading days",
        total_trades=len(trades),
        wins=len(wins), losses=len(losses),
        win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
        total_pnl_pct=round(total_pnl, 2),
        net_pnl_pct=round(net_pnl, 2),
        avg_win_pct=round(avg_win, 2),
        avg_loss_pct=round(avg_loss_val, 2),
        profit_factor=round(pf, 2),
        net_profit_factor=round(net_pf, 2),
        max_drawdown_pct=round(max_dd, 2),
        expectancy_pct=round(total_pnl / len(trades), 2) if trades else 0,
        net_expectancy_pct=round(net_pnl / len(trades), 2) if trades else 0,
        avg_holding_days=round(np.mean([t.holding_days for t in trades]), 1),
        best_trade_pct=round(max(t.net_pnl_pct for t in trades), 2),
        worst_trade_pct=round(min(t.net_pnl_pct for t in trades), 2),
        total_costs_pct=round(total_costs, 2),
        sharpe_ratio=sharpe,
        trades=trades,
        equity_curve=equity,
    )


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def backtest_walk_forward(
    df: pd.DataFrame, symbol: str, strategy: str,
    train_pct: float = 0.70, max_hold: int = 30,
    cost_model: CostModel = None
) -> Optional[BacktestResult]:
    if df is None or len(df) < 400:
        return None

    split_idx  = int(len(df) * train_pct)
    train_df   = df.iloc[:split_idx]
    test_df    = df.iloc[split_idx - 200:]  # overlap for warmup

    train_result = backtest_strategy(train_df, symbol, strategy, len(train_df), max_hold, cost_model)
    test_result  = backtest_strategy(test_df,  symbol, strategy, len(test_df),  max_hold, cost_model)

    if test_result is None:
        return train_result

    test_result.is_walk_forward  = True
    test_result.train_win_rate   = train_result.win_rate if train_result else 0
    test_result.test_win_rate    = test_result.win_rate
    test_result.period           = f"Walk-Forward — Train {train_pct*100:.0f}% / Test {(1-train_pct)*100:.0f}%"
    test_result.overfit_warning  = (
        train_result is not None and
        (train_result.win_rate - test_result.win_rate) > 15
    )
    return test_result


# ============================================================
# MULTI-STOCK PORTFOLIO BACKTEST
# ============================================================

def backtest_multi_stock(
    symbols: List[str], data_dict: Dict[str, pd.DataFrame],
    strategy: str, lookback: int = 500, max_hold: int = 30,
    cost_model: CostModel = None
) -> Optional[BacktestResult]:
    all_trades = []
    for symbol in symbols:
        if symbol not in data_dict:
            continue
        result = backtest_strategy(data_dict[symbol], symbol, strategy, lookback, max_hold, cost_model)
        if result:
            all_trades.extend(result.trades)

    if not all_trades:
        return None

    all_trades.sort(key=lambda t: t.entry_date)
    return _compute_stats(all_trades, strategy, f"{len(symbols)} stocks", lookback, cost_model or DEFAULT_COSTS)
