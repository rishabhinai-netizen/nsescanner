"""
Backtesting Engine — v5.2
===========================
- Slippage + STT + brokerage costs modeled
- Walk-forward validation (train/test split)
- No lookahead bias — each day only sees data available up to that point.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

from data_engine import Indicators

logger = logging.getLogger(__name__)


# ============================================================================
# COST MODEL — Indian market realistic costs
# ============================================================================

@dataclass
class CostModel:
    """Realistic execution cost model for NSE trading."""
    slippage_pct: float = 0.10       # 0.10% per side (market order slippage)
    brokerage_per_order: float = 20  # ₹20 per order (Zerodha-style)
    stt_pct: float = 0.025           # STT: 0.025% on sell side (delivery)
    exchange_txn_pct: float = 0.00345  # NSE transaction charges
    sebi_pct: float = 0.0001         # SEBI turnover fee
    gst_pct: float = 18.0            # GST on brokerage + txn charges

    def total_cost_pct(self, entry_price: float, exit_price: float,
                       shares: int = 1) -> float:
        """
        Compute total round-trip cost as % of entry price.
        Includes: slippage (both sides) + brokerage + STT + exchange + SEBI + GST
        """
        position_value = entry_price * shares

        # Slippage: both entry and exit
        slippage = self.slippage_pct * 2

        # Brokerage: ₹20 per order, 2 orders (buy + sell)
        brokerage_total = self.brokerage_per_order * 2
        brokerage_pct = (brokerage_total / position_value * 100) if position_value > 0 else 0

        # STT: on sell side only (delivery)
        stt = self.stt_pct

        # Exchange + SEBI: both sides
        exchange = self.exchange_txn_pct * 2
        sebi = self.sebi_pct * 2

        # GST on brokerage + exchange charges
        taxable = brokerage_pct + exchange
        gst = taxable * (self.gst_pct / 100)

        return slippage + brokerage_pct + stt + exchange + sebi + gst

    def adjust_pnl(self, gross_pnl_pct: float, entry_price: float,
                   shares: int = 100) -> float:
        """Subtract costs from gross P&L."""
        costs = self.total_cost_pct(entry_price, entry_price, shares)
        return gross_pnl_pct - costs


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
    pnl_pct: float = 0.0        # GROSS P&L
    net_pnl_pct: float = 0.0    # NET P&L (after costs)
    pnl_abs: float = 0.0
    holding_days: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    cost_pct: float = 0.0       # Total cost for this trade


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
    net_pnl_pct: float          # NEW: after costs
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    net_profit_factor: float    # NEW: after costs
    max_drawdown_pct: float
    expectancy_pct: float
    net_expectancy_pct: float   # NEW: after costs
    avg_holding_days: float
    best_trade_pct: float
    worst_trade_pct: float
    total_costs_pct: float      # NEW: total execution costs
    trades: List[BacktestTrade]
    equity_curve: List[dict]
    # Walk-forward fields
    is_walk_forward: bool = False
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    overfit_warning: bool = False


def backtest_strategy(df: pd.DataFrame, symbol: str, strategy: str,
                      lookback_days: int = 500, max_hold: int = 30,
                      cost_model: CostModel = None) -> Optional[BacktestResult]:
    """
    Walk-forward backtest of a strategy on a single stock.
    Now includes execution costs (slippage + STT + brokerage).
    """
    costs = cost_model or DEFAULT_COSTS

    if df is None or len(df) < 220:
        return None

    test_data = df.iloc[-min(lookback_days, len(df)):]
    trades: List[BacktestTrade] = []
    in_trade = False
    current_trade = None
    start_idx = 200

    for i in range(start_idx, len(test_data)):
        hist = test_data.iloc[:i+1].copy()
        if len(hist) < 200:
            continue

        try:
            enriched = Indicators.enrich_dataframe(hist)
        except Exception:
            continue

        latest = enriched.iloc[-1]
        date_str = str(enriched.index[-1].date())

        if in_trade and current_trade:
            current_trade.holding_days += 1

            if current_trade.signal == "BUY":
                if latest["low"] <= current_trade.stop_loss:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = "STOP LOSS"
                    current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
                    current_trade.cost_pct = costs.total_cost_pct(current_trade.entry_price, current_trade.exit_price)
                    current_trade.net_pnl_pct = current_trade.pnl_pct - current_trade.cost_pct
                    trades.append(current_trade)
                    in_trade = False; current_trade = None; continue

                if latest["high"] >= current_trade.target_1:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = current_trade.target_1
                    current_trade.exit_reason = "TARGET 1"
                    current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
                    current_trade.cost_pct = costs.total_cost_pct(current_trade.entry_price, current_trade.exit_price)
                    current_trade.net_pnl_pct = current_trade.pnl_pct - current_trade.cost_pct
                    trades.append(current_trade)
                    in_trade = False; current_trade = None; continue

                day_pnl = (latest["close"] / current_trade.entry_price - 1) * 100
                current_trade.max_favorable = max(current_trade.max_favorable, day_pnl)
                current_trade.max_adverse = min(current_trade.max_adverse, day_pnl)

            else:  # SHORT
                if latest["high"] >= current_trade.stop_loss:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = "STOP LOSS"
                    current_trade.pnl_pct = (current_trade.entry_price / current_trade.exit_price - 1) * 100
                    current_trade.cost_pct = costs.total_cost_pct(current_trade.entry_price, current_trade.exit_price)
                    current_trade.net_pnl_pct = current_trade.pnl_pct - current_trade.cost_pct
                    trades.append(current_trade)
                    in_trade = False; current_trade = None; continue

                if latest["low"] <= current_trade.target_1:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = current_trade.target_1
                    current_trade.exit_reason = "TARGET 1"
                    current_trade.pnl_pct = (current_trade.entry_price / current_trade.exit_price - 1) * 100
                    current_trade.cost_pct = costs.total_cost_pct(current_trade.entry_price, current_trade.exit_price)
                    current_trade.net_pnl_pct = current_trade.pnl_pct - current_trade.cost_pct
                    trades.append(current_trade)
                    in_trade = False; current_trade = None; continue

                day_pnl = (current_trade.entry_price / latest["close"] - 1) * 100
                current_trade.max_favorable = max(current_trade.max_favorable, day_pnl)
                current_trade.max_adverse = min(current_trade.max_adverse, day_pnl)

            if current_trade.holding_days >= max_hold:
                current_trade.exit_date = date_str
                current_trade.exit_price = latest["close"]
                current_trade.exit_reason = f"MAX HOLD ({max_hold}d)"
                if current_trade.signal == "BUY":
                    current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
                else:
                    current_trade.pnl_pct = (current_trade.entry_price / current_trade.exit_price - 1) * 100
                current_trade.cost_pct = costs.total_cost_pct(current_trade.entry_price, current_trade.exit_price)
                current_trade.net_pnl_pct = current_trade.pnl_pct - current_trade.cost_pct
                trades.append(current_trade)
                in_trade = False; current_trade = None
            continue

        signal = _check_signal(enriched, latest, strategy, symbol)
        if signal:
            # Apply entry slippage
            slip = signal.entry_price * (costs.slippage_pct / 100)
            if signal.signal == "BUY":
                signal.entry_price = round(signal.entry_price + slip, 2)
            else:
                signal.entry_price = round(signal.entry_price - slip, 2)
            in_trade = True
            current_trade = signal

    # Close any remaining open trade
    if in_trade and current_trade:
        last = test_data.iloc[-1]
        current_trade.exit_date = str(test_data.index[-1].date())
        current_trade.exit_price = last["close"]
        current_trade.exit_reason = "END OF DATA"
        if current_trade.signal == "BUY":
            current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
        else:
            current_trade.pnl_pct = (current_trade.entry_price / current_trade.exit_price - 1) * 100
        current_trade.cost_pct = costs.total_cost_pct(current_trade.entry_price, current_trade.exit_price)
        current_trade.net_pnl_pct = current_trade.pnl_pct - current_trade.cost_pct
        trades.append(current_trade)

    if not trades:
        return None

    return _compute_stats(trades, strategy, symbol, lookback_days)


# ============================================================================
# WALK-FORWARD BACKTEST — Train/Test validation
# ============================================================================

def backtest_walk_forward(df: pd.DataFrame, symbol: str, strategy: str,
                          train_pct: float = 0.70, max_hold: int = 30,
                          cost_model: CostModel = None) -> Optional[BacktestResult]:
    """
    Walk-forward validation: split data into train (70%) and test (30%).
    Run backtest on both, compare win rates to detect overfitting.
    Returns the TEST period result with train/test comparison.
    """
    if df is None or len(df) < 400:
        return None

    split_idx = int(len(df) * train_pct)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx - 200:]  # Overlap 200 bars for indicator warmup

    train_result = backtest_strategy(train_df, symbol, strategy,
                                     lookback_days=len(train_df), max_hold=max_hold,
                                     cost_model=cost_model)
    test_result = backtest_strategy(test_df, symbol, strategy,
                                    lookback_days=len(test_df), max_hold=max_hold,
                                    cost_model=cost_model)

    if test_result is None:
        return train_result

    # Annotate with walk-forward fields
    test_result.is_walk_forward = True
    test_result.train_win_rate = train_result.win_rate if train_result else 0
    test_result.test_win_rate = test_result.win_rate
    test_result.period = f"Walk-Forward (Train {train_pct*100:.0f}% / Test {(1-train_pct)*100:.0f}%)"

    # Overfit warning: train win rate > test by > 15%
    if train_result and train_result.win_rate - test_result.win_rate > 15:
        test_result.overfit_warning = True

    return test_result


# ============================================================================
# SIGNAL CHECK FUNCTIONS
# ============================================================================

def _check_signal(df: pd.DataFrame, latest, strategy: str,
                  symbol: str) -> Optional[BacktestTrade]:
    if strategy == "VCP":
        return _check_vcp(df, latest, symbol)
    elif strategy == "EMA21_Bounce":
        return _check_ema21(df, latest, symbol)
    elif strategy == "52WH_Breakout":
        return _check_52wh(df, latest, symbol)
    elif strategy == "Failed_Breakout_Short":
        return _check_short(df, latest, symbol)
    elif strategy == "Last30Min_ATH":
        return _check_ath(df, latest, symbol)
    return None


def _check_vcp(df, latest, symbol):
    if len(df) < 200: return None
    if latest["close"] < latest["sma_50"] or latest["sma_50"] < latest["sma_200"]: return None
    if latest["pct_from_52w_high"] < -25: return None
    pct_above_low = (latest["close"] - latest["low_52w"]) / latest["low_52w"] * 100
    if pct_above_low < 30: return None

    recent_range = df["high"].iloc[-10:].max() - df["low"].iloc[-10:].min()
    wide_range = df["high"].iloc[-40:].max() - df["low"].iloc[-40:].min()
    if wide_range == 0: return None
    contraction = recent_range / wide_range
    if contraction > 0.65: return None

    recent_vol = df["volume"].iloc[-10:].mean()
    avg_vol = latest["vol_sma_20"]
    if avg_vol > 0 and recent_vol / avg_vol > 1.2: return None

    pivot = df["high"].iloc[-10:].max()
    entry = latest["close"] if latest["close"] >= pivot else pivot
    atr = latest["atr_14"]
    sl = df["low"].iloc[-10:].min() - 0.5 * atr
    risk = entry - sl
    if risk <= 0: return None

    return BacktestTrade(
        symbol=symbol, strategy="VCP", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry + 2 * risk, 2),
        target_2=round(entry + 3.5 * risk, 2),
    )


def _check_ema21(df, latest, symbol):
    if len(df) < 60: return None
    if latest["close"] < latest["sma_50"]: return None
    if latest["sma_50"] < latest["sma_200"]: return None
    ema_dist = (latest["close"] - latest["ema_21"]) / latest["ema_21"] * 100
    if ema_dist > 3 or ema_dist < -2: return None
    if latest["low"] > latest["ema_21"] * 1.01: return None
    if latest["close"] < latest["ema_21"]: return None
    if latest["close"] < latest["open"]: return None

    entry = latest["close"]
    atr = latest["atr_14"]
    sl = min(latest["low"], latest["ema_21"]) - 0.5 * atr
    risk = entry - sl
    if risk <= 0: return None

    return BacktestTrade(
        symbol=symbol, strategy="EMA21_Bounce", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry + 1.5 * risk, 2),
        target_2=round(entry + 2.5 * risk, 2),
    )


def _check_52wh(df, latest, symbol):
    if len(df) < 200: return None
    if latest["pct_from_52w_high"] < -3: return None
    prev_high = df["high"].iloc[-20:-1].max()
    if latest["close"] <= prev_high: return None
    if latest["close"] < latest["sma_50"]: return None
    if latest["sma_50"] < latest["sma_200"]: return None
    vol_ratio = latest["volume"] / (latest["vol_sma_20"] + 1)
    if vol_ratio < 1.5: return None

    entry = latest["close"]
    atr = latest["atr_14"]
    sl = prev_high - 0.5 * atr
    risk = entry - sl
    if risk <= 0: sl = entry - 1.5 * atr; risk = entry - sl
    if risk <= 0: return None

    return BacktestTrade(
        symbol=symbol, strategy="52WH_Breakout", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry + 2 * risk, 2),
        target_2=round(entry + 3.5 * risk, 2),
    )


def _check_short(df, latest, symbol):
    if len(df) < 60: return None
    prev = df.iloc[-2]
    resistance = df["high"].iloc[-20:-2].max()
    if prev["high"] < resistance: return None
    if latest["close"] >= resistance: return None
    if latest["close"] >= latest["open"]: return None
    if latest["close"] >= prev["close"]: return None

    entry = latest["close"]
    sl = max(latest["high"], prev["high"]) + latest["atr_14"] * 0.3
    risk = sl - entry
    if risk <= 0: return None

    return BacktestTrade(
        symbol=symbol, strategy="Failed_Breakout_Short", signal="SHORT",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry - 1.5 * risk, 2),
        target_2=round(entry - 2.5 * risk, 2),
    )


def _check_ath(df, latest, symbol):
    if len(df) < 50: return None
    if latest["high"] == 0: return None
    close_vs_high = (latest["high"] - latest["close"]) / latest["high"] * 100
    if close_vs_high > 1.0: return None
    if latest["close"] < latest["ema_21"]: return None
    if latest["pct_from_52w_high"] < -5: return None
    if latest["close"] < latest["open"]: return None
    vol_ratio = latest["volume"] / (latest["vol_sma_20"] + 1)
    if vol_ratio < 1.0: return None

    entry = latest["close"]
    sl = latest["low"]
    risk = entry - sl
    if risk <= 0: risk = latest["atr_14"]; sl = entry - risk
    if risk <= 0: return None

    return BacktestTrade(
        symbol=symbol, strategy="Last30Min_ATH", signal="BUY",
        entry_date=str(df.index[-1].date()), entry_price=round(entry, 2),
        stop_loss=round(sl, 2), target_1=round(entry + 1.5 * risk, 2),
        target_2=round(entry + 2.5 * risk, 2),
    )


# ============================================================================
# STATS COMPUTATION
# ============================================================================

def _compute_stats(trades: List[BacktestTrade], strategy: str,
                   symbol: str, lookback: int) -> BacktestResult:
    wins = [t for t in trades if t.net_pnl_pct > 0]
    losses = [t for t in trades if t.net_pnl_pct <= 0]

    # Gross stats
    total_pnl = sum(t.pnl_pct for t in trades)
    avg_win_gross = np.mean([t.pnl_pct for t in [t for t in trades if t.pnl_pct > 0]]) if any(t.pnl_pct > 0 for t in trades) else 0
    avg_loss_gross = np.mean([abs(t.pnl_pct) for t in [t for t in trades if t.pnl_pct <= 0]]) if any(t.pnl_pct <= 0 for t in trades) else 0

    gross_wins_sum = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_losses_sum = abs(sum(t.pnl_pct for t in trades if t.pnl_pct <= 0))
    pf = (gross_wins_sum / gross_losses_sum) if gross_losses_sum > 0 else 0

    # Net stats (after costs)
    net_pnl = sum(t.net_pnl_pct for t in trades)
    total_costs = sum(t.cost_pct for t in trades)
    net_wins_sum = sum(t.net_pnl_pct for t in wins)
    net_losses_sum = abs(sum(t.net_pnl_pct for t in losses))
    net_pf = (net_wins_sum / net_losses_sum) if net_losses_sum > 0 else 0

    # Equity curve + max drawdown (using NET P&L)
    equity = []
    running = 0
    peak = 0
    max_dd = 0
    for t in trades:
        running += t.net_pnl_pct
        peak = max(peak, running)
        dd = peak - running
        max_dd = max(max_dd, dd)
        equity.append({"date": t.exit_date, "equity": round(running, 2),
                        "drawdown": round(dd, 2), "gross": round(running + sum(tr.cost_pct for tr in trades[:trades.index(t)+1]), 2)})

    return BacktestResult(
        strategy=strategy, symbol=symbol,
        period=f"Last {lookback} trading days",
        total_trades=len(trades),
        wins=len(wins), losses=len(losses),
        win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
        total_pnl_pct=round(total_pnl, 2),
        net_pnl_pct=round(net_pnl, 2),
        avg_win_pct=round(avg_win_gross, 2),
        avg_loss_pct=round(avg_loss_gross, 2),
        profit_factor=round(pf, 2),
        net_profit_factor=round(net_pf, 2),
        max_drawdown_pct=round(max_dd, 2),
        expectancy_pct=round(total_pnl / len(trades), 2) if trades else 0,
        net_expectancy_pct=round(net_pnl / len(trades), 2) if trades else 0,
        avg_holding_days=round(np.mean([t.holding_days for t in trades]), 1),
        best_trade_pct=round(max(t.net_pnl_pct for t in trades), 2),
        worst_trade_pct=round(min(t.net_pnl_pct for t in trades), 2),
        total_costs_pct=round(total_costs, 2),
        trades=trades,
        equity_curve=equity,
    )


def backtest_multi_stock(symbols: List[str], data_dict: Dict[str, pd.DataFrame],
                         strategy: str, lookback: int = 500,
                         max_hold: int = 30,
                         cost_model: CostModel = None) -> Optional[BacktestResult]:
    all_trades = []
    for symbol in symbols:
        if symbol not in data_dict:
            continue
        result = backtest_strategy(data_dict[symbol], symbol, strategy, lookback,
                                   max_hold, cost_model)
        if result:
            all_trades.extend(result.trades)

    if not all_trades:
        return None

    all_trades.sort(key=lambda t: t.entry_date)
    return _compute_stats(all_trades, strategy, f"{len(symbols)} stocks", lookback)
