"""
NSE Elite Scanners v15 — Regime-Aware, SQI, RRG, Option Chain + IPO
====================================================================
1. Market Regime Engine (4 regimes: Accumulation/Expansion/Distribution/Panic)
2. Intraday scanners require Breeze API (no proxies)
3. RS > 70 filter on all long signals
4. RRG Sector Rotation — confidence +8 LEADING / -15 LAGGING
5. Multi-factor SQI confidence scoring
6. Strategy×Regime profit factor matrix (auto-block PF < 0.7)
7. Approaching Setup Watchlist (50-95% through setups)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, time
import logging

from data_engine import Indicators, now_ist

logger = logging.getLogger(__name__)


# ============================================================================
# RRG-LITE — Relative Rotation Graph (Sector Rotation)
# ============================================================================

def compute_sector_rrg(data_dict: Dict[str, pd.DataFrame],
                       nifty_df: pd.DataFrame,
                       sector_map: Dict[str, str] = None) -> Dict[str, Dict]:
    """
    RRG-Lite: For each sector, compute RS Ratio and RS Momentum vs Nifty.
    Classifies sectors into 4 quadrants:
    - LEADING: RS > Nifty AND accelerating → +8 confidence boost on signals
    - WEAKENING: RS > Nifty BUT decelerating → no adjustment
    - IMPROVING: RS < Nifty BUT accelerating → no adjustment
    - LAGGING: RS < Nifty AND decelerating → -15 confidence penalty on signals

    Returns: {sector: {"ratio", "momentum", "quadrant", "score", "stocks_count"}}
    """
    if nifty_df is None or len(nifty_df) < 63:
        return {}

    from stock_universe import get_sector

    sector_stocks: Dict[str, List[pd.DataFrame]] = {}
    for symbol, df in data_dict.items():
        if df is None or len(df) < 63:
            continue
        sector = (sector_map or {}).get(symbol, get_sector(symbol))
        if sector not in sector_stocks:
            sector_stocks[sector] = []
        sector_stocks[sector].append(df)

    nifty_ret_63 = (nifty_df["close"].iloc[-1] / nifty_df["close"].iloc[-63] - 1) * 100
    nifty_ret_21 = (nifty_df["close"].iloc[-1] / nifty_df["close"].iloc[-21] - 1) * 100

    results = {}
    for sector, stocks in sector_stocks.items():
        if len(stocks) < 2:
            continue

        ret_63_list = []
        ret_21_list = []
        for sdf in stocks:
            if len(sdf) >= 63:
                ret_63_list.append((sdf["close"].iloc[-1] / sdf["close"].iloc[-63] - 1) * 100)
            if len(sdf) >= 21:
                ret_21_list.append((sdf["close"].iloc[-1] / sdf["close"].iloc[-21] - 1) * 100)

        if not ret_63_list:
            continue

        avg_ret_63 = np.mean(ret_63_list)
        avg_ret_21 = np.mean(ret_21_list) if ret_21_list else avg_ret_63

        rs_ratio = 100 + (avg_ret_63 - nifty_ret_63) * 2
        rs_momentum = 100 + (avg_ret_21 - nifty_ret_21) * 3

        if rs_ratio >= 100 and rs_momentum >= 100:
            quadrant = "LEADING"
            score = 90
        elif rs_ratio >= 100 and rs_momentum < 100:
            quadrant = "WEAKENING"
            score = 65
        elif rs_ratio < 100 and rs_momentum < 100:
            quadrant = "LAGGING"
            score = 25
        else:
            quadrant = "IMPROVING"
            score = 55

        score += min(10, max(-10, (rs_ratio - 100) * 0.5))
        score += min(10, max(-10, (rs_momentum - 100) * 0.3))
        score = max(0, min(100, score))

        results[sector] = {
            "ratio": round(rs_ratio, 1),
            "momentum": round(rs_momentum, 1),
            "quadrant": quadrant,
            "score": round(score, 1),
            "stocks_count": len(stocks),
        }

    return results


@dataclass
class ScanResult:
    """A single scan result with trade parameters."""
    symbol: str
    strategy: str
    signal: str  # "BUY" or "SHORT"
    cmp: float
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward: float
    confidence: int  # 0-100
    reasons: List[str]
    entry_type: str = "AT CMP"
    sector: str = ""
    rs_rating: float = 50.0
    volume_ratio: float = 0.0
    rsi: float = 50.0
    hold_type: str = "Swing"
    timestamp: str = ""
    regime_fit: str = ""  # NEW: "IDEAL" / "OK" / "CAUTION" / "BLOCKED"
    weekly_aligned: bool = False  # NEW: multi-timeframe confirmation
    data_quality: str = "OK"  # NEW: "OK" / "STALE" / "INCOMPLETE"
    
    @property
    def risk_pct(self) -> float:
        if self.entry == 0: return 0
        return abs((self.entry - self.stop_loss) / self.entry * 100)
    
    @property
    def entry_gap_pct(self) -> float:
        if self.cmp == 0: return 0
        return abs((self.entry - self.cmp) / self.cmp * 100)


def compute_smart_targets(df: pd.DataFrame, entry: float, stop_loss: float,
                          signal: str = "BUY", lookback: int = 60) -> dict:
    """
    Compute stock-specific targets using price structure instead of fixed multiples.
    
    For BUY signals:
      T1 = nearest resistance (recent swing high within lookback)
      T2 = further resistance (52-week high or Fibonacci 2.618 extension)
    For SHORT signals:
      T1 = nearest support (recent swing low)
      T2 = further support (Fibonacci 1.618 extension downward)
    
    Falls back to ATR-based targets if no clear levels found.
    Returns: {"t1": float, "t2": float, "t3": float, "rr": float}
    """
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return {"t1": entry, "t2": entry, "t3": entry, "rr": 0}
    
    atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else risk
    high_52w = df["high"].max()
    low_52w = df["low"].min()
    
    if signal == "BUY":
        # Find swing highs in lookback period as resistance levels
        highs = df["high"].iloc[-lookback:]
        
        # T1: Nearest swing high above entry
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
               highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                if highs.iloc[i] > entry * 1.01:  # At least 1% above entry
                    resistance_levels.append(highs.iloc[i])
        
        # Also consider recent N-day highs
        for window in [20, 50]:
            if len(df) >= window:
                recent_high = df["high"].iloc[-window:].max()
                if recent_high > entry * 1.01:
                    resistance_levels.append(recent_high)
        
        # 52-week high is always a key level
        if high_52w > entry * 1.01:
            resistance_levels.append(high_52w)
        
        # Fibonacci extensions from recent swing (low to entry)
        swing_low = df["low"].iloc[-lookback:].min()
        swing_range = entry - swing_low
        if swing_range > 0:
            fib_1618 = entry + swing_range * 0.618
            fib_2618 = entry + swing_range * 1.618
            resistance_levels.extend([fib_1618, fib_2618])
        
        # Sort and pick
        resistance_levels = sorted(set(r for r in resistance_levels if r > entry * 1.005))
        
        if len(resistance_levels) >= 3:
            t1 = round(resistance_levels[0], 2)
            t2 = round(resistance_levels[1], 2)
            t3 = round(resistance_levels[2], 2)
        elif len(resistance_levels) == 2:
            t1 = round(resistance_levels[0], 2)
            t2 = round(resistance_levels[1], 2)
            t3 = round(entry + 4 * risk, 2)
        elif len(resistance_levels) == 1:
            t1 = round(resistance_levels[0], 2)
            t2 = round(entry + 2.5 * risk, 2)
            t3 = round(entry + 4 * risk, 2)
        else:
            # No clear resistance — use ATR-based
            t1 = round(entry + 1.5 * atr, 2)
            t2 = round(entry + 3 * atr, 2)
            t3 = round(entry + 5 * atr, 2)
        
        # Ensure minimum 1:1.5 R:R for T1
        min_t1 = entry + 1.5 * risk
        if t1 < min_t1:
            t1 = round(min_t1, 2)
        
        # Enforce T1 < T2 < T3
        if t2 <= t1:
            t2 = round(t1 + risk, 2)
        if t3 <= t2:
            t3 = round(t2 + risk, 2)
        
        rr = round((t2 - entry) / risk, 1) if risk > 0 else 0
        
    else:  # SHORT
        lows = df["low"].iloc[-lookback:]
        
        support_levels = []
        for i in range(2, len(lows) - 2):
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
               lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                if lows.iloc[i] < entry * 0.99:
                    support_levels.append(lows.iloc[i])
        
        for window in [20, 50]:
            if len(df) >= window:
                recent_low = df["low"].iloc[-window:].min()
                if recent_low < entry * 0.99:
                    support_levels.append(recent_low)
        
        # Fibonacci extensions downward
        swing_high = df["high"].iloc[-lookback:].max()
        swing_range = swing_high - entry
        if swing_range > 0:
            fib_1618 = entry - swing_range * 0.618
            fib_2618 = entry - swing_range * 1.618
            support_levels.extend([fib_1618, fib_2618])
        
        support_levels = sorted(set(s for s in support_levels if s < entry * 0.995), reverse=True)
        
        if len(support_levels) >= 3:
            t1 = round(support_levels[0], 2)
            t2 = round(support_levels[1], 2)
            t3 = round(support_levels[2], 2)
        elif len(support_levels) == 2:
            t1 = round(support_levels[0], 2)
            t2 = round(support_levels[1], 2)
            t3 = round(entry - 4 * risk, 2)
        elif len(support_levels) == 1:
            t1 = round(support_levels[0], 2)
            t2 = round(entry - 2.5 * risk, 2)
            t3 = round(entry - 4 * risk, 2)
        else:
            t1 = round(entry - 1.5 * atr, 2)
            t2 = round(entry - 3 * atr, 2)
            t3 = round(entry - 5 * atr, 2)
        
        min_t1 = entry - 1.5 * risk
        if t1 > min_t1:
            t1 = round(min_t1, 2)
        
        # Enforce T1 > T2 > T3 (descending for shorts)
        if t2 >= t1:
            t2 = round(t1 - risk, 2)
        if t3 >= t2:
            t3 = round(t2 - risk, 2)
        
        rr = round((entry - t2) / risk, 1) if risk > 0 else 0
    
    # Clamp R:R to reasonable range
    rr = max(1.0, min(rr, 8.0))
    
    return {"t1": t1, "t2": t2, "t3": t3, "rr": rr}


# ============================================================================
# STRATEGY PROFILES
# ============================================================================

STRATEGY_PROFILES = {
    "ORB": {
        "name": "Opening Range Breakout", "icon": "🔓", "type": "Intraday",
        "hold": "2-6 hours", "win_rate": 58.2, "expectancy": 0.47,
        "profit_factor": 1.72, "best_time": "9:30-10:30 AM",
        "time_window": (time(9,30), time(10,30)),
        "description": "Price breaks above first 15-min high with volume + VWAP",
        "requires_intraday": True,
        "ideal_regimes": ["EXPANSION"],
        "ok_regimes": ["ACCUMULATION"],
        "blocked_regimes": ["PANIC"],
    },
    "VWAP_Reclaim": {
        "name": "VWAP Reclaim", "icon": "📈", "type": "Intraday",
        "hold": "2-4 hours", "win_rate": 61.8, "expectancy": 0.39,
        "profit_factor": 1.84, "best_time": "10:00 AM - 12:30 PM",
        "time_window": (time(10,0), time(12,30)),
        "description": "Price reclaims VWAP from below with volume surge",
        "requires_intraday": True,
        "ideal_regimes": ["EXPANSION", "ACCUMULATION"],
        "ok_regimes": ["DISTRIBUTION"],
        "blocked_regimes": ["PANIC"],
    },
    "Last30Min_ATH": {
        "name": "Last 30 Min ATH", "icon": "⭐", "type": "Overnight",
        "hold": "Overnight", "win_rate": 68.4, "expectancy": 0.89,
        "profit_factor": 2.1, "best_time": "3:00-3:25 PM",
        "time_window": (time(15,0), time(15,25)),
        "description": "Stock at all-time high in last 30 min — overnight momentum",
        "requires_intraday": False,
        "ideal_regimes": ["EXPANSION"],
        "ok_regimes": ["ACCUMULATION"],
        "blocked_regimes": ["PANIC", "DISTRIBUTION"],
    },
    "Lunch_Low": {
        "name": "Lunch Low Buy", "icon": "🍽️", "type": "Intraday",
        "hold": "2-3 hours", "win_rate": 56.3, "expectancy": 0.28,
        "profit_factor": 1.52, "best_time": "12:30-1:30 PM",
        "time_window": (time(12,30), time(13,30)),
        "description": "Reversal buy at lunch-hour low — mean reversion",
        "requires_intraday": True,
        "ideal_regimes": ["EXPANSION", "ACCUMULATION"],
        "ok_regimes": ["DISTRIBUTION"],
        "blocked_regimes": ["PANIC"],
    },
    "VCP": {
        "name": "VCP (Minervini v2)", "icon": "🏆", "type": "Swing",
        "hold": "15-40 days", "win_rate": 45.0, "expectancy": 3.5,
        "profit_factor": 1.8, "best_time": "Post-Market (3:30 PM+)",
        "time_window": None,  # Any time
        "description": "Volatility Contraction Pattern — tight base, volume dry-up, breakout confirmation",
        "requires_intraday": False,
        "ideal_regimes": ["EXPANSION", "ACCUMULATION"],
        "ok_regimes": [],
        "blocked_regimes": ["PANIC", "DISTRIBUTION"],
    },
    "EMA21_Bounce": {
        "name": "21 EMA Bounce", "icon": "🔄", "type": "Swing",
        "hold": "5-15 days", "win_rate": 40.0, "expectancy": 2.14,
        "profit_factor": 1.96, "best_time": "Post-Market (3:30 PM+)",
        "time_window": None,
        "description": "Pullback to 21 EMA in strong uptrend — best long strategy by backtest",
        "requires_intraday": False,
        "ideal_regimes": ["EXPANSION"],
        "ok_regimes": ["ACCUMULATION"],
        "blocked_regimes": ["PANIC"],
    },
    "52WH_Breakout": {
        "name": "52-Week High Breakout", "icon": "🚀", "type": "Positional",
        "hold": "20-60 days", "win_rate": 35.0, "expectancy": 5.82,
        "profit_factor": 1.86, "best_time": "Post-Market (3:30 PM+)",
        "time_window": None,
        "description": "Breaking to new 52-week highs with volume — rare but high reward",
        "requires_intraday": False,
        "ideal_regimes": ["EXPANSION"],
        "ok_regimes": ["ACCUMULATION"],
        "blocked_regimes": ["PANIC", "DISTRIBUTION"],
    },
    "Failed_Breakout_Short": {
        "name": "Failed Breakout Short", "icon": "📉", "type": "Swing",
        "hold": "3-10 days", "win_rate": 24.0, "expectancy": 3.12,
        "profit_factor": 1.60, "best_time": "Post-Market (3:30 PM+)",
        "time_window": None,
        "description": "Attempted breakout that reversed — most consistent strategy across regimes",
        "requires_intraday": False,
        "ideal_regimes": ["DISTRIBUTION", "PANIC"],
        "ok_regimes": ["ACCUMULATION"],
        "blocked_regimes": [],  # Shorts work everywhere
    },
}


# ============================================================================
# MARKET REGIME ENGINE — The Brain
# ============================================================================

def detect_market_regime(nifty_df: pd.DataFrame, 
                         breadth_data: dict = None) -> dict:
    """
    4-regime market detection — RECALIBRATED v3.1
    
    Regimes:
    - ACCUMULATION: Base building, golden cross intact, mixed short-term
    - EXPANSION: Clear uptrend, high participation  
    - DISTRIBUTION: Topping out, breadth diverging, breakdown starting
    - PANIC: Vol spike, crash/correction, everything correlated down
    
    Scoring: max ~12 points. Thresholds calibrated against real Nifty states.
    """
    if nifty_df is None or len(nifty_df) < 200:
        return _default_regime()
    
    df = Indicators.enrich_dataframe(nifty_df)
    lat = df.iloc[-1]
    
    score = 0
    details = []
    
    # === TREND (max +/-5) ===
    
    # Golden/Death cross — BIG structural signal
    if lat["sma_50"] > lat["sma_200"]:
        score += 2
        details.append("✅ Golden Cross (50 SMA > 200 SMA) — long-term uptrend intact")
    else:
        score -= 2
        details.append("❌ Death Cross (50 SMA < 200 SMA) — structural breakdown")
    
    # Price vs 50 DMA — short-term health
    if lat["close"] > lat["sma_50"]:
        score += 1
        details.append("✅ Above 50 DMA — healthy short-term trend")
    else:
        score -= 1
        details.append("⚠️ Below 50 DMA — short-term weakness")
    
    # Price vs 21 EMA — immediate trend
    if lat["close"] > lat["ema_21"]:
        score += 1
        details.append("✅ Above 21 EMA — immediate trend positive")
    else:
        score -= 1
        details.append("❌ Below 21 EMA — near-term bearish")
    
    # === MOMENTUM (max +/-3) ===
    rsi = lat["rsi_14"]
    if rsi > 60:
        score += 2
        details.append(f"✅ RSI {rsi:.0f} — strong momentum")
    elif rsi >= 45:
        # Neutral zone: no penalty, no bonus
        details.append(f"➖ RSI {rsi:.0f} — neutral (45-60 = no edge)")
    elif rsi >= 35:
        score -= 1
        details.append(f"⚠️ RSI {rsi:.0f} — weakening momentum")
    else:
        score -= 2
        details.append(f"🔴 RSI {rsi:.0f} — oversold / capitulation zone")
    
    # MACD crossover
    if lat["macd"] > lat["macd_signal"]:
        score += 1
        details.append("✅ MACD bullish crossover")
    else:
        score -= 1
        details.append("❌ MACD bearish")
    
    # === POSITION vs 52W RANGE ===
    pct_from_high = lat["pct_from_52w_high"]
    if pct_from_high > -5:
        score += 1
        details.append(f"✅ Near 52W highs ({pct_from_high:.1f}%) — strength")
    elif pct_from_high > -10:
        details.append(f"➖ Pullback from highs ({pct_from_high:.1f}%)")
    elif pct_from_high > -20:
        score -= 1
        details.append(f"⚠️ Correction zone ({pct_from_high:.1f}% from highs)")
    else:
        score -= 2
        details.append(f"🔴 Deep correction ({pct_from_high:.1f}%)")
    
    # === VOLATILITY (panic detection — only severe spikes penalized) ===
    recent_atr = df["atr_14"].iloc[-5:].mean()
    older_atr = df["atr_14"].iloc[-25:-5].mean()
    vol_expansion = recent_atr / older_atr if older_atr > 0 else 1
    
    if vol_expansion > 2.0:
        score -= 3
        details.append(f"🔴 VOLATILITY SPIKE ({vol_expansion:.1f}x) — panic conditions")
    elif vol_expansion > 1.5:
        score -= 1
        details.append(f"⚠️ Volatility elevated ({vol_expansion:.1f}x) — caution")
    else:
        details.append(f"✅ Volatility normal ({vol_expansion:.1f}x)")
    
    # ADX (trend strength bonus)
    if lat["adx_14"] > 25:
        score += 1
        details.append(f"✅ ADX {lat['adx_14']:.0f} — decisive trend")
    
    # === BREADTH (when available) ===
    if breadth_data and breadth_data.get("total", 0) > 10:
        ad = breadth_data.get("ad_ratio", 1)
        pct_200 = breadth_data.get("above_200sma_pct", 50)
        if ad > 1.5 and pct_200 > 55:
            score += 1
            details.append(f"✅ Broad participation (A/D {ad:.1f}, {pct_200:.0f}% > 200 SMA)")
        elif ad < 0.7 and pct_200 < 35:
            score -= 2
            details.append(f"🔴 Breadth collapse (A/D {ad:.1f}, {pct_200:.0f}% > 200 SMA)")
        elif ad < 1:
            score -= 1
            details.append(f"⚠️ Weak breadth (A/D {ad:.1f})")
        else:
            details.append(f"➖ Breadth neutral (A/D {ad:.1f})")
    
    # === REGIME CLASSIFICATION ===
    max_score = 12
    
    # Panic override: severe vol spike + heavily negative
    if vol_expansion > 2.0 and score < -3:
        regime = "PANIC"
        regime_icon = "🔴"
        position_mult = 0.15
    elif score >= 6:
        regime = "EXPANSION"
        regime_icon = "🟢"
        position_mult = 1.0
    elif score >= 2:
        regime = "ACCUMULATION"
        regime_icon = "🟡"
        position_mult = 0.7
    elif score >= -2:
        regime = "DISTRIBUTION"
        regime_icon = "🟠"
        position_mult = 0.4
    else:
        regime = "PANIC"
        regime_icon = "🔴"
        position_mult = 0.15
    
    # Build allowed/blocked strategy lists
    allowed, caution, blocked = [], [], []
    for key, prof in STRATEGY_PROFILES.items():
        if regime in prof.get("ideal_regimes", []):
            allowed.append(key)
        elif regime in prof.get("ok_regimes", []):
            caution.append(key)
        elif regime in prof.get("blocked_regimes", []):
            blocked.append(key)
        else:
            caution.append(key)
    
    return {
        "regime": regime,
        "regime_display": f"{regime_icon} {regime}",
        "score": score,
        "max_score": max_score,
        "details": details,
        "position_multiplier": position_mult,
        "allowed_strategies": allowed,
        "caution_strategies": caution,
        "blocked_strategies": blocked,
        "nifty_close": round(lat["close"], 2),
        "nifty_rsi": round(rsi, 1),
        "nifty_pct_52wh": round(pct_from_high, 1),
        "vol_expansion": round(vol_expansion, 1),
    }


def _default_regime():
    return {
        "regime": "UNKNOWN", "regime_display": "⚪ UNKNOWN",
        "score": 0, "max_score": 12,
        "details": ["⚠️ Insufficient data for regime detection"],
        "position_multiplier": 0.3,
        "allowed_strategies": [], "caution_strategies": list(STRATEGY_PROFILES.keys()),
        "blocked_strategies": [],
        "nifty_close": 0, "nifty_rsi": 0, "nifty_pct_52wh": 0, "vol_expansion": 1.0,
    }


# Keep backward compatibility
def check_market_health(nifty_df):
    return detect_market_regime(nifty_df)


# ============================================================================
# STRATEGY SCANNERS — Daily data (honest, no proxies)
# ============================================================================

def scan_vcp(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    """
    VCP (Volatility Contraction Pattern) — Minervini style, REWRITTEN v2.
    
    KEY CHANGES from broken v1:
    1. Uses std_10/std_50 volatility compression ratio instead of fixed windows
    2. Requires actual breakout confirmation (close > pivot on volume)
    3. Checks down-day volume < up-day volume (accumulation)
    4. Tighter stop loss (last contraction low, not 10-day low minus ATR)
    5. Stricter contraction threshold (< 0.5 instead of 0.65)
    6. Range depth check (base not too deep)
    """
    if df is None or len(df) < 200:
        return None
    
    latest = df.iloc[-1]
    
    # ── Stage 2 Uptrend (Minervini Trend Template) ──
    if latest["close"] < latest["sma_50"] or latest["close"] < latest["sma_200"]:
        return None
    if latest["sma_50"] < latest["sma_200"]:
        return None
    
    # Must be within 25% of 52W high and > 30% above 52W low
    if latest["pct_from_52w_high"] < -25:
        return None
    pct_above_low = (latest["close"] - latest["low_52w"]) / latest["low_52w"] * 100
    if pct_above_low < 30:
        return None
    
    # ── Volatility Compression Check (CORE FIX) ──
    # std_10/std_50 < 0.50 = strong contraction (Gemini's recommendation)
    from data_engine import Indicators
    vol_ratio = Indicators.volatility_compression_ratio(df, short=10, long=50)
    if vol_ratio > 0.50:
        return None  # Not compressed enough
    
    # ── Range Depth Check — base shouldn't be too deep ──
    # Recent 25-day range as % of price should be < 15%
    recent_high = df["high"].iloc[-25:].max()
    recent_low = df["low"].iloc[-25:].min()
    range_depth = (recent_high - recent_low) / recent_high
    if range_depth > 0.15:
        return None  # Base too loose — not a tight VCP
    
    # ── Volume Dry-Up in Contraction ──
    avg_vol_5d = df["volume"].iloc[-5:].mean()
    avg_vol_50d = df["vol_sma_50"].iloc[-1] if "vol_sma_50" in df.columns else df["volume"].iloc[-50:].mean()
    vol_dryup = avg_vol_5d / (avg_vol_50d + 1) 
    if vol_dryup > 0.70:
        return None  # Volume not dried up enough (want < 70% of 50d avg)
    
    # ── Down-Day Volume < Up-Day Volume (Accumulation, not Distribution) ──
    vol_dd_ratio = Indicators.volume_down_day_ratio(df, lookback=20)
    if vol_dd_ratio > 1.2:
        return None  # More volume on down days = distribution, not VCP
    
    # ── Breakout Confirmation (CRITICAL FIX) ──
    # Entry ONLY when close > pivot on above-average volume
    pivot = round(recent_high, 2)
    cmp = round(latest["close"], 2)
    
    # Check if breakout happened TODAY
    breakout_confirmed = False
    if cmp >= pivot * 0.998:  # Within 0.2% of pivot (allow slight tolerance)
        today_vol_ratio = latest["volume"] / (avg_vol_50d + 1)
        if today_vol_ratio >= 1.3:  # Volume 30%+ above average on breakout day
            breakout_confirmed = True
    
    if not breakout_confirmed:
        # If not breaking out yet, only show as "ABOVE pivot" entry
        # The old code entered BEFORE breakout — that's the main reason VCP was losing money
        if cmp < pivot * 0.97:
            return None  # Too far from pivot, not a valid VCP setup
    
    # ── Build Confidence Score ──
    reasons = []
    confidence = 45
    
    reasons.append("Stage 2 uptrend confirmed (Price > 50 SMA > 200 SMA)")
    confidence += 8
    
    reasons.append(f"Volatility compressed to {vol_ratio:.2f}x (std₁₀/std₅₀ < 0.50)")
    confidence += 8 if vol_ratio < 0.35 else 5
    
    reasons.append(f"Volume dried up to {vol_dryup:.1f}x 50d avg — institutional pause")
    confidence += 5 if vol_dryup < 0.5 else 3
    
    reasons.append(f"Down-day vol ratio: {vol_dd_ratio:.2f} — {'accumulation' if vol_dd_ratio < 0.9 else 'neutral'}")
    confidence += 5 if vol_dd_ratio < 0.8 else 2
    
    reasons.append(f"Range depth: {range_depth*100:.1f}% — tight base")
    confidence += 3 if range_depth < 0.10 else 1
    
    if latest["pct_from_52w_high"] > -10:
        reasons.append(f"{latest['pct_from_52w_high']:.1f}% from 52W high — strong stock")
        confidence += 7
    elif latest["pct_from_52w_high"] > -15:
        confidence += 4
    
    if breakout_confirmed:
        today_vol_ratio = latest["volume"] / (avg_vol_50d + 1)
        reasons.append(f"🔥 BREAKOUT confirmed on {today_vol_ratio:.1f}x volume!")
        confidence += 10
    else:
        reasons.append(f"Approaching pivot ₹{pivot:,.0f} — wait for volume breakout")
    
    if latest["adx_14"] > 20:
        reasons.append(f"ADX {latest['adx_14']:.0f} — trending market")
        confidence += 3
    
    confidence = min(confidence, 92)
    
    # ── Entry and Risk ──
    if breakout_confirmed:
        entry = cmp
        entry_type = "AT CMP"
    else:
        entry = pivot
        entry_type = f"ABOVE ₹{pivot:,.0f}"
    
    # TIGHTER stop loss: last contraction low (not 10-day low minus ATR)
    # Find the lowest low in the contraction period
    contraction_low = df["low"].iloc[-15:].min()  # 15-day contraction low
    stop_loss = round(contraction_low * 0.995, 2)  # 0.5% below contraction low
    
    risk = entry - stop_loss
    if risk <= 0 or (risk / entry * 100) > 8:
        return None  # Invalid risk or stop too wide (max 8%)
    
    targets = compute_smart_targets(df, entry, stop_loss, "BUY")
    
    return ScanResult(
        symbol=symbol, strategy="VCP", signal="BUY",
        cmp=cmp, entry=entry, stop_loss=stop_loss,
        target_1=targets["t1"],
        target_2=targets["t2"],
        target_3=targets["t3"],
        risk_reward=targets["rr"],
        confidence=confidence, reasons=reasons,
        entry_type=entry_type,
        volume_ratio=round(vol_dryup, 2),
        rsi=round(latest["rsi_14"], 1),
        hold_type="Swing (15-40d)",
        timestamp=str(df.index[-1].date()),
    )


def scan_ema21_bounce(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    """21 EMA Bounce — pullback entry in strong uptrend."""
    if df is None or len(df) < 60:
        return None
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Uptrend: above 50 & 200 SMA
    if latest["close"] < latest["sma_50"]:
        return None
    if latest["sma_50"] < latest["sma_200"]:
        return None
    
    # Pulled back to 21 EMA (within 2% of it)
    ema_dist = (latest["close"] - latest["ema_21"]) / latest["ema_21"] * 100
    if ema_dist > 3 or ema_dist < -2:
        return None
    
    # Bounce signal: low touched or crossed below 21 EMA, closed above
    if latest["low"] > latest["ema_21"] * 1.01:
        return None  # Didn't pull back enough
    if latest["close"] < latest["ema_21"]:
        return None  # Didn't reclaim
    
    # Bullish candle
    if latest["close"] < latest["open"]:
        return None
    
    # Volume
    vol_ratio = latest["volume"] / (latest["vol_sma_20"] + 1)
    
    reasons = []
    confidence = 48
    
    reasons.append("Pullback to 21 EMA in Stage 2 uptrend — classic buy zone")
    confidence += 10
    
    reasons.append(f"Bounced off 21 EMA (distance: {ema_dist:.1f}%)")
    confidence += 5
    
    if vol_ratio > 1.3:
        reasons.append(f"Volume surge on bounce: {vol_ratio:.1f}x — confirmation")
        confidence += 7
    else:
        reasons.append(f"Volume: {vol_ratio:.1f}x — moderate")
        confidence += 2
    
    # RSI between 40-65 is ideal (not overbought, not dead)
    rsi = latest["rsi_14"]
    if 45 <= rsi <= 60:
        reasons.append(f"RSI {rsi:.0f} — ideal pullback zone")
        confidence += 5
    
    confidence = min(confidence, 85)
    
    cmp = round(latest["close"], 2)
    entry = cmp
    atr = latest["atr_14"]
    stop_loss = round(min(latest["low"], latest["ema_21"]) - 0.5 * atr, 2)
    risk = entry - stop_loss
    if risk <= 0:
        return None
    
    targets = compute_smart_targets(df, entry, stop_loss, "BUY")
    
    return ScanResult(
        symbol=symbol, strategy="EMA21_Bounce", signal="BUY",
        cmp=cmp, entry=entry, stop_loss=stop_loss,
        target_1=targets["t1"],
        target_2=targets["t2"],
        target_3=targets["t3"],
        risk_reward=targets["rr"],
        confidence=confidence, reasons=reasons,
        entry_type="AT CMP",
        volume_ratio=round(vol_ratio, 1),
        rsi=round(rsi, 1),
        hold_type="Swing (5-15d)",
        timestamp=str(df.index[-1].date()),
    )


def scan_52wh_breakout(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    """52-Week High Breakout with volume."""
    if df is None or len(df) < 200:
        return None
    
    latest = df.iloc[-1]
    
    # Must be at or very near 52W high
    if latest["pct_from_52w_high"] < -3:
        return None
    
    # Breakout above recent resistance
    prev_high = df["high"].iloc[-20:-1].max()
    if latest["close"] <= prev_high:
        return None
    
    # Trend filter
    if latest["close"] < latest["sma_50"]:
        return None
    if latest["sma_50"] < latest["sma_200"]:
        return None
    
    # Volume surge
    vol_ratio = latest["volume"] / (latest["vol_sma_20"] + 1)
    if vol_ratio < 1.5:
        return None
    
    reasons = []
    confidence = 50
    
    reasons.append(f"Breaking 52-week high! ({latest['pct_from_52w_high']:.1f}% from peak)")
    confidence += 12
    
    reasons.append(f"Volume {vol_ratio:.1f}x average — institutional interest")
    confidence += 5 if vol_ratio < 2.5 else 10
    
    # ADX trending
    if latest["adx_14"] > 25:
        reasons.append(f"ADX {latest['adx_14']:.0f} — strong trend")
        confidence += 5
    
    confidence = min(confidence, 88)
    
    cmp = round(latest["close"], 2)
    entry = cmp
    atr = latest["atr_14"]
    stop_loss = round(prev_high - 0.5 * atr, 2)
    risk = entry - stop_loss
    if risk <= 0:
        stop_loss = round(entry - 1.5 * atr, 2)
        risk = entry - stop_loss
    if risk <= 0:
        return None
    
    _tgt = compute_smart_targets(df, entry, stop_loss, "BUY")
    return ScanResult(
        symbol=symbol, strategy="52WH_Breakout", signal="BUY",
        cmp=cmp, entry=entry, stop_loss=stop_loss,
        target_1=_tgt["t1"],
        target_2=_tgt["t2"],
        target_3=_tgt["t3"],
        risk_reward=_tgt["rr"],
        confidence=confidence, reasons=reasons,
        entry_type="AT CMP",
        volume_ratio=round(vol_ratio, 1),
        rsi=round(latest["rsi_14"], 1),
        hold_type="Positional (20-60d)",
        timestamp=str(df.index[-1].date()),
    )


def scan_last30min_ath(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    """Last 30 Min ATH — stock near day's high at close."""
    if df is None or len(df) < 50:
        return None
    
    latest = df.iloc[-1]
    
    # Close must be within 1% of day's high (closing at highs)
    if latest["high"] == 0:
        return None
    close_vs_high = (latest["high"] - latest["close"]) / latest["high"] * 100
    if close_vs_high > 1.0:
        return None
    
    # Must be in uptrend
    if latest["close"] < latest["ema_21"]:
        return None
    
    # Within 5% of 52W high
    if latest["pct_from_52w_high"] < -5:
        return None
    
    # Bullish day
    if latest["close"] < latest["open"]:
        return None
    
    # Volume
    vol_ratio = latest["volume"] / (latest["vol_sma_20"] + 1)
    if vol_ratio < 1.0:
        return None
    
    reasons = []
    confidence = 52
    
    reasons.append(f"Closing at day's high (within {close_vs_high:.1f}%) — strong demand")
    confidence += 10
    
    reasons.append(f"Near 52W high ({latest['pct_from_52w_high']:.1f}%) — momentum stock")
    confidence += 5 if latest["pct_from_52w_high"] > -2 else 3
    
    reasons.append(f"Volume: {vol_ratio:.1f}x — institutional activity")
    if vol_ratio > 1.5:
        confidence += 7
    else:
        confidence += 3
    
    confidence = min(confidence, 82)
    
    cmp = round(latest["close"], 2)
    entry = cmp
    atr = latest["atr_14"]
    stop_loss = round(latest["low"], 2)
    risk = entry - stop_loss
    if risk <= 0:
        stop_loss = round(entry - atr, 2)
        risk = entry - stop_loss
    if risk <= 0:
        return None
    
    _tgt = compute_smart_targets(df, entry, stop_loss, "BUY")
    return ScanResult(
        symbol=symbol, strategy="Last30Min_ATH", signal="BUY",
        cmp=cmp, entry=entry, stop_loss=stop_loss,
        target_1=_tgt["t1"],
        target_2=_tgt["t2"],
        target_3=_tgt["t3"],
        risk_reward=_tgt["rr"],
        confidence=confidence, reasons=reasons,
        entry_type="AT CMP",
        volume_ratio=round(vol_ratio, 1),
        rsi=round(latest["rsi_14"], 1),
        hold_type="Overnight → Swing",
        timestamp=str(df.index[-1].date()),
    )


def scan_failed_breakout_short(df: pd.DataFrame, symbol: str) -> Optional[ScanResult]:
    """Failed Breakout Short — trap reversal."""
    if df is None or len(df) < 60:
        return None
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Previous day attempted breakout (high > recent resistance)
    resistance = df["high"].iloc[-20:-2].max()
    if prev["high"] < resistance:
        return None
    
    # Today failed (closed below resistance)
    if latest["close"] >= resistance:
        return None
    
    # Bearish close
    if latest["close"] >= latest["open"]:
        return None
    
    # Close below previous close
    if latest["close"] >= prev["close"]:
        return None
    
    # Volume on reversal
    vol_ratio = latest["volume"] / (latest["vol_sma_20"] + 1)
    if vol_ratio < 1.0:
        return None
    
    reasons = []
    confidence = 50
    
    reasons.append("Failed breakout — tried to break resistance but reversed sharply")
    confidence += 12
    
    reasons.append(f"Bearish reversal with {vol_ratio:.1f}x volume — trapped longs")
    confidence += 5 if vol_ratio < 2 else 8
    
    rsi = latest["rsi_14"]
    if rsi > 65:
        reasons.append(f"RSI {rsi:.0f} — overbought, adding to short thesis")
        confidence += 5
    
    confidence = min(confidence, 82)
    
    cmp = round(latest["close"], 2)
    entry = cmp
    stop_loss = round(max(latest["high"], prev["high"]) + latest["atr_14"] * 0.3, 2)
    risk = stop_loss - entry
    if risk <= 0:
        return None
    
    _tgt = compute_smart_targets(df, entry, stop_loss, "SHORT")
    return ScanResult(
        symbol=symbol, strategy="Failed_Breakout_Short", signal="SHORT",
        cmp=cmp, entry=entry, stop_loss=stop_loss,
        target_1=_tgt["t1"],
        target_2=_tgt["t2"],
        target_3=_tgt["t3"],
        risk_reward=_tgt["rr"],
        confidence=confidence, reasons=reasons,
        entry_type="AT CMP",
        volume_ratio=round(vol_ratio, 1),
        rsi=round(rsi, 1),
        hold_type="Short Swing (3-10d)",
        timestamp=str(df.index[-1].date()),
    )


# ============================================================================
# APPROACHING SETUP WATCHLIST — Stocks near (but not at) setups
# ============================================================================

@dataclass
class ApproachingSetup:
    """A stock approaching a scanner's trigger criteria."""
    symbol: str
    strategy: str
    progress_pct: float  # How close to triggering (0-100)
    description: str
    cmp: float
    trigger_level: float  # Price level needed to trigger
    rs_rating: float = 50.0
    sector: str = ""

def scan_approaching_setups(df: pd.DataFrame, symbol: str,
                             nifty_df: pd.DataFrame = None) -> List[ApproachingSetup]:
    """
    Find stocks 70-95% through a setup — not triggering yet, but on watchlist.
    Solves the "zero signals" problem by showing what's forming.
    """
    setups = []
    if df is None or len(df) < 200:
        return setups
    
    from data_engine import Indicators
    enriched = Indicators.enrich_dataframe(df)
    latest = enriched.iloc[-1]
    cmp = round(latest["close"], 2)
    
    rs = 50.0
    if nifty_df is not None:
        enriched_nifty = Indicators.enrich_dataframe(nifty_df)
        rs = Indicators.relative_strength(enriched, enriched_nifty)
    
    # ── Approaching VCP ──
    # Stage 2 uptrend but volatility not yet fully compressed
    if (latest["close"] > latest["sma_50"] > latest["sma_200"] and
        latest["pct_from_52w_high"] > -25):
        vol_ratio = Indicators.volatility_compression_ratio(enriched, 10, 50)
        if 0.50 <= vol_ratio <= 0.75:  # Contracting but not tight enough yet
            progress = max(0, min(100, (0.75 - vol_ratio) / 0.25 * 100))
            pivot = round(enriched["high"].iloc[-25:].max(), 2)
            setups.append(ApproachingSetup(
                symbol=symbol, strategy="VCP",
                progress_pct=round(progress, 0),
                description=f"Volatility contracting ({vol_ratio:.2f}x) — needs < 0.50x",
                cmp=cmp, trigger_level=pivot, rs_rating=rs,
            ))
    
    # ── Approaching EMA21 Bounce ──
    # In uptrend, pulling back toward 21 EMA but not yet touching
    if (latest["close"] > latest["sma_50"] and 
        latest["sma_50"] > latest["sma_200"]):
        ema_dist = (latest["close"] - latest["ema_21"]) / latest["ema_21"] * 100
        if 1.5 < ema_dist <= 5.0:  # Getting close but not touching
            progress = max(0, min(100, (5.0 - ema_dist) / 3.5 * 100))
            setups.append(ApproachingSetup(
                symbol=symbol, strategy="EMA21_Bounce",
                progress_pct=round(progress, 0),
                description=f"Pulling back to 21 EMA ({ema_dist:.1f}% above) — needs touch",
                cmp=cmp, trigger_level=round(latest["ema_21"], 2), rs_rating=rs,
            ))
    
    # ── Approaching 52WH Breakout ──
    # Within 3% of 52-week high
    if -5 < latest["pct_from_52w_high"] < -1:
        progress = max(0, min(100, (5 + latest["pct_from_52w_high"]) / 4 * 100))
        setups.append(ApproachingSetup(
            symbol=symbol, strategy="52WH_Breakout",
            progress_pct=round(progress, 0),
            description=f"{latest['pct_from_52w_high']:.1f}% from 52W high — approaching breakout",
            cmp=cmp, trigger_level=round(latest["high_52w"], 2), rs_rating=rs,
        ))
    
    return setups


# ============================================================================
# INTRADAY SCANNERS — REQUIRE BREEZE (No Proxy Fallback)
# ============================================================================

def scan_orb_intraday(df: pd.DataFrame, symbol: str, 
                      has_intraday: bool = False) -> Optional[ScanResult]:
    """ORB — REQUIRES real intraday data. Returns None without Breeze."""
    if not has_intraday:
        return None  # DO NOT GUESS from daily candles
    # Real ORB logic would use 15-min candle data from Breeze
    return None


def scan_vwap_reclaim_intraday(df: pd.DataFrame, symbol: str,
                                has_intraday: bool = False) -> Optional[ScanResult]:
    """VWAP Reclaim — REQUIRES real intraday data."""
    if not has_intraday:
        return None
    return None


def scan_lunch_low_intraday(df: pd.DataFrame, symbol: str,
                             has_intraday: bool = False) -> Optional[ScanResult]:
    """Lunch Low — REQUIRES real intraday data."""
    if not has_intraday:
        return None
    return None


# ============================================================================
# STRATEGY HEALTH TRACKER — Auto-dim weak strategies
# ============================================================================

class StrategyHealthTracker:
    """
    Track rolling profit factor per strategy.
    If PF drops below 0.8 over last 20 closed trades, flag as WEAK.
    """
    
    def __init__(self):
        self._rolling_pf = {}  # strategy -> rolling PF
        self._trade_counts = {}  # strategy -> count of closed trades
    
    def update_from_tracker(self, tracker_df):
        """Compute rolling PF from signal tracker data."""
        if tracker_df is None or tracker_df.empty:
            return
        
        closed = tracker_df[tracker_df["Status"].isin(["TARGET", "STOPPED"])]
        if closed.empty:
            return
        
        for strategy in closed["Strategy"].unique():
            strat_trades = closed[closed["Strategy"] == strategy].tail(20)
            pnl = pd.to_numeric(strat_trades.get("PnL_Pct", pd.Series()), errors="coerce")
            
            wins = pnl[pnl > 0].sum()
            losses = abs(pnl[pnl < 0].sum())
            
            if losses > 0:
                self._rolling_pf[strategy] = round(wins / losses, 2)
            elif wins > 0:
                self._rolling_pf[strategy] = 3.0  # Cap at 3.0 for display
            
            self._trade_counts[strategy] = len(strat_trades)
    
    def get_health(self, strategy: str) -> dict:
        """Get health status for a strategy."""
        pf = self._rolling_pf.get(strategy)
        count = self._trade_counts.get(strategy, 0)
        
        if pf is None or count < 5:
            return {"status": "UNKNOWN", "icon": "❓", "pf": None, "trades": count,
                    "sqi_weight": 1.0, "warning": ""}
        
        if pf >= 1.5:
            return {"status": "STRONG", "icon": "💪", "pf": pf, "trades": count,
                    "sqi_weight": 1.0, "warning": ""}
        elif pf >= 1.0:
            return {"status": "OK", "icon": "✅", "pf": pf, "trades": count,
                    "sqi_weight": 1.0, "warning": ""}
        elif pf >= 0.8:
            return {"status": "WEAK", "icon": "⚠️", "pf": pf, "trades": count,
                    "sqi_weight": 0.7, "warning": f"⚠️ {strategy} PF={pf:.2f} — use smaller size"}
        else:
            return {"status": "FAILING", "icon": "🔴", "pf": pf, "trades": count,
                    "sqi_weight": 0.5, "warning": f"🔴 {strategy} PF={pf:.2f} — strategy failing, consider skipping"}

# Global instance
strategy_health = StrategyHealthTracker()


# ============================================================================
# MASTER SCANNER — Regime-filtered, RS-filtered, SQI-ranked
# ============================================================================

# Only daily scanners (honest scanners that work with yfinance)
DAILY_SCANNERS = {
    "VCP": scan_vcp,
    "EMA21_Bounce": scan_ema21_bounce,
    "52WH_Breakout": scan_52wh_breakout,
    "Last30Min_ATH": scan_last30min_ath,
    "Failed_Breakout_Short": scan_failed_breakout_short,
}

# Intraday scanners (require Breeze)
INTRADAY_SCANNERS = {
    "ORB": scan_orb_intraday,
    "VWAP_Reclaim": scan_vwap_reclaim_intraday,
    "Lunch_Low": scan_lunch_low_intraday,
}

ALL_SCANNERS = {**DAILY_SCANNERS, **INTRADAY_SCANNERS}
INTRADAY_PROXY_SCANNERS = []  # KILLED — no more fake proxies


def run_scanner(scanner_name: str, data_dict: Dict[str, pd.DataFrame],
                nifty_df: pd.DataFrame = None,
                regime: dict = None,
                has_intraday: bool = False,
                sector_rankings: Dict[str, float] = None,
                min_rs: float = 0,
                compute_sqi_flag: bool = True) -> List[ScanResult]:
    """
    Run a single scanner with regime/RS/sector filtering and SQI ranking.
    
    v2 changes:
    - Strategy×Regime profit factor gating (blocks strategies with PF < 1.0 in current regime)
    - Signal Quality Index computation and sorting
    - Auto-dim integration from strategy health tracker
    - RS acceleration computation
    """
    scanner_func = ALL_SCANNERS.get(scanner_name) or DAILY_SCANNERS.get(scanner_name)
    if not scanner_func:
        return []
    
    current_regime = regime.get("regime", "UNKNOWN") if regime else "UNKNOWN"
    
    # ── Strategy × Regime Profit Factor Gate ──
    # Block strategies with PF < 1.0 in current regime
    from signal_quality import is_strategy_allowed, compute_sqi as _compute_sqi
    if current_regime != "UNKNOWN" and not is_strategy_allowed(scanner_name, current_regime, min_pf=0.7):
        logger.info(f"Scanner {scanner_name} GATED by {current_regime} regime (PF < 0.7)")
        return []
    
    # Check if this strategy is blocked by current regime
    if regime:
        blocked = regime.get("blocked_strategies", [])
        if scanner_name in blocked:
            logger.info(f"Scanner {scanner_name} BLOCKED by {current_regime} regime")
            return []
    
    # Determine regime fit for this scanner
    def get_regime_fit(scanner_name: str, regime: dict) -> str:
        if not regime: return "OK"
        if scanner_name in regime.get("allowed_strategies", []): return "IDEAL"
        if scanner_name in regime.get("caution_strategies", []): return "CAUTION"
        if scanner_name in regime.get("blocked_strategies", []): return "BLOCKED"
        return "OK"
    
    # Get strategy health for auto-dim
    health = strategy_health.get_health(scanner_name)
    
    results = []
    for symbol, df in data_dict.items():
        try:
            enriched = Indicators.enrich_dataframe(df)
            
            # Call scanner
            if scanner_name in INTRADAY_SCANNERS:
                result = scanner_func(enriched, symbol, has_intraday=has_intraday)
            else:
                result = scanner_func(enriched, symbol)
            
            if result is None:
                continue
            
            # Compute RS rating
            rs_accel = 0.0
            if nifty_df is not None:
                enriched_nifty = Indicators.enrich_dataframe(nifty_df)
                result.rs_rating = Indicators.relative_strength(enriched, enriched_nifty)
                rs_accel = Indicators.rs_acceleration(enriched, enriched_nifty, lookback=21)
            
            # RS FILTER: Long signals require RS > min_rs
            if result.signal == "BUY" and min_rs > 0 and result.rs_rating < min_rs:
                continue
            
            # SECTOR FILTER: Only buy in top-performing sectors
            from stock_universe import get_sector
            result.sector = get_sector(symbol)
            if sector_rankings and result.signal == "BUY":
                sector_rank = sector_rankings.get(result.sector, 50)
                if sector_rank < 30:  # Bottom 30% sectors
                    result.confidence = max(result.confidence - 15, 20)
                    result.reasons.append(f"⚠️ Weak sector ({result.sector}) — reduced confidence")
                elif sector_rank > 70:  # Top 30% sectors
                    result.confidence = min(result.confidence + 5, 95)
                    result.reasons.append(f"✅ Strong sector tailwind ({result.sector})")
            
            # Regime fit tag
            fit = get_regime_fit(scanner_name, regime)
            result.regime_fit = fit
            if fit == "IDEAL":
                result.confidence = min(result.confidence + 5, 95)
                result.reasons.append(f"✅ Regime: {current_regime} — ideal for this strategy")
            elif fit == "CAUTION":
                result.confidence = max(result.confidence - 10, 25)
                result.reasons.append(f"⚠️ Regime: {current_regime} — use smaller size")
            
            # ── Compute SQI ──
            if compute_sqi_flag:
                vol_compression = Indicators.volatility_compression_ratio(enriched, 10, 50)
                vol_dd = Indicators.volume_down_day_ratio(enriched, 20)
                
                sqi_result = _compute_sqi(
                    strategy=scanner_name,
                    regime=current_regime,
                    rs_rating=result.rs_rating,
                    rs_acceleration=rs_accel,
                    vol_compression_ratio=vol_compression,
                    volume_ratio=result.volume_ratio,
                    confidence=result.confidence,
                    vol_down_day_ratio=vol_dd,
                    weekly_aligned=result.weekly_aligned,
                    rolling_pf=health.get("pf"),
                )
                
                # Attach SQI to result (using dynamic attributes)
                result.sqi = sqi_result.sqi
                result.sqi_grade = sqi_result.grade
                result.sqi_icon = sqi_result.grade_icon
                result.sqi_breakdown = sqi_result.breakdown
                
                # Auto-dim: add warning if strategy is weak
                if health.get("warning"):
                    result.reasons.append(health["warning"])
                    # Reduce SQI by strategy health weight
                    result.sqi = round(result.sqi * health.get("sqi_weight", 1.0), 1)
                
                # RS acceleration note
                if rs_accel > 0.3:
                    result.reasons.append(f"📈 RS accelerating (+{rs_accel:.2f}/day)")
                elif rs_accel < -0.3:
                    result.reasons.append(f"📉 RS decelerating ({rs_accel:.2f}/day)")
            
            results.append(result)
            
        except Exception as e:
            continue
    
    # Sort by SQI (if computed) then confidence
    if compute_sqi_flag:
        results.sort(key=lambda x: getattr(x, 'sqi', x.confidence), reverse=True)
    else:
        results.sort(key=lambda x: x.confidence, reverse=True)
    return results


def run_all_scanners(data_dict: Dict[str, pd.DataFrame],
                     nifty_df: pd.DataFrame = None,
                     daily_only: bool = True,
                     regime: dict = None,
                     has_intraday: bool = False,
                     sector_rankings: Dict[str, float] = None,
                     min_rs: float = 70,
                     compute_sqi_flag: bool = True) -> Dict[str, List[ScanResult]]:
    """Run all scanners with regime/RS/sector filters and SQI ranking."""
    results = {}
    scanners = DAILY_SCANNERS if daily_only else ALL_SCANNERS
    
    for name in scanners:
        res = run_scanner(name, data_dict, nifty_df, regime, 
                          has_intraday, sector_rankings, min_rs,
                          compute_sqi_flag)
        if res:
            results[name] = res
    
    return results


def collect_approaching_setups(data_dict: Dict[str, pd.DataFrame],
                                nifty_df: pd.DataFrame = None,
                                min_progress: float = 50) -> List[ApproachingSetup]:
    """
    Scan all stocks for approaching setups.
    Returns list sorted by progress % (most ready first).
    """
    all_setups = []
    for symbol, df in data_dict.items():
        try:
            enriched = Indicators.enrich_dataframe(df)
            setups = scan_approaching_setups(enriched, symbol, nifty_df)
            all_setups.extend([s for s in setups if s.progress_pct >= min_progress])
        except Exception:
            continue
    
    all_setups.sort(key=lambda x: x.progress_pct, reverse=True)
    return all_setups
