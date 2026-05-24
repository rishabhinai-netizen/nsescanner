"""
Intraday Scanners v1 — Real implementations using Breeze 5-min data
=====================================================================
Replaces the stub functions in scanners.py:
  - scan_orb_intraday          → 15-min Opening Range Breakout
  - scan_vwap_reclaim_intraday → VWAP reclaim with volume surge
  - scan_lunch_low_intraday    → Lunch-hour low reversal

All three require Breeze API for 5-min OHLCV data.
If has_intraday=False, they return None silently (no fake signals from daily bars).

Apply: replace lines 1120–1142 in scanners.py with these implementations,
and pass the breeze_engine instance into the scanner runner.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dtime
from typing import Optional, List, Dict


# ============================================================
# UTILITIES
# ============================================================

def _filter_by_session(intraday_df: pd.DataFrame, start: dtime, end: dtime) -> pd.DataFrame:
    """Return only bars whose timestamp falls within [start, end] IST."""
    if intraday_df is None or intraday_df.empty:
        return intraday_df
    idx_time = pd.to_datetime(intraday_df.index).time
    mask = [start <= t <= end for t in idx_time]
    return intraday_df.loc[mask]


def _today_bars(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """Return only bars from the most recent trading day in the dataframe."""
    if intraday_df is None or intraday_df.empty:
        return intraday_df
    idx = pd.to_datetime(intraday_df.index)
    last_date = idx[-1].date()
    mask = [d.date() == last_date for d in idx]
    return intraday_df.loc[mask]


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tpv = (tp * df["volume"]).cumsum()
    return cum_tpv / cum_vol.replace(0, 1)


def _build_intraday_result(symbol: str, strategy: str, signal: str,
                           cmp: float, entry: float, sl: float,
                           t1: float, t2: float, t3: float,
                           rr: float, confidence: int, reasons: List[str],
                           volume_ratio: float, rsi: float,
                           hold_type: str, timestamp: str):
    """Build a ScanResult — imported lazily to avoid circular deps."""
    from scanners import ScanResult
    return ScanResult(
        symbol=symbol, strategy=strategy, signal=signal,
        cmp=cmp, entry=entry, stop_loss=sl,
        target_1=t1, target_2=t2, target_3=t3,
        risk_reward=rr, confidence=confidence, reasons=reasons,
        entry_type="AT CMP",
        volume_ratio=volume_ratio, rsi=rsi,
        hold_type=hold_type, timestamp=timestamp,
    )


# ============================================================
# ORB — Opening Range Breakout (15-min)
# ============================================================

def scan_orb_intraday(df: pd.DataFrame, symbol: str,
                      has_intraday: bool = False,
                      breeze=None) -> Optional[dict]:
    """
    Opening Range Breakout — 15-minute opening range, breakout after 9:30 AM IST.

    Logic:
      1. Compute 9:15-9:30 AM high/low (the 15-min opening range)
      2. After 9:30 AM, look for a 5-min candle closing ABOVE OR high (BUY) or
         BELOW OR low (SHORT) on volume > 2x 5-min average
      3. Signal valid only between 9:30 AM and 10:30 AM IST
      4. Stop = opposite end of opening range
      5. Target = 1.5x risk (T1), 2.5x risk (T2), 4x risk (T3)

    Requires Breeze API for 5-min intraday data.
    Returns None gracefully if breeze unavailable or no signal.
    """
    if not has_intraday or breeze is None:
        return None

    try:
        intraday = breeze.fetch_intraday(symbol, interval="5minute", days_back=1)
        if intraday is None or intraday.empty:
            return None
    except Exception:
        return None

    today = _today_bars(intraday)
    if today.empty:
        return None

    # ── Opening Range: 9:15-9:30 AM IST ──
    or_window = _filter_by_session(today, dtime(9, 15), dtime(9, 30))
    if or_window.empty or len(or_window) < 2:
        return None
    or_high = float(or_window["high"].max())
    or_low = float(or_window["low"].min())
    or_volume = float(or_window["volume"].sum())

    # ── Check current time window (9:30 AM - 10:30 AM) ──
    breakout_window = _filter_by_session(today, dtime(9, 30), dtime(10, 30))
    if breakout_window.empty:
        return None

    # ── Latest bar must be the breakout bar ──
    latest = breakout_window.iloc[-1]
    cmp = float(latest["close"])
    vol = float(latest["volume"])

    # Average volume of prior 5-min bars (excluding the opening range)
    prior_bars = breakout_window.iloc[:-1] if len(breakout_window) > 1 else or_window
    avg_vol_5min = float(prior_bars["volume"].mean()) if not prior_bars.empty else or_volume / 3
    vol_ratio = vol / (avg_vol_5min + 1)

    if vol_ratio < 2.0:
        return None  # Volume confirmation required

    # ── Direction ──
    if cmp > or_high:
        signal = "BUY"
        entry = cmp
        sl = round(or_low, 2)
        risk = entry - sl
        if risk <= 0 or risk / entry > 0.04:  # Cap stop at 4%
            return None
        t1 = round(entry + 1.5 * risk, 2)
        t2 = round(entry + 2.5 * risk, 2)
        t3 = round(entry + 4.0 * risk, 2)
    elif cmp < or_low:
        signal = "SHORT"
        entry = cmp
        sl = round(or_high, 2)
        risk = sl - entry
        if risk <= 0 or risk / entry > 0.04:
            return None
        t1 = round(entry - 1.5 * risk, 2)
        t2 = round(entry - 2.5 * risk, 2)
        t3 = round(entry - 4.0 * risk, 2)
    else:
        return None

    # ── Confidence Build-up ──
    confidence = 55
    reasons = []
    reasons.append(f"Opening range: ₹{or_low:.2f} - ₹{or_high:.2f} ({(or_high-or_low)/or_low*100:.2f}% width)")
    reasons.append(f"Breakout on {vol_ratio:.1f}x volume vs 5-min average")
    confidence += 10 if vol_ratio >= 3 else 5

    # OR width quality (tight OR > wide OR)
    or_width_pct = (or_high - or_low) / or_low * 100
    if or_width_pct < 1.5:
        confidence += 8
        reasons.append(f"Tight opening range ({or_width_pct:.2f}%) — high-conviction setup")
    elif or_width_pct < 3.0:
        confidence += 4

    # Time of breakout (earlier = better)
    breakout_time = breakout_window.index[-1]
    if hasattr(breakout_time, "time"):
        bt = breakout_time.time()
        if bt < dtime(9, 45):
            confidence += 7
            reasons.append("Early breakout (before 9:45 AM) — strongest statistical edge")
        elif bt < dtime(10, 0):
            confidence += 4

    # Compute current intraday RSI for context
    closes = today["close"].tail(14)
    if len(closes) >= 14:
        diff = closes.diff()
        gain = diff.where(diff > 0, 0).rolling(14).mean()
        loss = -diff.where(diff < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    else:
        rsi = 50.0

    confidence = min(confidence, 88)
    rr = round((t2 - entry) / risk if signal == "BUY" else (entry - t2) / risk, 1)

    return _build_intraday_result(
        symbol=symbol, strategy="ORB", signal=signal,
        cmp=cmp, entry=entry, sl=sl, t1=t1, t2=t2, t3=t3,
        rr=rr, confidence=confidence, reasons=reasons,
        volume_ratio=round(vol_ratio, 1), rsi=round(rsi, 1),
        hold_type="Intraday (2-6h)",
        timestamp=str(today.index[-1]),
    )


# ============================================================
# VWAP RECLAIM — Reclaim of intraday VWAP from below
# ============================================================

def scan_vwap_reclaim_intraday(df: pd.DataFrame, symbol: str,
                                has_intraday: bool = False,
                                breeze=None) -> Optional[dict]:
    """
    VWAP Reclaim — institutional support level reclaimed after intraday dip.

    Logic:
      1. Stock opened above VWAP, then dipped below VWAP intraday
      2. Latest 5-min candle reclaims VWAP (close > VWAP)
      3. Reclaim volume > 1.5x 5-min average
      4. VWAP slope flat or rising
      5. Time window: 10:15 AM - 12:30 PM IST
      6. Stop = day's intraday low
      7. Target = next intraday resistance / 1.5R, 2.5R
    """
    if not has_intraday or breeze is None:
        return None

    try:
        intraday = breeze.fetch_intraday(symbol, interval="5minute", days_back=1)
        if intraday is None or intraday.empty:
            return None
    except Exception:
        return None

    today = _today_bars(intraday)
    if today.empty or len(today) < 12:
        return None

    # Check time window
    valid_window = _filter_by_session(today, dtime(10, 15), dtime(12, 30))
    if valid_window.empty:
        return None

    # Compute VWAP for the full day so far
    today_with_vwap = today.copy()
    today_with_vwap["vwap"] = _compute_vwap(today_with_vwap)

    latest = today_with_vwap.iloc[-1]
    prev = today_with_vwap.iloc[-2]
    cmp = float(latest["close"])
    vwap_now = float(latest["vwap"])

    # ── Reclaim condition ──
    # Previous bar below VWAP, current bar above VWAP
    if float(prev["close"]) >= float(prev["vwap"]):
        return None  # Was already above — not a reclaim
    if cmp <= vwap_now:
        return None  # Hasn't reclaimed

    # ── Open of day must have been above VWAP (otherwise just a bounce-from-low) ──
    open_bar = today_with_vwap.iloc[0]
    if float(open_bar["close"]) < float(open_bar["vwap"]):
        # Started weak — not the classic reclaim setup
        # (Still valid in some cases, but reduces confidence)
        opened_above = False
    else:
        opened_above = True

    # ── Volume confirmation ──
    vol_5min_avg = float(today_with_vwap["volume"].iloc[-13:-1].mean())
    vol_ratio = float(latest["volume"]) / (vol_5min_avg + 1)
    if vol_ratio < 1.5:
        return None

    # ── VWAP slope flat or rising ──
    vwap_5_bars_ago = float(today_with_vwap["vwap"].iloc[-5]) if len(today_with_vwap) >= 5 else vwap_now
    vwap_slope_pct = (vwap_now - vwap_5_bars_ago) / vwap_5_bars_ago * 100
    if vwap_slope_pct < -0.1:  # Declining VWAP — bearish bias
        return None

    # ── Entry / Stop / Target ──
    entry = cmp
    sl = round(float(today_with_vwap["low"].min()), 2)  # Day's low as stop
    risk = entry - sl
    if risk <= 0 or risk / entry > 0.025:  # Cap at 2.5%
        return None

    t1 = round(entry + 1.5 * risk, 2)
    t2 = round(entry + 2.5 * risk, 2)
    t3 = round(entry + 4.0 * risk, 2)
    rr = round((t2 - entry) / risk, 1)

    # ── Confidence ──
    confidence = 55
    reasons = []
    reasons.append(f"VWAP reclaimed at ₹{vwap_now:.2f} (CMP ₹{cmp:.2f})")
    reasons.append(f"Reclaim volume {vol_ratio:.1f}x avg — institutional confirmation")
    confidence += 8 if vol_ratio >= 2.5 else 4

    if opened_above:
        reasons.append("Opened above VWAP, dipped, now reclaimed — classic pattern")
        confidence += 8
    else:
        reasons.append("Opened below VWAP — weaker setup, but reclaim is valid")

    if vwap_slope_pct > 0.1:
        reasons.append(f"VWAP rising ({vwap_slope_pct:+.2f}%) — institutional bid intact")
        confidence += 5

    # Day's low must be reasonably close
    distance_to_low = (cmp - sl) / cmp * 100
    if distance_to_low < 1.5:
        reasons.append(f"Tight stop {distance_to_low:.2f}% from low — excellent R:R")
        confidence += 6

    confidence = min(confidence, 85)

    # Intraday RSI
    closes = today_with_vwap["close"].tail(14)
    diff = closes.diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = -diff.where(diff < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1]) if len(closes) >= 14 else 50.0

    return _build_intraday_result(
        symbol=symbol, strategy="VWAP_Reclaim", signal="BUY",
        cmp=cmp, entry=entry, sl=sl, t1=t1, t2=t2, t3=t3,
        rr=rr, confidence=confidence, reasons=reasons,
        volume_ratio=round(vol_ratio, 1), rsi=round(rsi, 1),
        hold_type="Intraday (2-4h)",
        timestamp=str(today.index[-1]),
    )


# ============================================================
# LUNCH LOW REVERSAL — afternoon buy at lunch-hour low
# ============================================================

def scan_lunch_low_intraday(df: pd.DataFrame, symbol: str,
                             has_intraday: bool = False,
                             breeze=None) -> Optional[dict]:
    """
    Lunch Low Reversal — fade the lunch-hour weakness.

    Logic:
      1. Time window: 12:30 PM - 1:30 PM IST only
      2. Price within 0.5% of intraday low
      3. Intraday RSI(9) on 5-min bars < 35 (oversold)
      4. Volume DECLINING in last 3 candles (exhaustion)
      5. Daily close > SMA 50 (daily uptrend intact)
      6. Stop = intraday low - 0.3 * intraday ATR
      7. Target = VWAP / day's open / +1.5R, +2.5R
    """
    if not has_intraday or breeze is None:
        return None

    try:
        intraday = breeze.fetch_intraday(symbol, interval="5minute", days_back=1)
        if intraday is None or intraday.empty:
            return None
    except Exception:
        return None

    today = _today_bars(intraday)
    if today.empty or len(today) < 24:  # Need at least 2 hours of bars
        return None

    # Time gate
    valid_window = _filter_by_session(today, dtime(12, 30), dtime(13, 30))
    if valid_window.empty:
        return None

    # Daily trend filter (use df = daily data passed in)
    if df is not None and not df.empty and len(df) >= 50:
        if "sma_50" in df.columns:
            daily_close = float(df["close"].iloc[-1])
            daily_sma50 = float(df["sma_50"].iloc[-1])
            if daily_close < daily_sma50:
                return None  # Daily downtrend — don't fade
        elif "Close" in df.columns:
            daily_close = float(df["Close"].iloc[-1])
            daily_sma50 = float(df["Close"].iloc[-50:].mean())
            if daily_close < daily_sma50:
                return None

    latest = valid_window.iloc[-1]
    cmp = float(latest["close"])
    intraday_low = float(today["low"].min())

    # Price near intraday low (within 0.5%)
    if (cmp - intraday_low) / intraday_low > 0.005:
        return None

    # Intraday RSI(9) on 5-min bars
    closes = today["close"].tail(20)
    if len(closes) < 10:
        return None
    diff = closes.diff()
    gain = diff.where(diff > 0, 0).rolling(9).mean()
    loss = -diff.where(diff < 0, 0).rolling(9).mean()
    rs = gain / (loss + 1e-10)
    intraday_rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    if intraday_rsi >= 35:
        return None  # Not oversold enough

    # Volume declining (last 3 bars)
    last3_vol = today["volume"].tail(3).values
    if len(last3_vol) < 3:
        return None
    if not (last3_vol[0] >= last3_vol[1] >= last3_vol[2]):
        return None  # Volume not declining

    # ── Entry / Stop / Target ──
    entry = cmp
    intraday_high = float(today["high"].max())
    intraday_atr = float((today["high"] - today["low"]).tail(10).mean())
    sl = round(intraday_low - 0.3 * intraday_atr, 2)
    risk = entry - sl
    if risk <= 0 or risk / entry > 0.02:  # Cap at 2%
        return None

    # Targets: VWAP and day's open are natural intraday resistance
    today_with_vwap = today.copy()
    today_with_vwap["vwap"] = _compute_vwap(today_with_vwap)
    vwap_now = float(today_with_vwap["vwap"].iloc[-1])
    day_open = float(today["open"].iloc[0])

    natural_targets = sorted([vwap_now, day_open, intraday_high * 0.99])
    natural_targets = [t for t in natural_targets if t > entry * 1.01]

    if len(natural_targets) >= 1:
        t1 = round(max(natural_targets[0], entry + 1.5 * risk), 2)
    else:
        t1 = round(entry + 1.5 * risk, 2)
    t2 = round(entry + 2.5 * risk, 2)
    t3 = round(entry + 4.0 * risk, 2)
    rr = round((t2 - entry) / risk, 1)

    # Avg volume of prior bars (not the last 3)
    avg_vol = float(today["volume"].iloc[:-3].mean()) if len(today) > 3 else 1.0
    vol_ratio = float(last3_vol[-1]) / (avg_vol + 1)

    confidence = 50
    reasons = []
    reasons.append(f"Price at intraday low ₹{intraday_low:.2f} during lunch hour")
    reasons.append(f"Intraday RSI(9) = {intraday_rsi:.1f} — oversold")
    confidence += 8 if intraday_rsi < 25 else 5

    reasons.append(f"Volume declining ({last3_vol[0]:.0f}→{last3_vol[1]:.0f}→{last3_vol[2]:.0f}) — selling exhausted")
    confidence += 6

    reasons.append(f"Daily uptrend intact (daily close > SMA50)")
    confidence += 5

    if cmp < vwap_now and t1 >= vwap_now:
        reasons.append(f"VWAP at ₹{vwap_now:.2f} — natural reversion target")
        confidence += 5

    confidence = min(confidence, 80)

    return _build_intraday_result(
        symbol=symbol, strategy="Lunch_Low", signal="BUY",
        cmp=cmp, entry=entry, sl=sl, t1=t1, t2=t2, t3=t3,
        rr=rr, confidence=confidence, reasons=reasons,
        volume_ratio=round(vol_ratio, 1), rsi=round(intraday_rsi, 1),
        hold_type="Intraday (2-3h)",
        timestamp=str(today.index[-1]),
    )


# ============================================================
# WIRING — How to call from run_scanner()
# ============================================================
#
# In scanners.py::run_scanner(), the INTRADAY_SCANNERS dict needs to be
# augmented so the runner can pass `breeze` into the call:
#
# def run_scanner(scanner_name, data_dict, nifty_df=None, regime=None,
#                 has_intraday=False, sector_rankings=None, min_rs=0,
#                 compute_sqi_flag=True, breeze=None):
#     ...
#     for symbol, df in data_dict.items():
#         ...
#         if scanner_name in INTRADAY_SCANNERS:
#             result = scanner_func(enriched, symbol, has_intraday=has_intraday, breeze=breeze)
#         else:
#             result = scanner_func(enriched, symbol)
#
# And callers should pass: run_scanner(..., breeze=st.session_state.breeze_engine)
# ============================================================
