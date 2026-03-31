"""
Option Chain Analysis Module v2 — NSE Scanner Pro
===================================================
FIX: Breeze API does not return implied volatility.
When IV is unavailable (all zeros), the 3 IV-dependent factors are excluded
and weights are redistributed to the 4 working factors.

Without IV (typical):           With IV (if available):
  OI Change: 25% → 38%          OI Change: 25%
  UOA:       20% → 31%          UOA:       20%
  PCR:       15% → 19%          PCR:       15%
  Max Pain:  10% → 12%          Max Pain:  10%
  IVP:        0%  (skipped)     IVP:       15%
  Skew:       0%  (skipped)     Skew:       7.5%
  IVS:        0%  (skipped)     IVS:        7.5%
  Total: 100%                   Total: 100%

The composite score label now shows "(4-factor)" vs "(7-factor)" so the user
always knows exactly what data was used.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# WEIGHTS — two sets
# ============================================================

WEIGHTS_7F = {   # Full 7-factor (when IV is available)
    "oi_change": 0.25, "uoa": 0.20, "ivp": 0.15, "pcr": 0.15,
    "max_pain":  0.10, "skew": 0.075, "ivs": 0.075,
}
WEIGHTS_4F = {   # 4-factor (when Breeze doesn't return IV)
    "oi_change": 0.38, "uoa": 0.31, "pcr": 0.19,
    "max_pain":  0.12, "ivp": 0.0, "skew": 0.0, "ivs": 0.0,
}

NSE_PCR = {
    "extreme_bullish": 1.5, "bullish": 1.2,
    "neutral_high": 1.0,    "neutral_low": 0.8,
    "bearish": 0.7,         "extreme_bearish": 0.5,
}

SIGNAL_THRESHOLDS = {"strong_buy": 75, "buy": 60, "neutral_low": 45, "sell": 30}


@dataclass
class OptionChainSignal:
    symbol: str
    signal: str
    composite_score: float
    confidence: float
    factors_used: int           # 4 or 7
    iv_available: bool          # honest flag

    oi_change_score: float
    uoa_score: float
    ivp_score: float
    pcr_score: float
    max_pain_score: float
    skew_score: float
    ivs_score: float

    spot: float
    pcr: float
    max_pain: float
    call_wall: float
    put_wall: float
    iv_spread: float
    total_call_oi: float
    total_put_oi: float
    expiry_date: str
    dte: int

    pcr_interpretation: str
    oi_pattern: str
    signal_icon: str
    component_breakdown: str
    reasons: List[str]
    warnings: List[str]
    setup_type: str
    is_fno_banned: bool = False
    liquidity_ok: bool = True

    # Score alias used by app.py
    score: float = 0.0

    def __post_init__(self):
        self.score = self.composite_score


# ============================================================
# IV DETECTION
# ============================================================

def _has_iv(calls: pd.DataFrame, puts: pd.DataFrame) -> bool:
    """Returns True only if real IV data is present (not all zeros)."""
    for df in [calls, puts]:
        if df.empty or "iv" not in df.columns:
            continue
        iv_vals = pd.to_numeric(df["iv"], errors="coerce").dropna()
        if (iv_vals > 0).any():
            return True
    return False


# ============================================================
# MAIN SCORING
# ============================================================

def compute_option_chain_signal(oc_data: Dict) -> Optional[OptionChainSignal]:
    if not oc_data:
        return None

    calls          = oc_data.get("calls", pd.DataFrame())
    puts           = oc_data.get("puts",  pd.DataFrame())
    spot           = float(oc_data.get("spot", 0))
    pcr            = float(oc_data.get("pcr", 1.0))
    max_pain       = float(oc_data.get("max_pain", 0))
    call_wall      = float(oc_data.get("call_wall", 0))
    put_wall       = float(oc_data.get("put_wall", 0))
    iv_spread      = float(oc_data.get("iv_spread", 0))
    total_call_oi  = float(oc_data.get("total_call_oi", 0))
    total_put_oi   = float(oc_data.get("total_put_oi", 0))
    expiry_date    = oc_data.get("expiry_date", "")
    dte            = int(oc_data.get("dte", 30))
    symbol         = oc_data.get("symbol", "")

    reasons  = []
    warnings = []

    # ── Liquidity check ──
    liquidity_ok = (total_call_oi >= 5000 or total_put_oi >= 5000)
    if not liquidity_ok:
        warnings.append(f"Low OI ({total_call_oi:.0f} calls / {total_put_oi:.0f} puts) — unreliable signals")

    # ── Detect IV availability ──
    iv_avail = _has_iv(calls, puts)
    weights  = WEIGHTS_7F if iv_avail else WEIGHTS_4F
    factors  = 7 if iv_avail else 4

    if not iv_avail:
        warnings.append(
            "Breeze API does not provide IV data — scoring on 4 factors (OI, UOA, PCR, Max Pain). "
            "IV-based signals (IVP, Skew, IV Spread) are excluded."
        )

    # ── Component scores ──
    oi_score,   oi_pattern        = _score_oi_change(calls, puts, spot)
    uoa_score,  uoa_desc          = _score_unusual_activity(calls, puts)
    ivp_score,  ivp_desc          = _score_iv_percentile(calls, puts) if iv_avail else (50.0, "IV unavailable")
    pcr_score,  pcr_interpretation= _score_pcr_contrarian(pcr)
    mp_score,   mp_desc           = _score_max_pain(spot, max_pain, dte)
    skew_score, skew_desc         = _score_iv_skew(calls, puts, spot) if iv_avail else (50.0, "IV unavailable")
    ivs_score,  ivs_desc          = _score_iv_spread(iv_spread) if iv_avail else (50.0, "IV unavailable")

    reasons.append(f"OI: {oi_pattern}")
    if uoa_desc: reasons.append(uoa_desc)
    reasons.append(f"PCR {pcr:.2f}: {pcr_interpretation}")
    if mp_desc:  reasons.append(mp_desc)
    if iv_avail:
        reasons.append(ivp_desc)
        reasons.append(skew_desc)
        reasons.append(ivs_desc)
    if call_wall > 0: reasons.append(f"Call wall (resistance): ₹{call_wall:,.0f}")
    if put_wall  > 0: reasons.append(f"Put wall (support): ₹{put_wall:,.0f}")

    # ── Composite (expiry-adjusted weights) ──
    w = _expiry_adjusted_weights(weights, dte)

    composite = (
        w["oi_change"] * oi_score  +
        w["uoa"]       * uoa_score +
        w["ivp"]       * ivp_score +
        w["pcr"]       * pcr_score +
        w["max_pain"]  * mp_score  +
        w["skew"]      * skew_score+
        w["ivs"]       * ivs_score
    )
    composite = round(min(max(composite, 0), 100), 1)

    # ── Confidence = 100 - stddev of ACTIVE components ──
    active_scores = [oi_score, uoa_score, pcr_score, mp_score]
    if iv_avail:
        active_scores += [ivp_score, skew_score, ivs_score]
    confidence = round(max(0, 100 - float(np.std(active_scores))), 1)

    # ── Signal ──
    if   composite >= SIGNAL_THRESHOLDS["strong_buy"]:  signal, icon = "STRONG BUY",  "🟢🟢"
    elif composite >= SIGNAL_THRESHOLDS["buy"]:          signal, icon = "BUY",         "🟢"
    elif composite >= SIGNAL_THRESHOLDS["neutral_low"]:  signal, icon = "NEUTRAL",     "⚪"
    elif composite >= SIGNAL_THRESHOLDS["sell"]:         signal, icon = "SELL",        "🔴"
    else:                                                 signal, icon = "STRONG SELL", "🔴🔴"

    setup_type = "UNUSUAL_ACTIVITY" if uoa_score >= 80 else ("EXPIRY_PLAY" if dte <= 3 else "OI_COMPOSITE")
    factor_tag = f"{'7' if iv_avail else '4'}-factor"

    breakdown = (
        f"[{factor_tag}] OI:{oi_score:.0f} UOA:{uoa_score:.0f} "
        f"PCR:{pcr_score:.0f} MP:{mp_score:.0f}"
        + (f" IVP:{ivp_score:.0f} Skew:{skew_score:.0f} IVS:{ivs_score:.0f}" if iv_avail else "")
    )

    return OptionChainSignal(
        symbol=symbol, signal=signal, composite_score=composite, confidence=confidence,
        factors_used=factors, iv_available=iv_avail,
        oi_change_score=oi_score, uoa_score=uoa_score, ivp_score=ivp_score,
        pcr_score=pcr_score, max_pain_score=mp_score, skew_score=skew_score, ivs_score=ivs_score,
        spot=spot, pcr=pcr, max_pain=max_pain, call_wall=call_wall, put_wall=put_wall,
        iv_spread=iv_spread, total_call_oi=total_call_oi, total_put_oi=total_put_oi,
        expiry_date=expiry_date, dte=dte,
        pcr_interpretation=pcr_interpretation, oi_pattern=oi_pattern,
        signal_icon=icon, component_breakdown=breakdown,
        reasons=reasons, warnings=warnings,
        setup_type=setup_type, liquidity_ok=liquidity_ok,
        score=composite,
    )


# ============================================================
# COMPONENT SCORERS
# ============================================================

def _score_oi_change(calls: pd.DataFrame, puts: pd.DataFrame, spot: float):
    if calls.empty or puts.empty or spot <= 0:
        return 50.0, "Insufficient OI data"
    try:
        atm_c = calls[(calls["strike"] >= spot*0.90) & (calls["strike"] <= spot*1.10)]
        atm_p = puts[ (puts["strike"]  >= spot*0.90) & (puts["strike"]  <= spot*1.10)]
        nc = float(atm_c["oi_change"].sum()) if not atm_c.empty else 0
        np_ = float(atm_p["oi_change"].sum()) if not atm_p.empty else 0

        if nc > 0 and np_ < 0:  return min(92, 90 + min(abs(nc)+abs(np_),1e6)/1e6*5), "Long Buildup 📈"
        if nc < 0 and np_ > 0:  return max(8,  10 - min(abs(nc)+abs(np_),1e6)/1e6*5), "Short Buildup 📉"
        if nc > 0 and np_ > 0:  return 50.0, "Dual OI Increase (Hedging)"
        if nc < 0 and np_ < 0:  return 45.0, "OI Unwinding"
        if nc > 0:               return 65.0, "Short Covering"
        if np_ > 0:              return 35.0, "Long Unwinding"
        return 50.0, "Neutral OI"
    except Exception: return 50.0, "OI processing error"


def _score_unusual_activity(calls: pd.DataFrame, puts: pd.DataFrame):
    if calls.empty and puts.empty: return 50.0, ""
    try:
        threshold   = 5
        c_avg       = calls["volume"].mean() if not calls.empty else 1
        p_avg       = puts["volume"].mean()  if not puts.empty  else 1
        unc = int((calls["volume"] > c_avg * threshold).sum()) if not calls.empty else 0
        unp = int((puts["volume"]  > p_avg * threshold).sum()) if not puts.empty  else 0
        if unc > unp > 0: return min(95, 80 + unc*5), f"🔥 Unusual calls ({unc} strikes > {threshold}x)"
        if unp > unc > 0: return max(5,  20 - unp*5), f"⚠️ Unusual puts  ({unp} strikes > {threshold}x)"
        cv = calls["volume"].sum() if not calls.empty else 0
        pv = puts["volume"].sum()  if not puts.empty  else 0
        ratio = cv / (pv + 1)
        if ratio > 2: return 72.0, f"Call vol {ratio:.1f}x put vol"
        if ratio < 0.5: return 30.0, f"Put vol {1/ratio:.1f}x call vol"
        return 50.0, "Normal activity"
    except Exception: return 50.0, ""


def _score_iv_percentile(calls: pd.DataFrame, puts: pd.DataFrame):
    try:
        ivs = []
        for df in [calls, puts]:
            if not df.empty and "iv" in df.columns:
                ivs.extend(pd.to_numeric(df["iv"], errors="coerce").dropna().tolist())
        if not ivs: return 55.0, "IV data empty"
        avg = np.mean(ivs)
        if avg > 60:  return 80.0, f"IV elevated ({avg:.0f}%) — contrarian bullish"
        if avg > 40:  return 65.0, f"IV above normal ({avg:.0f}%)"
        if avg > 20:  return 55.0, f"IV normal ({avg:.0f}%)"
        if avg > 10:  return 40.0, f"IV suppressed ({avg:.0f}%) — complacency"
        return 35.0, f"IV very low ({avg:.0f}%)"
    except Exception: return 55.0, "IV calculation error"


def _score_pcr_contrarian(pcr: float):
    cfg = NSE_PCR
    if pcr >= cfg["extreme_bullish"]: return 90.0, f"Extreme fear PCR {pcr:.2f} → contrarian BUY"
    if pcr >= cfg["bullish"]:         return 78.0, f"High fear PCR {pcr:.2f} → bullish"
    if pcr >= cfg["neutral_high"]:    return 65.0, f"Slightly bearish PCR {pcr:.2f}"
    if pcr >= cfg["neutral_low"]:     return 50.0, f"Neutral PCR {pcr:.2f}"
    if pcr >= cfg["bearish"]:         return 40.0, f"Complacency PCR {pcr:.2f}"
    if pcr >= cfg["extreme_bearish"]: return 25.0, f"High complacency PCR {pcr:.2f}"
    return 15.0, f"Extreme complacency PCR {pcr:.2f} → contrarian SELL"


def _score_max_pain(spot: float, max_pain: float, dte: int):
    if max_pain <= 0 or spot <= 0: return 50.0, ""
    dist = (spot - max_pain) / max_pain * 100
    if dist < -5:   score, desc = 75.0, f"Spot {dist:.1f}% below Max Pain ₹{max_pain:,.0f} → drift up"
    elif dist < -2: score, desc = 65.0, f"Spot {dist:.1f}% below Max Pain ₹{max_pain:,.0f}"
    elif dist < 2:  score, desc = 55.0, f"Spot near Max Pain ₹{max_pain:,.0f} ({dist:+.1f}%)"
    elif dist < 5:  score, desc = 40.0, f"Spot {dist:.1f}% above Max Pain ₹{max_pain:,.0f}"
    else:           score, desc = 30.0, f"Spot {dist:.1f}% above Max Pain ₹{max_pain:,.0f} → drift down"
    if dte > 10:
        score = 50 + (score - 50) * 0.5
    return round(score, 1), desc


def _score_iv_skew(calls: pd.DataFrame, puts: pd.DataFrame, spot: float):
    try:
        otm_puts  = puts[ (puts["strike"]  < spot*0.97) & (puts["strike"]  > spot*0.85)] if not puts.empty  else pd.DataFrame()
        otm_calls = calls[(calls["strike"] > spot*1.03) & (calls["strike"] < spot*1.15)] if not calls.empty else pd.DataFrame()
        pp = float(otm_puts["iv"].mean())  if not otm_puts.empty  and "iv" in otm_puts.columns  else 0
        cp = float(otm_calls["iv"].mean()) if not otm_calls.empty and "iv" in otm_calls.columns else 0
        if pp <= 0 and cp <= 0: return 50.0, "OTM IV unavailable"
        skew = pp - cp
        if skew > 20: return 72.0, f"Steep put skew ({skew:.0f}) — panic buying"
        if skew > 10: return 63.0, f"Moderate put skew ({skew:.0f})"
        if skew > 0:  return 55.0, f"Normal put skew ({skew:.0f})"
        if skew > -10:return 45.0, f"Call skew developing ({skew:.0f})"
        return 35.0, f"Call skew ({skew:.0f}) — FOMO"
    except Exception: return 50.0, "IV skew error"


def _score_iv_spread(iv_spread: float):
    if iv_spread > 5:   return 85.0, f"IV Spread +{iv_spread:.1f}% — strong informed buying"
    if iv_spread > 2:   return 78.0, f"IV Spread +{iv_spread:.1f}% — calls premium (Cremers)"
    if iv_spread > 0.5: return 62.0, f"IV Spread +{iv_spread:.1f}% — slight call premium"
    if iv_spread > -0.5:return 52.0, f"IV Spread {iv_spread:.1f}% — parity"
    if iv_spread > -2:  return 40.0, f"IV Spread {iv_spread:.1f}% — slight put premium"
    if iv_spread > -5:  return 25.0, f"IV Spread {iv_spread:.1f}% — puts expensive (Cremers)"
    return 15.0, f"IV Spread {iv_spread:.1f}% — strong informed selling"


def _expiry_adjusted_weights(base: dict, dte: int) -> dict:
    w = dict(base)
    if dte <= 3:
        # Boost Max Pain to 20% near expiry (only matters in last 3 days)
        extra = 0.10
        ratio = (1 - 0.20) / max(1 - base.get("max_pain", 0.10), 0.01)
        w = {k: round(v * ratio, 4) for k, v in w.items()}
        w["max_pain"] = 0.20
    # Normalise to sum = 1.0
    total = sum(w.values())
    return {k: round(v / total, 4) for k, v in w.items()}


# ============================================================
# COMPATIBILITY WRAPPERS (app.py interface)
# ============================================================

def analyze_option_chain(symbol: str, oc_data: Dict,
                          price_df=None) -> Optional[OptionChainSignal]:
    oc_data["symbol"] = symbol
    return compute_option_chain_signal(oc_data)


OptionChainResult = OptionChainSignal


def format_oc_signal(result: OptionChainSignal) -> str:
    if result is None: return "No option chain data."
    factor_tag = f"{'7' if result.iv_available else '4'}-factor"
    return (
        f"{result.signal_icon} OC Signal ({factor_tag}) — {result.symbol}\n"
        f"Score: {result.composite_score:.0f}/100 ({result.signal}) | "
        f"Confidence: {result.confidence:.0f}%\n"
        f"OI: {result.oi_pattern} | PCR: {result.pcr:.2f}\n"
        f"Max Pain: ₹{result.max_pain:,.0f} | Spot: ₹{result.spot:,.0f} | DTE: {result.dte}"
        + ("\nNote: IV data unavailable — using 4-factor model" if not result.iv_available else "")
    )
