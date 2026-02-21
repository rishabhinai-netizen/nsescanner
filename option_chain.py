"""
Option Chain Analysis Module — NSE Scanner Pro v15
====================================================
7-factor composite scoring system backed by academic research.

Research foundation:
- Pan & Poteshman (2006): Options contain 40bps/day directional info
- Cremers & Weinbaum (2010): IV spread predicts 50bps/week — most robust signal
- Bondarenko & Muravyev (2022): PCR predictability died post-2009 for individual stocks
  → PCR now used as CONTRARIAN only, not directional
- Gamma positioning (GEX): predicts volatility regime, not direction
- OI Change quadrant matrix: real-time positioning intelligence

Composite Score:
  OI Change Pattern     25% — 4-quadrant Price+OI matrix (core directional)
  Unusual Options Act.  20% — volume spike > 5x avg (smart money detection)
  IV Percentile         15% — contrarian at extremes
  PCR Contrarian        15% — NSE-specific thresholds (not US CBOE thresholds)
  Max Pain Distance     10% — expiry proximity weighted (→ 20% if DTE ≤ 3)
  IV Skew               7.5% — put vs call skew (panic/FOMO detector)
  IV Spread             7.5% — call IV - put IV at ATM (Cremers signal)

Signal:
  > 75 = STRONG BUY | 60-75 = BUY | 45-60 = NEUTRAL
  30-45 = SELL | < 30 = STRONG SELL

Confidence = 100 - stddev(component scores) → high when all signals agree.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION — NSE-specific thresholds
# ============================================================================

OC_CONFIG = {
    # Signal weights (sum = 1.0 for non-expiry)
    "w_oi_change": 0.25,
    "w_uoa": 0.20,
    "w_ivp": 0.15,
    "w_pcr": 0.15,
    "w_max_pain": 0.10,
    "w_skew": 0.075,
    "w_ivs": 0.075,

    # NSE/Nifty PCR thresholds (not US CBOE thresholds)
    "pcr_extreme_bullish": 1.5,   # > 1.5 = extreme fear → contrarian BUY
    "pcr_bullish": 1.2,
    "pcr_neutral_high": 1.0,
    "pcr_neutral_low": 0.8,
    "pcr_bearish": 0.7,
    "pcr_extreme_bearish": 0.5,   # < 0.5 = extreme complacency → contrarian SELL

    # Decision thresholds
    "strong_buy": 75,
    "buy": 60,
    "neutral_low": 45,
    "sell": 30,

    # Expiry proximity modifier
    "dte_near": 3,    # Max Pain weight → 20% when DTE ≤ 3
    "dte_far": 30,    # Max Pain weight → 5% when DTE > 30

    # Liquidity filter (minimum for reliable signals)
    "min_oi_contracts": 5000,
    "min_daily_volume": 1000,

    # Unusual options activity threshold
    "uoa_volume_multiplier": 5,   # Volume > 5x avg = institutional activity
    "uoa_sweep_threshold": 5,     # Large order detection threshold (lots)

    # MWPL thresholds (F&O ban)
    "mwpl_ban": 0.95,
    "mwpl_caution": 0.80,
}


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class OptionChainSignal:
    """Complete option chain signal with all components."""
    symbol: str
    signal: str                  # STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
    composite_score: float       # 0-100
    confidence: float            # 0-100 (100 - stddev of components)

    # Component scores (0-100 each)
    oi_change_score: float
    uoa_score: float
    ivp_score: float
    pcr_score: float
    max_pain_score: float
    skew_score: float
    ivs_score: float

    # Market data
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

    # Human-readable
    pcr_interpretation: str
    oi_pattern: str              # "Long Buildup" / "Short Buildup" / "Short Covering" / "Long Unwinding"
    signal_icon: str
    component_breakdown: str
    reasons: List[str]
    warnings: List[str]

    # Setup type
    setup_type: str              # "OI_COMPOSITE" / "UNUSUAL_ACTIVITY" / "EXPIRY_PLAY"

    # For further display
    is_fno_banned: bool = False
    liquidity_ok: bool = True


# ============================================================================
# MAIN SCORING ENGINE
# ============================================================================

def compute_option_chain_signal(oc_data: Dict) -> Optional[OptionChainSignal]:
    """
    Compute composite 7-factor option chain signal from raw Breeze data.

    Args:
        oc_data: Dict from BreezeEngine.fetch_option_chain()

    Returns:
        OptionChainSignal or None if insufficient data.
    """
    if not oc_data:
        return None

    calls: pd.DataFrame = oc_data.get("calls", pd.DataFrame())
    puts: pd.DataFrame = oc_data.get("puts", pd.DataFrame())
    spot = float(oc_data.get("spot", 0))
    pcr = float(oc_data.get("pcr", 1.0))
    max_pain = float(oc_data.get("max_pain", 0))
    call_wall = float(oc_data.get("call_wall", 0))
    put_wall = float(oc_data.get("put_wall", 0))
    iv_spread = float(oc_data.get("iv_spread", 0))
    total_call_oi = float(oc_data.get("total_call_oi", 0))
    total_put_oi = float(oc_data.get("total_put_oi", 0))
    expiry_date = oc_data.get("expiry_date", "")
    dte = int(oc_data.get("dte", 30))
    symbol = oc_data.get("symbol", "")

    reasons = []
    warnings = []

    # ── Liquidity Check ──
    liquidity_ok = (
        total_call_oi >= OC_CONFIG["min_oi_contracts"] or
        total_put_oi >= OC_CONFIG["min_oi_contracts"]
    )
    if not liquidity_ok:
        warnings.append(f"Low OI ({total_call_oi:.0f} calls, {total_put_oi:.0f} puts) — signals unreliable")

    # ── COMPONENT 1: OI Change Pattern (25%) ──
    # The 4-quadrant matrix: Price + OI direction
    oi_change_score, oi_pattern = _score_oi_change_pattern(calls, puts, spot)
    reasons.append(f"OI Pattern: {oi_pattern}")

    # ── COMPONENT 2: Unusual Options Activity (20%) ──
    uoa_score, uoa_desc = _score_unusual_activity(calls, puts)
    if uoa_desc:
        reasons.append(uoa_desc)

    # ── COMPONENT 3: IV Percentile (15%) ──
    ivp_score, ivp_desc = _score_iv_percentile(calls, puts)
    reasons.append(ivp_desc)

    # ── COMPONENT 4: PCR Contrarian (15%) ──
    # NSE-specific thresholds — PCR is CONTRARIAN not directional for individual stocks
    pcr_score, pcr_interpretation = _score_pcr_contrarian(pcr)
    reasons.append(f"PCR {pcr:.2f}: {pcr_interpretation}")

    # ── COMPONENT 5: Max Pain Distance (10%, adjustable) ──
    # Higher weight near expiry (DTE ≤ 3 → weight becomes 20%)
    max_pain_score, max_pain_desc = _score_max_pain(spot, max_pain, dte)
    if max_pain_desc:
        reasons.append(max_pain_desc)

    # ── COMPONENT 6: IV Skew (7.5%) ──
    skew_score, skew_desc = _score_iv_skew(calls, puts, spot)
    reasons.append(skew_desc)

    # ── COMPONENT 7: IV Spread / Parity (7.5%) ──
    # Cremers & Weinbaum: ATM call IV - put IV, 50bps/week predictability
    ivs_score, ivs_desc = _score_iv_spread(iv_spread)
    reasons.append(ivs_desc)

    # ── Apply Expiry Proximity Modifier ──
    weights = _get_weights_with_expiry_mod(dte)

    # ── Composite Score ──
    components = [oi_change_score, uoa_score, ivp_score, pcr_score,
                  max_pain_score, skew_score, ivs_score]
    w = [weights["w_oi_change"], weights["w_uoa"], weights["w_ivp"],
         weights["w_pcr"], weights["w_max_pain"], weights["w_skew"], weights["w_ivs"]]

    composite = sum(c * wt for c, wt in zip(components, w))
    composite = round(min(max(composite, 0), 100), 1)

    # ── Confidence = 100 - stddev of components ──
    # Low stddev = all signals agree = high confidence
    stddev = float(np.std(components))
    confidence = round(max(0, 100 - stddev), 1)

    # ── Signal Classification ──
    if composite >= OC_CONFIG["strong_buy"]:
        signal = "STRONG_BUY"
        signal_icon = "🟢🟢"
    elif composite >= OC_CONFIG["buy"]:
        signal = "BUY"
        signal_icon = "🟢"
    elif composite >= OC_CONFIG["neutral_low"]:
        signal = "NEUTRAL"
        signal_icon = "⚪"
    elif composite >= OC_CONFIG["sell"]:
        signal = "SELL"
        signal_icon = "🔴"
    else:
        signal = "STRONG_SELL"
        signal_icon = "🔴🔴"

    # ── Setup Type ──
    if uoa_score >= 80:
        setup_type = "UNUSUAL_ACTIVITY"
    elif dte <= 3:
        setup_type = "EXPIRY_PLAY"
    else:
        setup_type = "OI_COMPOSITE"

    # ── Key Levels Description ──
    if call_wall > 0 and spot > 0:
        reasons.append(f"Call Wall (Resistance): ₹{call_wall:,.0f}")
    if put_wall > 0 and spot > 0:
        reasons.append(f"Put Wall (Support): ₹{put_wall:,.0f}")
    if max_pain > 0 and spot > 0:
        mp_dist = ((spot - max_pain) / max_pain * 100) if max_pain > 0 else 0
        reasons.append(f"Max Pain: ₹{max_pain:,.0f} (Spot {mp_dist:+.1f}% away)")

    component_breakdown = (
        f"OI:{oi_change_score:.0f} UOA:{uoa_score:.0f} IVP:{ivp_score:.0f} "
        f"PCR:{pcr_score:.0f} MP:{max_pain_score:.0f} Skew:{skew_score:.0f} IVS:{ivs_score:.0f}"
    )

    return OptionChainSignal(
        symbol=symbol,
        signal=signal,
        composite_score=composite,
        confidence=confidence,
        oi_change_score=oi_change_score,
        uoa_score=uoa_score,
        ivp_score=ivp_score,
        pcr_score=pcr_score,
        max_pain_score=max_pain_score,
        skew_score=skew_score,
        ivs_score=ivs_score,
        spot=spot,
        pcr=pcr,
        max_pain=max_pain,
        call_wall=call_wall,
        put_wall=put_wall,
        iv_spread=iv_spread,
        total_call_oi=total_call_oi,
        total_put_oi=total_put_oi,
        expiry_date=expiry_date,
        dte=dte,
        pcr_interpretation=pcr_interpretation,
        oi_pattern=oi_pattern,
        signal_icon=signal_icon,
        component_breakdown=component_breakdown,
        reasons=reasons,
        warnings=warnings,
        setup_type=setup_type,
        liquidity_ok=liquidity_ok,
    )


# ============================================================================
# COMPONENT SCORERS
# ============================================================================

def _score_oi_change_pattern(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> tuple:
    """
    Score based on the 4-quadrant OI-Price matrix:
    Price↑ + OI↑ = Long Buildup (bullish) → 90-95
    Price↓ + OI↑ = Short Buildup (bearish) → 10-15
    Price↑ + OI↓ = Short Covering (temporary bullish) → 70
    Price↓ + OI↓ = Long Unwinding (temporary bearish) → 30
    """
    if calls.empty or puts.empty:
        return 50.0, "Insufficient OI data"

    try:
        # Near-money OI changes (±10% of spot)
        atm_calls = calls[(calls["strike"] >= spot * 0.9) & (calls["strike"] <= spot * 1.1)]
        atm_puts = puts[(puts["strike"] >= spot * 0.9) & (puts["strike"] <= spot * 1.1)]

        call_oi_change = atm_calls["oi_change"].sum() if not atm_calls.empty else 0
        put_oi_change = atm_puts["oi_change"].sum() if not atm_puts.empty else 0

        # Net OI direction
        # Call OI up + Put OI down = bullish (long buildup + put unwinding)
        # Call OI down + Put OI up = bearish (short buildup + put covering)
        net_call = call_oi_change  # Positive = more calls being bought
        net_put = put_oi_change    # Positive = more puts being bought

        if net_call > 0 and net_put < 0:
            # Classic long buildup
            strength = min(abs(net_call) + abs(net_put), 1e6) / 1e6
            score = min(90 + strength * 5, 95)
            pattern = "Long Buildup 📈"
        elif net_call < 0 and net_put > 0:
            # Classic short buildup
            strength = min(abs(net_call) + abs(net_put), 1e6) / 1e6
            score = max(10 - strength * 5, 5)
            pattern = "Short Buildup 📉"
        elif net_call > 0 and net_put > 0:
            # Both OI increasing — two-sided positioning (expiry hedging)
            score = 50
            pattern = "Dual OI Increase (Hedging)"
        elif net_call < 0 and net_put < 0:
            # Both OI decreasing — unwinding
            score = 45
            pattern = "OI Unwinding (Liquidation)"
        elif net_call > 0:
            # Short covering on calls (bearish to neutral reversal)
            score = 65
            pattern = "Short Covering (Call OI Up)"
        elif net_put > 0:
            # Long unwinding on puts
            score = 35
            pattern = "Long Unwinding (Put OI Up)"
        else:
            score = 50
            pattern = "Neutral OI"

        return round(score, 1), pattern

    except Exception as e:
        logger.debug(f"OI change score failed: {e}")
        return 50.0, "OI data processing error"


def _score_unusual_activity(calls: pd.DataFrame, puts: pd.DataFrame) -> tuple:
    """
    Detect unusual options activity: volume > 5x average on specific contracts.
    High call UOA = smart money buying upside → bullish.
    High put UOA = protection/hedge buying → bearish.
    """
    if calls.empty and puts.empty:
        return 50.0, ""

    try:
        threshold = OC_CONFIG["uoa_volume_multiplier"]

        call_vol = calls["volume"].sum() if not calls.empty else 0
        put_vol = puts["volume"].sum() if not puts.empty else 0

        call_avg = calls["volume"].mean() if not calls.empty else 1
        put_avg = puts["volume"].mean() if not puts.empty else 1

        # Check for unusual spikes on individual strikes
        unusual_calls = (calls["volume"] > call_avg * threshold).sum() if not calls.empty else 0
        unusual_puts = (puts["volume"] > put_avg * threshold).sum() if not puts.empty else 0

        call_vol_ratio = call_vol / (put_vol + 1)  # > 1 = more call activity

        if unusual_calls > unusual_puts and unusual_calls > 0:
            score = min(80 + unusual_calls * 5, 95)
            desc = f"🔥 Unusual call activity ({unusual_calls} strikes > {threshold}x avg) — smart money bullish"
        elif unusual_puts > unusual_calls and unusual_puts > 0:
            score = max(20 - unusual_puts * 5, 5)
            desc = f"⚠️ Unusual put activity ({unusual_puts} strikes > {threshold}x avg) — protection buying"
        elif call_vol_ratio > 2:
            score = 72
            desc = f"Call volume {call_vol_ratio:.1f}x put volume — call-heavy positioning"
        elif call_vol_ratio < 0.5:
            score = 30
            desc = f"Put volume {1/call_vol_ratio:.1f}x call volume — put-heavy positioning"
        else:
            score = 50
            desc = "Normal options activity"

        return round(score, 1), desc

    except Exception:
        return 50.0, "Volume data unavailable"


def _score_iv_percentile(calls: pd.DataFrame, puts: pd.DataFrame) -> tuple:
    """
    IV Percentile score. Works as contrarian at extremes:
    IVP > 90% = excessive fear → contrarian bullish (score 80)
    IVP < 10% = complacency → neutral (score 40)
    IVP 40-70% = normal range → directionally neutral (score 55)
    """
    if calls.empty and puts.empty:
        return 55.0, "IV data unavailable"

    try:
        all_ivs = []
        if not calls.empty and "iv" in calls.columns:
            all_ivs.extend(calls["iv"].dropna().tolist())
        if not puts.empty and "iv" in puts.columns:
            all_ivs.extend(puts["iv"].dropna().tolist())

        if not all_ivs:
            return 55.0, "IV data empty"

        # We don't have historical IV for real percentile calculation
        # Use absolute IV level as proxy (IV > 50% = elevated, < 20% = suppressed)
        avg_iv = np.mean(all_ivs)

        if avg_iv > 60:
            score = 80  # High fear → contrarian bullish
            desc = f"IV elevated ({avg_iv:.1f}%) — fear excessive, contrarian bullish signal"
        elif avg_iv > 40:
            score = 65
            desc = f"IV above normal ({avg_iv:.1f}%) — moderate fear"
        elif avg_iv > 20:
            score = 55
            desc = f"IV normal ({avg_iv:.1f}%) — no edge from volatility"
        elif avg_iv > 10:
            score = 40
            desc = f"IV suppressed ({avg_iv:.1f}%) — complacency, expect move"
        else:
            score = 35
            desc = f"IV very low ({avg_iv:.1f}%) — extreme complacency"

        return round(score, 1), desc

    except Exception:
        return 55.0, "IV calculation error"


def _score_pcr_contrarian(pcr: float) -> tuple:
    """
    PCR as contrarian indicator — NSE-specific thresholds.
    Post-2009, PCR directional edge lost. Use as sentiment extreme indicator.
    NSE Nifty PCR historically oscillates 0.8-1.3.
    """
    cfg = OC_CONFIG

    if pcr >= cfg["pcr_extreme_bullish"]:
        score = 90
        interpretation = f"Extreme fear (PCR {pcr:.2f} > 1.5) → contrarian STRONG BUY"
    elif pcr >= cfg["pcr_bullish"]:
        score = 78
        interpretation = f"High fear (PCR {pcr:.2f}) → contrarian bullish"
    elif pcr >= cfg["pcr_neutral_high"]:
        score = 65
        interpretation = f"Slightly bearish sentiment (PCR {pcr:.2f})"
    elif pcr >= cfg["pcr_neutral_low"]:
        score = 50
        interpretation = f"Neutral (PCR {pcr:.2f})"
    elif pcr >= cfg["pcr_bearish"]:
        score = 40
        interpretation = f"Complacency (PCR {pcr:.2f})"
    elif pcr >= cfg["pcr_extreme_bearish"]:
        score = 25
        interpretation = f"High complacency (PCR {pcr:.2f}) → contrarian bearish"
    else:
        score = 15
        interpretation = f"Extreme complacency (PCR {pcr:.2f} < 0.5) → contrarian STRONG SELL"

    return round(score, 1), interpretation


def _score_max_pain(spot: float, max_pain: float, dte: int) -> tuple:
    """
    Max Pain theory: spot tends to drift toward max pain near expiry.
    Effect strongest in last 5 trading days. Valid mainly for small/illiquid stocks.
    """
    if max_pain <= 0 or spot <= 0:
        return 50.0, ""

    dist_pct = (spot - max_pain) / max_pain * 100

    # Above max pain → expect drift down → slightly bearish
    # Below max pain → expect drift up → slightly bullish
    if dist_pct < -5:
        score = 75  # Well below max pain → bullish drift expected
        desc = f"Spot {dist_pct:.1f}% below Max Pain ₹{max_pain:,.0f} → drift up expected"
    elif dist_pct < -2:
        score = 65
        desc = f"Spot {dist_pct:.1f}% below Max Pain ₹{max_pain:,.0f}"
    elif dist_pct < 2:
        score = 55  # Near max pain → likely to stay
        desc = f"Spot near Max Pain ₹{max_pain:,.0f} (diff: {dist_pct:+.1f}%)"
    elif dist_pct < 5:
        score = 40
        desc = f"Spot {dist_pct:.1f}% above Max Pain ₹{max_pain:,.0f}"
    else:
        score = 30
        desc = f"Spot {dist_pct:.1f}% above Max Pain ₹{max_pain:,.0f} → drift down expected"

    # Reduce confidence if DTE > 10 (max pain loses predictive power far from expiry)
    if dte > 10:
        score = 50 + (score - 50) * 0.5  # Half the signal away from expiry

    return round(score, 1), desc


def _score_iv_skew(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> tuple:
    """
    IV Skew: compare OTM put IV vs OTM call IV.
    Steep put skew = panic protection buying → contrarian bullish (market bottoming).
    Call skew = speculative FOMO → contrarian bearish.
    """
    if calls.empty or puts.empty or spot <= 0:
        return 50.0, "IV skew unavailable"

    try:
        # OTM puts: strikes 5-15% below spot
        otm_puts = puts[(puts["strike"] < spot * 0.97) & (puts["strike"] > spot * 0.85)]
        # OTM calls: strikes 3-15% above spot
        otm_calls = calls[(calls["strike"] > spot * 1.03) & (calls["strike"] < spot * 1.15)]

        otm_put_iv = otm_puts["iv"].mean() if not otm_puts.empty and "iv" in otm_puts else 0
        otm_call_iv = otm_calls["iv"].mean() if not otm_calls.empty and "iv" in otm_calls else 0

        if otm_put_iv <= 0 and otm_call_iv <= 0:
            return 50.0, "OTM IV data unavailable"

        # Skew = put IV - call IV
        skew = otm_put_iv - otm_call_iv

        if skew > 20:
            score = 72  # Steep put skew = panic = contrarian bullish
            desc = f"Put skew steep ({skew:.1f}) — panic buying puts, contrarian bullish"
        elif skew > 10:
            score = 63
            desc = f"Moderate put skew ({skew:.1f}) — elevated protection demand"
        elif skew > 0:
            score = 55
            desc = f"Normal put skew ({skew:.1f}) — market healthy"
        elif skew > -10:
            score = 45
            desc = f"Call skew developing ({skew:.1f}) — speculative call buying"
        else:
            score = 35  # Call skew = FOMO = contrarian bearish
            desc = f"Call skew ({skew:.1f}) — FOMO-driven call buying, caution"

        return round(score, 1), desc

    except Exception:
        return 50.0, "IV skew calculation error"


def _score_iv_spread(iv_spread: float) -> tuple:
    """
    IV Spread = ATM Call IV - ATM Put IV.
    Cremers & Weinbaum (2010): Most robust option chain signal.
    IVS > +2% = calls expensive relative to puts = informed bullish flow → 80 score
    IVS < -2% = puts expensive = informed bearish flow → 20 score
    """
    if iv_spread > 5:
        score = 85
        desc = f"IV Spread +{iv_spread:.1f}% — calls very expensive, strong informed buying"
    elif iv_spread > 2:
        score = 78
        desc = f"IV Spread +{iv_spread:.1f}% — calls premium, bullish (Cremers signal)"
    elif iv_spread > 0.5:
        score = 62
        desc = f"IV Spread +{iv_spread:.1f}% — slight call premium"
    elif iv_spread > -0.5:
        score = 52
        desc = f"IV Spread {iv_spread:.1f}% — near parity, neutral"
    elif iv_spread > -2:
        score = 40
        desc = f"IV Spread {iv_spread:.1f}% — slight put premium"
    elif iv_spread > -5:
        score = 25
        desc = f"IV Spread {iv_spread:.1f}% — puts expensive, bearish (Cremers signal)"
    else:
        score = 15
        desc = f"IV Spread {iv_spread:.1f}% — puts very expensive, strong informed selling"

    return round(score, 1), desc


def _get_weights_with_expiry_mod(dte: int) -> Dict[str, float]:
    """
    Adjust weights based on days to expiry.
    Near expiry (DTE ≤ 3): Max Pain weight → 20%, others proportionally reduced.
    Far expiry (DTE > 30): Max Pain weight → 5%, others proportionally increased.
    """
    base = {
        "w_oi_change": 0.25,
        "w_uoa": 0.20,
        "w_ivp": 0.15,
        "w_pcr": 0.15,
        "w_max_pain": 0.10,
        "w_skew": 0.075,
        "w_ivs": 0.075,
    }

    if dte <= OC_CONFIG["dte_near"]:
        # Boost Max Pain to 20%, reduce others proportionally
        extra = 0.10  # 20% - 10% = extra 10% going to Max Pain
        ratio = (1 - 0.20) / (1 - 0.10)  # Scale others down
        weights = {k: round(v * ratio, 4) for k, v in base.items()}
        weights["w_max_pain"] = 0.20
    elif dte > OC_CONFIG["dte_far"]:
        # Reduce Max Pain to 5%, distribute to OI and UOA
        savings = 0.05  # 10% - 5% saved from Max Pain
        weights = base.copy()
        weights["w_max_pain"] = 0.05
        weights["w_oi_change"] += savings * 0.5
        weights["w_uoa"] += savings * 0.5
    else:
        weights = base.copy()

    # Normalize to sum = 1.0
    total = sum(weights.values())
    return {k: round(v / total, 4) for k, v in weights.items()}


# ============================================================================
# PARTICIPANT ANALYSIS — FII/DII/Retail (NSE official data)
# ============================================================================

def interpret_participant_data(fii_long: float, fii_short: float,
                                retail_long: float, retail_short: float) -> Dict:
    """
    Analyze NSE participant-wise F&O data.
    FII long/short ratio > 1 = bullish, < 1 = bearish.
    Retail is contrarian — when retail is heavily long, often signals top.
    """
    fii_ratio = fii_long / (fii_short + 1)
    retail_ratio = retail_long / (retail_short + 1)

    fii_sentiment = "BULLISH" if fii_ratio > 1.1 else ("BEARISH" if fii_ratio < 0.9 else "NEUTRAL")
    # Retail is contrarian
    retail_signal = "BEARISH" if retail_ratio > 1.2 else ("BULLISH" if retail_ratio < 0.8 else "NEUTRAL")

    return {
        "fii_ratio": round(fii_ratio, 2),
        "fii_sentiment": fii_sentiment,
        "retail_ratio": round(retail_ratio, 2),
        "retail_contrarian_signal": retail_signal,
        "composite": fii_sentiment if fii_sentiment != "NEUTRAL" else retail_signal,
    }
