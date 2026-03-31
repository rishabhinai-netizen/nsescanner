"""
Signal Quality Index (SQI) v2
=================================
Now reads live blended profit factors from Supabase strategy_performance table.
Falls back to hardcoded priors if Supabase is unavailable.
auto_learner.py updates the Supabase table weekly.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import logging, os

logger = logging.getLogger(__name__)

# ============================================================
# HARDCODED PRIORS — used when Supabase unavailable
# ============================================================

STRATEGY_REGIME_PF = {
    "VCP":                   {"EXPANSION": 1.80, "ACCUMULATION": 1.30, "DISTRIBUTION": 0.60, "PANIC": 0.30},
    "EMA21_Bounce":          {"EXPANSION": 1.96, "ACCUMULATION": 1.40, "DISTRIBUTION": 0.80, "PANIC": 0.50},
    "52WH_Breakout":         {"EXPANSION": 1.86, "ACCUMULATION": 1.50, "DISTRIBUTION": 0.70, "PANIC": 0.30},
    "Failed_Breakout_Short": {"EXPANSION": 1.00, "ACCUMULATION": 1.30, "DISTRIBUTION": 1.66, "PANIC": 1.55},
    "Last30Min_ATH":         {"EXPANSION": 2.10, "ACCUMULATION": 1.30, "DISTRIBUTION": 0.60, "PANIC": 0.30},
    "ORB":                   {"EXPANSION": 1.72, "ACCUMULATION": 1.20, "DISTRIBUTION": 0.80, "PANIC": 0.50},
    "VWAP_Reclaim":          {"EXPANSION": 1.84, "ACCUMULATION": 1.40, "DISTRIBUTION": 0.90, "PANIC": 0.50},
    "Lunch_Low":             {"EXPANSION": 1.52, "ACCUMULATION": 1.30, "DISTRIBUTION": 0.90, "PANIC": 0.60},
}

# In-process cache so we don't hit Supabase on every scan
_live_pf_cache: Dict[str, Dict[str, float]] = {}
_cache_loaded = False


def _load_live_pf() -> Dict[str, Dict[str, float]]:
    """Load live blended PF from Supabase. Returns hardcoded priors on failure."""
    global _live_pf_cache, _cache_loaded
    if _cache_loaded:
        return _live_pf_cache

    _cache_loaded = True  # set even on failure to avoid repeated attempts

    try:
        url = _get_secret("SUPABASE_URL")
        key = _get_secret("SUPABASE_SERVICE_KEY") or _get_secret("SUPABASE_ANON_KEY")
        if not url or not key:
            _live_pf_cache = {s: dict(r) for s, r in STRATEGY_REGIME_PF.items()}
            return _live_pf_cache

        from supabase import create_client
        sb = create_client(url, key)
        resp = sb.table("strategy_performance").select(
            "strategy,regime,blended_pf,trade_count"
        ).execute()

        if not resp.data:
            _live_pf_cache = {s: dict(r) for s, r in STRATEGY_REGIME_PF.items()}
            return _live_pf_cache

        # Build live dict
        live: Dict[str, Dict[str, float]] = {}
        for row in resp.data:
            strat  = row["strategy"]
            regime = row["regime"]
            # Use blended_pf only if we have enough real trades (≥ 10)
            if int(row.get("trade_count", 0)) >= 10:
                pf = float(row.get("blended_pf") or STRATEGY_REGIME_PF.get(strat, {}).get(regime, 1.0))
            else:
                pf = float(STRATEGY_REGIME_PF.get(strat, {}).get(regime, 1.0))
            if strat not in live:
                live[strat] = {}
            live[strat][regime] = round(pf, 3)

        # Fill missing entries from priors
        for strat, regimes in STRATEGY_REGIME_PF.items():
            if strat not in live:
                live[strat] = dict(regimes)
            else:
                for regime, pf in regimes.items():
                    if regime not in live[strat]:
                        live[strat][regime] = pf

        _live_pf_cache = live
        logger.info(f"SQI: loaded live PF for {len(live)} strategies from Supabase")
        return _live_pf_cache

    except Exception as e:
        logger.warning(f"SQI: Supabase PF load failed ({e}) — using priors")
        _live_pf_cache = {s: dict(r) for s, r in STRATEGY_REGIME_PF.items()}
        return _live_pf_cache


def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val: return val
    except Exception:
        pass
    return os.environ.get(key, "")


def reset_pf_cache():
    """Call this after auto_learner updates the DB, or at app startup."""
    global _cache_loaded
    _cache_loaded = False


# ============================================================
# SQI COMPUTATION
# ============================================================

@dataclass
class SignalQuality:
    sqi: float
    grade: str
    grade_icon: str
    backtest_edge: float
    rs_accel_score: float
    regime_fit_score: float
    vol_contraction_score: float
    volume_confirm_score: float
    breakdown: str
    live_pf: float = 0.0
    is_live_pf: bool = False  # True if PF came from real trade data


def compute_sqi(
    strategy: str,
    regime: str,
    rs_rating: float = 50.0,
    rs_acceleration: float = 0.0,
    vol_compression_ratio: float = 1.0,
    volume_ratio: float = 1.0,
    confidence: int = 50,
    vol_down_day_ratio: float = 1.0,
    weekly_aligned: bool = False,
    rolling_pf: Optional[float] = None,
) -> SignalQuality:

    # ── Component 1: Backtest Edge (30%) — from live Supabase PF ──
    live_pf_dict = _load_live_pf()
    pf = live_pf_dict.get(strategy, {}).get(regime, 1.0)
    is_live_pf = False

    # Override with caller-supplied rolling PF (from strategy health tracker)
    if rolling_pf is not None and rolling_pf > 0:
        pf = rolling_pf
        is_live_pf = True
    elif live_pf_dict.get(strategy, {}).get(regime) is not None:
        is_live_pf = True

    if pf <= 0.5:
        backtest_edge = 10
    elif pf <= 1.0:
        backtest_edge = 10 + (pf - 0.5) * 60
    elif pf <= 2.0:
        backtest_edge = 40 + (pf - 1.0) * 45
    else:
        backtest_edge = min(85 + (pf - 2.0) * 15, 100)

    # ── Component 2: RS Acceleration (25%) ──
    rs_base        = min(max(rs_rating, 0), 100)
    accel_bonus    = min(max(rs_acceleration * 30, -30), 30)
    rs_accel_score = min(max(rs_base * 0.6 + 20 + accel_bonus, 0), 100)

    # ── Component 3: Regime Fit (20%) ──
    if pf >= 1.8: regime_fit_score = 90
    elif pf >= 1.3: regime_fit_score = 70
    elif pf >= 1.0: regime_fit_score = 50
    elif pf >= 0.7: regime_fit_score = 30
    else: regime_fit_score = 10
    if weekly_aligned:
        regime_fit_score = min(regime_fit_score + 10, 100)

    # ── Component 4: Volatility Contraction (15%) ──
    is_short = strategy in ("Failed_Breakout_Short",)
    if is_short:
        if vol_compression_ratio > 1.2:  vol_contraction_score = 80
        elif vol_compression_ratio > 0.8: vol_contraction_score = 60
        else: vol_contraction_score = 40
    else:
        if vol_compression_ratio < 0.35:  vol_contraction_score = 95
        elif vol_compression_ratio < 0.50: vol_contraction_score = 80
        elif vol_compression_ratio < 0.70: vol_contraction_score = 60
        elif vol_compression_ratio < 1.00: vol_contraction_score = 40
        else: vol_contraction_score = 20
    if not is_short and vol_down_day_ratio < 0.8:
        vol_contraction_score = min(vol_contraction_score + 10, 100)

    # ── Component 5: Volume Confirmation (10%) ──
    if volume_ratio >= 2.0:     volume_confirm_score = 95
    elif volume_ratio >= 1.5:   volume_confirm_score = 80
    elif volume_ratio >= 1.2:   volume_confirm_score = 65
    elif volume_ratio >= 0.8:   volume_confirm_score = 45
    else:                        volume_confirm_score = 25

    # ── Composite ──
    sqi = round(min(max(
        0.30 * backtest_edge +
        0.25 * rs_accel_score +
        0.20 * regime_fit_score +
        0.15 * vol_contraction_score +
        0.10 * volume_confirm_score,
        0), 100), 1)

    if sqi >= 80:   grade, grade_icon = "ELITE",    "🏆"
    elif sqi >= 65: grade, grade_icon = "STRONG",   "💪"
    elif sqi >= 50: grade, grade_icon = "MODERATE", "⚡"
    else:           grade, grade_icon = "WEAK",     "⚠️"

    pf_tag = f"{'live' if is_live_pf else 'prior'} PF:{pf:.2f}"
    breakdown = (f"Edge:{backtest_edge:.0f}({pf_tag}) RS:{rs_accel_score:.0f} "
                 f"Regime:{regime_fit_score:.0f} Vol:{vol_contraction_score:.0f} "
                 f"Confirm:{volume_confirm_score:.0f}")

    return SignalQuality(
        sqi=sqi, grade=grade, grade_icon=grade_icon,
        backtest_edge=round(backtest_edge, 1),
        rs_accel_score=round(rs_accel_score, 1),
        regime_fit_score=round(regime_fit_score, 1),
        vol_contraction_score=round(vol_contraction_score, 1),
        volume_confirm_score=round(volume_confirm_score, 1),
        breakdown=breakdown,
        live_pf=round(pf, 3),
        is_live_pf=is_live_pf,
    )


def is_strategy_allowed(strategy: str, regime: str, min_pf: float = 1.0) -> bool:
    live_pf_dict = _load_live_pf()
    pf = live_pf_dict.get(strategy, {}).get(regime, 0.5)
    return pf >= min_pf


def get_regime_strategy_matrix() -> Dict[str, Dict[str, float]]:
    """Return live matrix (Supabase) for dashboard display."""
    return _load_live_pf()
