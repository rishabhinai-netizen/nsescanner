"""
Signal Quality Index (SQI) — Multi-factor signal ranking system
================================================================
Replaces raw confidence scores with evidence-based ranking.

SQI = (0.30 × Backtest_Edge) + (0.25 × RS_Acceleration) + 
      (0.20 × Regime_Fit) + (0.15 × Vol_Contraction) + (0.10 × Volume_Confirm)

Score range: 0-100
- 80+: Elite signal (top priority)
- 65-79: Strong signal (take it)
- 50-64: Moderate signal (smaller size)
- <50: Weak signal (skip or paper-trade only)
"""

from dataclasses import dataclass
from typing import Optional, Dict


# ============================================================================
# BACKTEST EDGE — Pre-computed profit factors per strategy×regime
# ============================================================================

# From our real backtests on 38 NSE stocks (Feb 2025):
# VCP PFs here reflect the NEW rewritten VCP (estimated conservative values)
STRATEGY_REGIME_PF = {
    #                   EXPANSION  ACCUMULATION  DISTRIBUTION  PANIC
    "VCP":              {  "EXPANSION": 1.8, "ACCUMULATION": 1.3, "DISTRIBUTION": 0.6, "PANIC": 0.3 },
    "EMA21_Bounce":     {  "EXPANSION": 1.96, "ACCUMULATION": 1.4, "DISTRIBUTION": 0.8, "PANIC": 0.5 },
    "52WH_Breakout":    {  "EXPANSION": 1.86, "ACCUMULATION": 1.5, "DISTRIBUTION": 0.7, "PANIC": 0.3 },
    "Failed_Breakout_Short": { "EXPANSION": 1.0, "ACCUMULATION": 1.3, "DISTRIBUTION": 1.66, "PANIC": 1.55 },
    "Last30Min_ATH":    {  "EXPANSION": 2.1, "ACCUMULATION": 1.3, "DISTRIBUTION": 0.6, "PANIC": 0.3 },
    "ORB":              {  "EXPANSION": 1.72, "ACCUMULATION": 1.2, "DISTRIBUTION": 0.8, "PANIC": 0.5 },
    "VWAP_Reclaim":     {  "EXPANSION": 1.84, "ACCUMULATION": 1.4, "DISTRIBUTION": 0.9, "PANIC": 0.5 },
    "Lunch_Low":        {  "EXPANSION": 1.52, "ACCUMULATION": 1.3, "DISTRIBUTION": 0.9, "PANIC": 0.6 },
}


@dataclass
class SignalQuality:
    """Signal quality assessment."""
    sqi: float  # 0-100 composite score
    grade: str  # "ELITE" / "STRONG" / "MODERATE" / "WEAK"
    grade_icon: str  # Emoji icon
    backtest_edge: float  # 0-100 component
    rs_accel_score: float  # 0-100 component
    regime_fit_score: float  # 0-100 component
    vol_contraction_score: float  # 0-100 component
    volume_confirm_score: float  # 0-100 component
    breakdown: str  # Human-readable breakdown


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
    """
    Compute Signal Quality Index.
    
    Args:
        strategy: Strategy name (e.g., "VCP", "EMA21_Bounce")
        regime: Current market regime ("EXPANSION", etc.)
        rs_rating: Stock's RS rating (0-100)
        rs_acceleration: RS slope (negative to positive)
        vol_compression_ratio: std_10/std_50 (lower = more compressed)
        volume_ratio: Current volume vs average
        confidence: Raw confidence from scanner (0-100)
        vol_down_day_ratio: Down-day vol / up-day vol (< 1 = bullish)
        weekly_aligned: Whether weekly timeframe confirms
        rolling_pf: If available, rolling profit factor for this strategy
    """
    
    # ── Component 1: Backtest Edge (30% weight) ──
    # Based on pre-computed profit factors per strategy×regime
    pf = STRATEGY_REGIME_PF.get(strategy, {}).get(regime, 1.0)
    
    # Override with rolling PF if available and we have enough data
    if rolling_pf is not None and rolling_pf > 0:
        pf = rolling_pf
    
    # Map PF to 0-100 score: PF 0.5→10, PF 1.0→40, PF 1.5→65, PF 2.0→85, PF 3.0→100
    if pf <= 0.5:
        backtest_edge = 10
    elif pf <= 1.0:
        backtest_edge = 10 + (pf - 0.5) * 60  # 10-40
    elif pf <= 2.0:
        backtest_edge = 40 + (pf - 1.0) * 45  # 40-85
    else:
        backtest_edge = min(85 + (pf - 2.0) * 15, 100)  # 85-100
    
    # ── Component 2: RS Acceleration (25% weight) ──
    # RS acceleration typically ranges from -1.5 to +1.5 points/day
    # Positive = improving relative strength
    rs_base = min(max(rs_rating, 0), 100)  # Raw RS as base
    
    # Acceleration bonus/penalty: +0.5/day → +15 points, -0.5/day → -15 points
    accel_bonus = min(max(rs_acceleration * 30, -30), 30)
    rs_accel_score = min(max(rs_base * 0.6 + 20 + accel_bonus, 0), 100)
    
    # ── Component 3: Regime Fit (20% weight) ──
    # How well does this strategy fit the current regime?
    regime_fit_map = {
        # (strategy_type, regime) → fit_score
    }
    
    # Simple approach: use the PF mapping
    if pf >= 1.8:
        regime_fit_score = 90
    elif pf >= 1.3:
        regime_fit_score = 70
    elif pf >= 1.0:
        regime_fit_score = 50
    elif pf >= 0.7:
        regime_fit_score = 30
    else:
        regime_fit_score = 10
    
    # Weekly alignment bonus
    if weekly_aligned:
        regime_fit_score = min(regime_fit_score + 10, 100)
    
    # ── Component 4: Volatility Contraction (15% weight) ──
    # For long strategies: lower vol compression = better (tight base)
    # For shorts: higher vol = better (volatile breakdown)
    is_short = strategy in ("Failed_Breakout_Short",)
    
    if is_short:
        # Shorts want volatility expansion
        if vol_compression_ratio > 1.2:
            vol_contraction_score = 80
        elif vol_compression_ratio > 0.8:
            vol_contraction_score = 60
        else:
            vol_contraction_score = 40
    else:
        # Longs want volatility contraction (tight bases)
        if vol_compression_ratio < 0.4:
            vol_contraction_score = 95  # Very tight — excellent VCP
        elif vol_compression_ratio < 0.5:
            vol_contraction_score = 80
        elif vol_compression_ratio < 0.7:
            vol_contraction_score = 60
        elif vol_compression_ratio < 1.0:
            vol_contraction_score = 40
        else:
            vol_contraction_score = 20
    
    # Down-day volume bonus for longs (accumulation signal)
    if not is_short and vol_down_day_ratio < 0.8:
        vol_contraction_score = min(vol_contraction_score + 10, 100)
    
    # ── Component 5: Volume Confirmation (10% weight) ──
    # Breakout/bounce should have above-average volume
    if volume_ratio >= 2.0:
        volume_confirm_score = 95
    elif volume_ratio >= 1.5:
        volume_confirm_score = 80
    elif volume_ratio >= 1.2:
        volume_confirm_score = 65
    elif volume_ratio >= 0.8:
        volume_confirm_score = 45
    else:
        volume_confirm_score = 25  # Low volume breakout = suspect
    
    # ── Composite SQI ──
    sqi = (
        0.30 * backtest_edge +
        0.25 * rs_accel_score +
        0.20 * regime_fit_score +
        0.15 * vol_contraction_score +
        0.10 * volume_confirm_score
    )
    
    sqi = round(min(max(sqi, 0), 100), 1)
    
    # ── Grade ──
    if sqi >= 80:
        grade, grade_icon = "ELITE", "🏆"
    elif sqi >= 65:
        grade, grade_icon = "STRONG", "💪"
    elif sqi >= 50:
        grade, grade_icon = "MODERATE", "⚡"
    else:
        grade, grade_icon = "WEAK", "⚠️"
    
    breakdown = (
        f"Edge:{backtest_edge:.0f} RS:{rs_accel_score:.0f} "
        f"Regime:{regime_fit_score:.0f} Vol:{vol_contraction_score:.0f} "
        f"Confirm:{volume_confirm_score:.0f}"
    )
    
    return SignalQuality(
        sqi=sqi,
        grade=grade,
        grade_icon=grade_icon,
        backtest_edge=round(backtest_edge, 1),
        rs_accel_score=round(rs_accel_score, 1),
        regime_fit_score=round(regime_fit_score, 1),
        vol_contraction_score=round(vol_contraction_score, 1),
        volume_confirm_score=round(volume_confirm_score, 1),
        breakdown=breakdown,
    )


def is_strategy_allowed(strategy: str, regime: str, min_pf: float = 1.0) -> bool:
    """Check if strategy has a positive edge in current regime."""
    pf = STRATEGY_REGIME_PF.get(strategy, {}).get(regime, 0.5)
    return pf >= min_pf


def get_regime_strategy_matrix() -> Dict[str, Dict[str, float]]:
    """Return the full strategy×regime profit factor matrix for display."""
    return STRATEGY_REGIME_PF
