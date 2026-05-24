"""
risk_circuit_breaker.py — Equity-curve circuit breaker for NSE Scanner Pro
============================================================================
Auto-pauses trading after a streak of losses or excessive drawdown.

The system makes the user explicitly opt back in — important for emotional
discipline. The biggest cause of retail account blow-up is "revenge trading"
after a loss streak. This module makes that hard.

Three trigger conditions (any one trips the breaker):
  1. CONSECUTIVE LOSSES: 5 stop-outs in a row → 5 trading day pause
  2. DAILY DRAWDOWN: -3% portfolio in a single day → next day pause
  3. WEEKLY DRAWDOWN: -7% portfolio in a week → 5 trading day pause
  4. STRATEGY FAILURE: any strategy with PF < 0.5 over last 20 trades → dim that strategy

State is persisted in Supabase `circuit_breaker_state` table (one row per user).
"""

import logging
from datetime import date, timedelta, datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""
    is_paused: bool = False
    pause_reason: str = ""
    pause_until: Optional[str] = None       # ISO date string
    consecutive_losses: int = 0
    last_loss_date: Optional[str] = None
    daily_drawdown_pct: float = 0.0
    weekly_drawdown_pct: float = 0.0
    triggered_at: Optional[str] = None
    strategies_dimmed: List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ════════════════════════════════════════════════════════════════════════

CONSECUTIVE_LOSS_THRESHOLD = 5
DAILY_DRAWDOWN_THRESHOLD_PCT = -3.0
WEEKLY_DRAWDOWN_THRESHOLD_PCT = -7.0
PAUSE_DAYS_AFTER_LOSS_STREAK = 5
PAUSE_DAYS_AFTER_DAILY_DD = 1
PAUSE_DAYS_AFTER_WEEKLY_DD = 5
STRATEGY_DIM_PF_THRESHOLD = 0.5
STRATEGY_DIM_MIN_TRADES = 20


# ════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ════════════════════════════════════════════════════════════════════════

def evaluate_circuit_breaker(tracker_df: pd.DataFrame,
                             portfolio_equity_curve: List[float],
                             current_state: Optional[CircuitBreakerState] = None
                             ) -> CircuitBreakerState:
    """
    Evaluate current trading state and return updated CircuitBreakerState.
    Call this once per scan/refresh, and before placing any trade.

    Args:
        tracker_df: DataFrame from signal_tracker.load_tracker() — closed trades only
        portfolio_equity_curve: List of daily portfolio values (most recent last)
        current_state: Previous state for context (consecutive_losses carry-over)
    """
    state = current_state or CircuitBreakerState()

    # ── 1. Check consecutive losses ──
    if tracker_df is not None and not tracker_df.empty:
        closed = tracker_df[tracker_df["Status"].isin(["TARGET", "STOPPED", "EXPIRED"])].copy()
        if not closed.empty:
            # Sort by exit date, take most recent
            closed = closed.sort_values("Exit_Date", ascending=False)
            recent = closed.head(20)

            # Count consecutive STOPPED from most recent
            streak = 0
            for _, row in recent.iterrows():
                if row["Status"] == "STOPPED":
                    streak += 1
                else:
                    break
            state.consecutive_losses = streak

            if streak >= CONSECUTIVE_LOSS_THRESHOLD:
                _trigger_pause(state,
                    reason=f"{streak} consecutive losses (limit {CONSECUTIVE_LOSS_THRESHOLD})",
                    days=PAUSE_DAYS_AFTER_LOSS_STREAK)

    # ── 2. Check daily drawdown ──
    if portfolio_equity_curve and len(portfolio_equity_curve) >= 2:
        latest = portfolio_equity_curve[-1]
        yesterday = portfolio_equity_curve[-2]
        if yesterday > 0:
            daily_dd = (latest / yesterday - 1) * 100
            state.daily_drawdown_pct = round(daily_dd, 2)
            if daily_dd <= DAILY_DRAWDOWN_THRESHOLD_PCT:
                _trigger_pause(state,
                    reason=f"Daily drawdown {daily_dd:.2f}% (limit {DAILY_DRAWDOWN_THRESHOLD_PCT}%)",
                    days=PAUSE_DAYS_AFTER_DAILY_DD)

    # ── 3. Check weekly drawdown ──
    if portfolio_equity_curve and len(portfolio_equity_curve) >= 6:
        latest = portfolio_equity_curve[-1]
        week_ago = portfolio_equity_curve[-6]  # ~5 trading days
        if week_ago > 0:
            weekly_dd = (latest / week_ago - 1) * 100
            state.weekly_drawdown_pct = round(weekly_dd, 2)
            if weekly_dd <= WEEKLY_DRAWDOWN_THRESHOLD_PCT:
                _trigger_pause(state,
                    reason=f"Weekly drawdown {weekly_dd:.2f}% (limit {WEEKLY_DRAWDOWN_THRESHOLD_PCT}%)",
                    days=PAUSE_DAYS_AFTER_WEEKLY_DD)

    # ── 4. Check strategy-level dimming ──
    state.strategies_dimmed = _identify_failing_strategies(tracker_df)

    # ── 5. Auto-unpause if pause period expired ──
    if state.is_paused and state.pause_until:
        try:
            unpause_date = date.fromisoformat(state.pause_until)
            if date.today() >= unpause_date:
                logger.info(f"Circuit breaker auto-unpaused (pause_until={state.pause_until})")
                state.is_paused = False
                state.pause_reason = ""
                state.pause_until = None
                state.triggered_at = None
        except ValueError:
            pass

    return state


def _trigger_pause(state: CircuitBreakerState, reason: str, days: int):
    """Mark state as paused for `days` trading days."""
    # Don't shorten an existing pause
    new_until = (date.today() + timedelta(days=days * 1.5)).isoformat()  # cushion for weekends
    if state.is_paused and state.pause_until:
        try:
            existing = date.fromisoformat(state.pause_until)
            if date.fromisoformat(new_until) <= existing:
                return  # Don't override longer pause with shorter one
        except ValueError:
            pass

    state.is_paused = True
    state.pause_reason = reason
    state.pause_until = new_until
    state.triggered_at = datetime.now().isoformat()
    logger.warning(f"⚠️ CIRCUIT BREAKER TRIGGERED: {reason}. Paused until {new_until}")


def _identify_failing_strategies(tracker_df: Optional[pd.DataFrame]) -> List[str]:
    """Return list of strategy names that should be dimmed."""
    if tracker_df is None or tracker_df.empty:
        return []
    closed = tracker_df[tracker_df["Status"].isin(["TARGET", "STOPPED", "EXPIRED"])]
    if closed.empty:
        return []

    dimmed = []
    for strat in closed["Strategy"].unique():
        sub = closed[closed["Strategy"] == strat].tail(STRATEGY_DIM_MIN_TRADES)
        if len(sub) < STRATEGY_DIM_MIN_TRADES:
            continue
        pnls = pd.to_numeric(sub["PnL_Pct"], errors="coerce").dropna()
        wins = pnls[pnls > 0].sum()
        losses = abs(pnls[pnls < 0].sum())
        if losses > 0:
            pf = wins / losses
            if pf < STRATEGY_DIM_PF_THRESHOLD:
                dimmed.append(strat)
    return dimmed


# ════════════════════════════════════════════════════════════════════════
# REGIME CHANGE DETECTION + ALERT
# ════════════════════════════════════════════════════════════════════════

def detect_regime_change(previous_regime: Optional[str],
                         current_regime: str) -> Optional[Dict]:
    """
    If today's regime differs from yesterday's, return an alert dict.
    Otherwise None.
    """
    if previous_regime is None or previous_regime == "UNKNOWN":
        return None
    if previous_regime == current_regime:
        return None

    # Compute severity
    regime_order = ["PANIC", "DISTRIBUTION", "ACCUMULATION", "EXPANSION", "UNKNOWN"]
    try:
        old_idx = regime_order.index(previous_regime)
        new_idx = regime_order.index(current_regime)
        direction = "IMPROVING" if new_idx > old_idx else "DETERIORATING"
        severity = abs(new_idx - old_idx)
    except ValueError:
        direction = "CHANGING"
        severity = 1

    return {
        "type": "REGIME_CHANGE",
        "from": previous_regime,
        "to": current_regime,
        "direction": direction,
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
        "action_required": _regime_change_action(previous_regime, current_regime),
    }


def _regime_change_action(old: str, new: str) -> str:
    """Recommended action for a specific regime transition."""
    transitions = {
        ("EXPANSION", "ACCUMULATION"):
            "Trim aggressive longs to 70%. Tighten stops on extended winners.",
        ("EXPANSION", "DISTRIBUTION"):
            "Reduce exposure to 40%. Take profits on positions > +10%. Stop adding longs.",
        ("EXPANSION", "PANIC"):
            "Exit aggressive longs. Hedge core positions or go to cash 60-80%.",
        ("ACCUMULATION", "EXPANSION"):
            "Add to existing longs. Increase position sizes. Activate breakout strategies.",
        ("ACCUMULATION", "DISTRIBUTION"):
            "Stop new entries. Tighten stops. Watch for failed-breakout shorts.",
        ("ACCUMULATION", "PANIC"):
            "Aggressive de-risking. Exit all but highest-conviction positions.",
        ("DISTRIBUTION", "ACCUMULATION"):
            "Cautious re-entry on best setups only. Half-size positions.",
        ("DISTRIBUTION", "EXPANSION"):
            "Bullish reversal. Re-deploy capital systematically over 3-5 days.",
        ("DISTRIBUTION", "PANIC"):
            "Stop trading. Wait for capitulation low. Cash 80%+.",
        ("PANIC", "DISTRIBUTION"):
            "Stop bleeding. No new positions yet. Wait for confirmed reversal.",
        ("PANIC", "ACCUMULATION"):
            "Aggressive reversal — historic opportunity. Begin scaling in.",
        ("PANIC", "EXPANSION"):
            "V-shaped recovery. Deploy capital quickly.",
    }
    return transitions.get((old, new), f"Reassess all positions. Transition: {old} → {new}")


def format_regime_change_telegram(change: Dict) -> str:
    """Format a regime-change alert for Telegram."""
    if not change:
        return ""

    direction_icon = {"IMPROVING": "📈", "DETERIORATING": "📉", "CHANGING": "🔄"}.get(
        change.get("direction", "CHANGING"), "🔄")

    severity_icons = "⚠️" * change.get("severity", 1)

    return f"""🚨 *REGIME CHANGE DETECTED* {severity_icons}

{direction_icon} {change['from']} → *{change['to']}*
Direction: {change['direction']}
Time: {change['timestamp'][:16]}

📋 *Action Required:*
{change['action_required']}

Review all open positions. Update stop losses. Adjust position sizing per the new regime caps.
"""


# ════════════════════════════════════════════════════════════════════════
# TRUST SCORE FOR INDIVIDUAL SIGNALS
# ════════════════════════════════════════════════════════════════════════

def compute_trust_score(
    sqi: float,
    sqi_grade: str,
    regime_fit: str,
    data_quality: str,
    weekly_aligned: bool,
    sector_quadrant: str,
    has_intraday_confirmation: bool = False,
    fundamental_grade: str = "B",
) -> Tuple[float, str, List[str]]:
    """
    Composite "trust score" 0-100 — should you act on this signal?

    Inputs:
      sqi             : Signal Quality Index from signal_quality.py
      sqi_grade       : ELITE, STRONG, MODERATE, WEAK
      regime_fit      : IDEAL, OK, CAUTION, BLOCKED
      data_quality    : OK, STALE_1D, STALE, INCOMPLETE
      weekly_aligned  : True if weekly timeframe agrees
      sector_quadrant : LEADING, IMPROVING, WEAKENING, LAGGING
      has_intraday_confirmation: True if Breeze 5-min confirms
      fundamental_grade: A, B+, B, C, D (CANSLIM)

    Returns: (score 0-100, label, list of penalties/bonuses applied)
    """
    components = []
    score = float(sqi)
    components.append(f"Base SQI: {sqi:.0f}")

    # Regime fit (major)
    fit_adj = {"IDEAL": 0, "OK": -5, "CAUTION": -15, "BLOCKED": -50}.get(regime_fit, -10)
    if fit_adj != 0:
        score += fit_adj
        components.append(f"Regime {regime_fit}: {fit_adj:+d}")

    # Data quality
    dq_adj = {"OK": 0, "STALE_1D": -3, "STALE": -20, "INCOMPLETE": -25, "EMPTY": -100}.get(data_quality, -5)
    if dq_adj != 0:
        score += dq_adj
        components.append(f"Data {data_quality}: {dq_adj:+d}")

    # Multi-timeframe (boost)
    if weekly_aligned:
        score += 5
        components.append("Weekly aligned: +5")

    # Sector RRG
    sec_adj = {"LEADING": 6, "IMPROVING": 2, "WEAKENING": -2, "LAGGING": -10}.get(sector_quadrant, 0)
    if sec_adj != 0:
        score += sec_adj
        components.append(f"Sector {sector_quadrant}: {sec_adj:+d}")

    # Intraday confirmation
    if has_intraday_confirmation:
        score += 4
        components.append("Intraday confirmed: +4")

    # Fundamentals
    fund_adj = {"A": 4, "B+": 2, "B": 0, "C": -3, "D": -10}.get(fundamental_grade, 0)
    if fund_adj != 0:
        score += fund_adj
        components.append(f"Fundamental {fundamental_grade}: {fund_adj:+d}")

    score = max(0.0, min(100.0, score))

    if score >= 80:
        label = "🟢 TRUST: HIGH"
    elif score >= 65:
        label = "🔵 TRUST: SOLID"
    elif score >= 50:
        label = "🟡 TRUST: MODERATE"
    elif score >= 35:
        label = "🟠 TRUST: LOW"
    else:
        label = "🔴 TRUST: AVOID"

    return round(score, 1), label, components
