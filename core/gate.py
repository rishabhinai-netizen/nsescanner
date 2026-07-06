"""The Gate — decides which signals go LIVE (alerted), which INCUBATE
(tracked silently), which get BLOCKED. This is the v2 core innovation:
in v15 the regime 'gate' only annotated; here it actually gates.

Every weight below is derived from 966 forward-tested EMA21 trades
(valid rows, Apr–Jul 2026) — not from hardcoded priors:

  RS >= 90            61.5% WR, +2.2%   → biggest positive factor
  RS 70–80            38.3% WR, +0.02%  → dead zone, penalized
  ACCUMULATION        59.5% WR, +2.0%   → best regime
  PANIC (RS>=70)      55.1% WR, +1.5%   → bounces work in panic (counter-intuitive, proven)
  DISTRIBUTION        39.0% WR, +0.19%  → the regime to block
  RR 2.0–3.0          sweet spot; RR>3 avg fell to +0.49%
"""
from typing import Dict, Optional
from core.strategies import Signal

PROMOTION_MIN_TRADES = 30
PROMOTION_MIN_PF = 1.2


def sqi_v2(sig: Signal, regime: str) -> float:
    """Signal Quality Index 0–100. Evidence-weighted."""
    score = 40.0
    rs = sig.rs_rank
    if rs is not None:
        if rs >= 90:   score += 30
        elif rs >= 80: score += 15
        elif rs >= 70: score -= 10   # dead zone
        elif rs >= 50: score += 5
        else:          score -= 5
    else:
        score -= 10  # unknown RS = uncertainty penalty

    score += {"ACCUMULATION": 25, "EXPANSION": 15,   # EXPANSION untested live — provisional
              "PANIC": 10, "DISTRIBUTION": -25}.get(regime, 0)

    if 2.0 <= sig.rr <= 3.0: score += 5
    elif sig.rr > 3.0:       score -= 10

    return round(max(0, min(100, score)), 1)


def tier(sqi: float) -> str:
    return "A" if sqi >= 75 else ("B" if sqi >= 55 else "C")


def gate(sig: Signal, regime: str, default_gate: str,
         live_stats: Dict[str, Dict] = None) -> Dict:
    """Returns {'gate': LIVE|INCUBATING|BLOCKED, 'sqi': x, 'tier': A|B|C, 'reason': str}."""
    s = sqi_v2(sig, regime)
    t = tier(s)

    # Hard blocks — regardless of strategy
    if regime == "DISTRIBUTION" and sig.side == "LONG" and (sig.rs_rank or 0) < 90:
        return {"gate": "BLOCKED", "sqi": s, "tier": t,
                "reason": "Long in DISTRIBUTION with RS<90 (39% WR live)"}
    if t == "C":
        return {"gate": "INCUBATING", "sqi": s, "tier": t, "reason": "SQI below B threshold"}

    if default_gate == "INCUBATING":
        # Promotion check: live PF from nx_strategy_stats
        st = (live_stats or {}).get(f"{sig.strategy}|{regime}")
        if st and st.get("n_closed", 0) >= PROMOTION_MIN_TRADES \
              and (st.get("profit_factor") or 0) >= PROMOTION_MIN_PF:
            return {"gate": "LIVE", "sqi": s, "tier": t,
                    "reason": f"Promoted: PF {st['profit_factor']} on n={st['n_closed']}"}
        return {"gate": "INCUBATING", "sqi": s, "tier": t, "reason": "Building live sample"}

    # LIVE strategy: demotion check — if live PF collapses, demote
    st = (live_stats or {}).get(f"{sig.strategy}|{regime}")
    if st and st.get("n_closed", 0) >= PROMOTION_MIN_TRADES \
          and (st.get("profit_factor") or 9) < 0.8:
        return {"gate": "INCUBATING", "sqi": s, "tier": t,
                "reason": f"Demoted: live PF {st['profit_factor']} < 0.8"}

    return {"gate": "LIVE", "sqi": s, "tier": t, "reason": "Passed gate"}
