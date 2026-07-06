"""Strategies — one LIVE strategy (earned it), everything else INCUBATING.

Forward-test verdict (Apr–Jul 2026, valid rows only):
  EMA21_Bounce            966 closed | +1.33% avg | RS>=90: 61.5% WR +2.2%
  52WH_Breakout           KILLED  (10.8% WR, -1.66% avg, negative in every regime)
  VCP                     KILLED  (0/18, -4.70% avg)
  Last30Min_ATH           KILLED  (10.2% WR, -0.40%)
  Failed_Breakout_Short   KILLED  (-1.11% avg; negative even in DISTRIBUTION)

A new strategy enters LIVE only after >=30 closed INCUBATING trades with
profit factor >= 1.2 in at least one regime (see gate.promote_check)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class Signal:
    strategy: str
    symbol: str
    side: str            # LONG / SHORT
    entry: float
    stop: float
    target1: float
    target2: Optional[float]
    rr: float
    rs_rank: Optional[float] = None
    sector: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _rs_rank(universe_returns: Dict[str, float], symbol: str) -> Optional[float]:
    """Percentile rank of 3-month return within the scanned universe (0-100)."""
    vals = sorted(universe_returns.values())
    if symbol not in universe_returns or len(vals) < 20:
        return None
    r = universe_returns[symbol]
    return round(sum(v <= r for v in vals) / len(vals) * 100, 1)


def compute_universe_returns(data: Dict[str, pd.DataFrame], lookback: int = 63) -> Dict[str, float]:
    out = {}
    for s, df in data.items():
        if len(df) > lookback:
            out[s] = float(df["Close"].iloc[-1] / df["Close"].iloc[-lookback - 1] - 1)
    return out


# ------------------------------------------------------- EMA21 Bounce (LIVE) --
def scan_ema21_bounce(df: pd.DataFrame, symbol: str,
                      rs: Optional[float]) -> Optional[Signal]:
    """Uptrending stock pulls back to rising 21 EMA, holds, closes strong.
    RR capped in the 2.0–3.0 band — the forward-tested sweet spot
    (RR>3 setups averaged just +0.49%)."""
    if len(df) < 120:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    e21 = c.ewm(span=21).mean()
    e50 = c.ewm(span=50).mean()
    px = float(c.iloc[-1])

    uptrend = px > float(e50.iloc[-1]) and float(e21.iloc[-1]) > float(e21.iloc[-6])
    if not uptrend:
        return None

    # Pullback: low tagged/near the 21 EMA within last 3 bars
    near = ((l.tail(3) - e21.tail(3)) / e21.tail(3)).min()
    touched = -0.015 <= float(near) <= 0.01
    if not touched:
        return None

    # Hold + strength: today closes above 21 EMA, upper half of range
    rng = float(h.iloc[-1] - l.iloc[-1])
    strong = px > float(e21.iloc[-1]) and rng > 0 and (px - float(l.iloc[-1])) / rng >= 0.5
    if not strong:
        return None

    # Volume not collapsing
    if float(v.iloc[-1]) < 0.6 * float(v.tail(20).mean()):
        return None

    atr = float((h - l).tail(14).mean())
    entry = px
    stop = round(min(float(l.tail(3).min()), float(e21.iloc[-1])) - 0.25 * atr, 2)
    risk = entry - stop
    if risk <= 0 or risk / entry > 0.06:
        return None
    t1 = round(entry + 2.0 * risk, 2)
    t2 = round(entry + 3.0 * risk, 2)
    return Signal("EMA21_Bounce", symbol, "LONG", round(entry, 2), stop, t1, t2,
                  rr=2.0, rs_rank=rs, meta={"atr": round(atr, 2)})


# --------------------------------------------- RS Leader Pullback (INCUBATING) --
def scan_rs_leader_pullback(df: pd.DataFrame, symbol: str,
                            rs: Optional[float]) -> Optional[Signal]:
    """Candidate: RS>=90 leader, 3–8% off 20d high, reclaim of prior day high.
    Ships INCUBATING — tracked, never alerted, until it earns promotion."""
    if len(df) < 120 or rs is None or rs < 90:
        return None
    c, h, l = df["Close"], df["High"], df["Low"]
    px = float(c.iloc[-1])
    hi20 = float(h.tail(20).max())
    off = (hi20 - px) / hi20
    if not (0.03 <= off <= 0.08):
        return None
    if px <= float(h.iloc[-2]):   # must reclaim prior day high
        return None
    atr = float((h - l).tail(14).mean())
    stop = round(float(l.tail(2).min()) - 0.25 * atr, 2)
    risk = px - stop
    if risk <= 0 or risk / px > 0.05:
        return None
    return Signal("RS_Leader_Pullback", symbol, "LONG", round(px, 2), stop,
                  round(px + 2.0 * risk, 2), round(px + 3.0 * risk, 2),
                  rr=2.0, rs_rank=rs)


STRATEGIES = {
    "EMA21_Bounce":        {"fn": scan_ema21_bounce,       "default_gate": "LIVE"},
    "RS_Leader_Pullback":  {"fn": scan_rs_leader_pullback, "default_gate": "INCUBATING"},
}


def run_all(data: Dict[str, pd.DataFrame], sectors: Dict[str, str] = None) -> List[Signal]:
    rets = compute_universe_returns(data)
    signals: List[Signal] = []
    for sym, df in data.items():
        rs = _rs_rank(rets, sym)
        for name, spec in STRATEGIES.items():
            try:
                sig = spec["fn"](df, sym, rs)
                if sig:
                    sig.sector = (sectors or {}).get(sym)
                    signals.append(sig)
            except Exception:
                continue
    return signals
