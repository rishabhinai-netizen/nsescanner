"""Regime engine — Nifty trend + universe breadth → one of four regimes.
The regime is the gate's primary input; forward-test proved regime context
matters more than raw setup quality."""
from typing import Dict
import pandas as pd


def detect_regime(nifty: pd.DataFrame, breadth_pct_above_50ema: float = None) -> Dict:
    """EXPANSION / ACCUMULATION / DISTRIBUTION / PANIC.
    Score axes: trend (Nifty vs 20/50/200 EMA), momentum (20d ROC), breadth."""
    c = nifty["Close"]
    e20, e50, e200 = (c.ewm(span=n).mean().iloc[-1] for n in (20, 50, 200))
    px = float(c.iloc[-1])
    roc20 = (px / float(c.iloc[-21]) - 1) * 100 if len(c) > 21 else 0.0
    atr_pct = float((nifty["High"] - nifty["Low"]).tail(14).mean() / px * 100)

    score = 0.0
    score += 2 if px > e20 else -2
    score += 2 if px > e50 else -2
    score += 1 if px > e200 else -1
    score += max(-3, min(3, roc20 / 2))
    if breadth_pct_above_50ema is not None:
        score += (breadth_pct_above_50ema - 50) / 12.5  # ±4 range

    vol_spike = atr_pct > 1.8

    if score >= 4 and not vol_spike:
        regime = "EXPANSION"
    elif score >= 0:
        regime = "ACCUMULATION"
    elif score >= -4 and not vol_spike:
        regime = "DISTRIBUTION"
    else:
        regime = "PANIC"

    return {"regime": regime, "score": round(score, 1), "roc20": round(roc20, 2),
            "atr_pct": round(atr_pct, 2), "px": px,
            "above_e20": px > e20, "above_e50": px > e50, "above_e200": px > e200}


def compute_breadth(data: Dict[str, pd.DataFrame]) -> float:
    """% of universe above its own 50 EMA."""
    above = total = 0
    for df in data.values():
        if len(df) < 60:
            continue
        total += 1
        if float(df["Close"].iloc[-1]) > float(df["Close"].ewm(span=50).mean().iloc[-1]):
            above += 1
    return round(above / total * 100, 1) if total else 50.0
