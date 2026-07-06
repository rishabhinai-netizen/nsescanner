"""EOD scan job — GitHub Actions entrypoint.
Pipeline: universe → data → regime → strategies → gate → Supabase → Telegram.
Writes ONLY to Supabase. Never commits artifacts to git (v15 antipattern removed).

Usage: python -m jobs.scan --universe nifty200
"""
import argparse
import logging
import sys

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
log = logging.getLogger("nx.scan")

from datetime import date

from core.data import fetch_batch_daily, fetch_nifty
from core.regime import detect_regime, compute_breadth
from core.strategies import run_all, STRATEGIES
from core.gate import gate as run_gate
from core.db import (save_signals, update_outcomes, rollup_strategy_stats,
                     load_live_stats)
from core.alerts import send_telegram, fmt_signal, fmt_summary
from jobs.universe import get_universe, get_sectors


def main(universe_name: str = "nifty200"):
    symbols = get_universe(universe_name)
    sectors = get_sectors()
    log.info(f"universe={universe_name} n={len(symbols)}")

    data = fetch_batch_daily(symbols)
    log.info(f"fetched {len(data)} symbols")
    if len(data) < 30:
        log.error("insufficient data — aborting (no signals better than bad signals)")
        return 1

    nifty = fetch_nifty()
    if nifty is None:
        log.error("nifty fetch failed — aborting")
        return 1

    breadth = compute_breadth(data)
    regime = detect_regime(nifty, breadth)
    log.info(f"regime={regime['regime']} score={regime['score']} breadth={breadth}%")

    # 1) close out yesterday's open signals first (clean stats before new entries)
    n_updated = update_outcomes(data)
    rollup_strategy_stats()
    live_stats = load_live_stats()
    log.info(f"outcomes updated: {n_updated}")

    # 2) scan + gate
    signals = run_all(data, sectors)
    rows, n_live = [], 0
    for sig in signals:
        g = run_gate(sig, regime["regime"],
                     STRATEGIES[sig.strategy]["default_gate"], live_stats)
        if g["gate"] == "BLOCKED":
            continue  # logged in summary count only — blocked signals aren't stored
        rows.append({
            "signal_date": date.today().isoformat(), "strategy": sig.strategy,
            "symbol": sig.symbol, "side": sig.side, "entry": sig.entry,
            "stop": sig.stop, "target1": sig.target1, "target2": sig.target2,
            "rr": sig.rr, "rs_rank": sig.rs_rank, "sector": sig.sector,
            "regime": regime["regime"], "regime_score": regime["score"],
            "sqi": g["sqi"], "sqi_tier": g["tier"], "gate": g["gate"],
            "meta": {"reason": g["reason"], **sig.meta},
        })
        if g["gate"] == "LIVE":
            n_live += 1

    saved = save_signals(rows)
    n_inc = len(rows) - n_live
    n_blocked = len(signals) - len(rows)
    log.info(f"saved={saved} live={n_live} incubating={n_inc} blocked={n_blocked}")

    # 3) alert — LIVE only, Tier A/B only, capped at 8 to protect signal trust
    live_rows = sorted([r for r in rows if r["gate"] == "LIVE"],
                       key=lambda r: -r["sqi"])[:8]
    for r in live_rows:
        send_telegram(fmt_signal(r, regime["regime"]))
    send_telegram(fmt_summary(n_live, n_inc, n_blocked, regime))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="nifty200",
                    choices=["nifty50", "nifty200", "nifty500"])
    args = ap.parse_args()
    sys.exit(main(args.universe))
