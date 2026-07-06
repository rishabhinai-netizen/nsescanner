"""One-time migration — v15 tracker.csv → nx_signals.
Preserves your 2,226-signal forward-test history so the gate starts with
live stats on day one instead of a cold start.

Usage:  SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python -m migrate.migrate path/to/tracker.csv
"""
import sys
import pandas as pd

sys.path.insert(0, ".")
from core.db import save_signals, rollup_strategy_stats

STATUS_MAP = {"TARGET": "TARGET", "STOPPED": "STOPPED", "EXPIRED": "EXPIRED",
              "OPEN": "OPEN", "INVALID_DAY0": "INVALID"}


def main(path: str):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        status = STATUS_MAP.get(str(r.get("Status", "")).strip(), "INVALID")
        rows.append({
            "signal_date": str(r["Date"]),
            "strategy": str(r["Strategy"]),
            "symbol": str(r["Symbol"]),
            "side": "SHORT" if str(r.get("Signal", "")).upper() == "SHORT" else "LONG",
            "entry": float(r["Entry"]), "stop": float(r["SL"]),
            "target1": float(r["T1"]),
            "target2": float(r["T2"]) if pd.notna(r.get("T2")) else None,
            "rr": float(r["RR"]) if pd.notna(r.get("RR")) else None,
            "rs_rank": float(r["RS"]) if pd.notna(r.get("RS")) else None,
            "sector": r.get("Sector") if pd.notna(r.get("Sector")) else None,
            "regime": r.get("Regime") if pd.notna(r.get("Regime")) else None,
            "regime_score": float(r["Regime_Score"]) if pd.notna(r.get("Regime_Score")) else None,
            "sqi": float(r["SQI"]) if pd.notna(r.get("SQI")) else None,
            "sqi_tier": None,
            "gate": "LIVE",                      # historical: everything was live in v15
            "status": status,
            "exit_date": str(r["Exit_Date"]) if pd.notna(r.get("Exit_Date")) else None,
            "exit_price": float(r["Exit_Price"]) if pd.notna(r.get("Exit_Price")) else None,
            "exit_reason": r.get("Exit_Reason") if pd.notna(r.get("Exit_Reason")) else None,
            "pnl_pct": float(r["PnL_Pct"]) if pd.notna(r.get("PnL_Pct")) else None,
            "meta": {"migrated_from": "v15_tracker"},
        })
    n = save_signals(rows)
    print(f"migrated {n} signals")
    print(f"stats cells rolled up: {rollup_strategy_stats()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tracker.csv")
