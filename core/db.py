"""DB layer — Supabase is the single source of truth. No CSVs in git.
Jobs use the service key (bypasses RLS); the app uses anon key + auth."""
import logging
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from supabase import create_client

from core.config import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY

logger = logging.getLogger("nx.db")


def client(service: bool = False):
    key = SUPABASE_SERVICE_KEY() if service else SUPABASE_ANON_KEY()
    url = SUPABASE_URL()
    if not url or not key:
        raise RuntimeError("Supabase credentials missing — set SUPABASE_URL and keys")
    return create_client(url, key)


# ------------------------------------------------------------------ signals --
def save_signals(rows: List[Dict], service: bool = True) -> int:
    if not rows:
        return 0
    sb = client(service)
    res = sb.table("nx_signals").upsert(
        rows, on_conflict="signal_date,strategy,symbol").execute()
    return len(res.data or [])


def load_signals(days: int = 30, service: bool = False) -> pd.DataFrame:
    sb = client(service)
    res = (sb.table("nx_signals").select("*")
             .gte("signal_date", (pd.Timestamp.today() - pd.Timedelta(days=days)).date().isoformat())
             .order("signal_date", desc=True).limit(2000).execute())
    return pd.DataFrame(res.data or [])


def open_signals(service: bool = True) -> pd.DataFrame:
    sb = client(service)
    res = sb.table("nx_signals").select("*").eq("status", "OPEN").execute()
    return pd.DataFrame(res.data or [])


def update_outcomes(data: Dict[str, pd.DataFrame], max_hold_days: int = 10) -> int:
    """Mark OPEN signals TARGET/STOPPED/EXPIRED from daily bars. Day-0 excluded
    (signal generated at EOD; fills start next session)."""
    sb = client(True)
    df = open_signals()
    if df.empty:
        return 0
    updated = 0
    for _, r in df.iterrows():
        bars = data.get(r["symbol"])
        if bars is None:
            continue
        after = bars[bars.index.date > pd.Timestamp(r["signal_date"]).date()]
        if after.empty:
            continue
        patch = None
        for ts, bar in after.iterrows():
            if r["side"] == "LONG":
                if float(bar["Low"]) <= float(r["stop"]):
                    patch = {"status": "STOPPED", "exit_price": float(r["stop"]),
                             "exit_reason": "Stop hit"}
                elif float(bar["High"]) >= float(r["target1"]):
                    patch = {"status": "TARGET", "exit_price": float(r["target1"]),
                             "exit_reason": "T1 hit"}
            if patch:
                patch["exit_date"] = ts.date().isoformat()
                break
        if patch is None and len(after) >= max_hold_days:
            last = after.iloc[max_hold_days - 1]
            patch = {"status": "EXPIRED", "exit_price": float(last["Close"]),
                     "exit_reason": f"Time stop {max_hold_days}d",
                     "exit_date": after.index[max_hold_days - 1].date().isoformat()}
        if patch:
            sign = 1 if r["side"] == "LONG" else -1
            patch["pnl_pct"] = round(sign * (patch["exit_price"] / float(r["entry"]) - 1) * 100, 2)
            sb.table("nx_signals").update(patch).eq("id", r["id"]).execute()
            updated += 1
    return updated


# -------------------------------------------------------------------- stats --
def rollup_strategy_stats() -> int:
    """Recompute live PF per strategy×regime → nx_strategy_stats (feeds the gate)."""
    sb = client(True)
    res = (sb.table("nx_signals").select("strategy,regime,status,pnl_pct")
             .in_("status", ["TARGET", "STOPPED", "EXPIRED"]).execute())
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return 0
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    df = df.dropna(subset=["pnl_pct"])
    rows = []
    for (strat, reg), g in df.groupby(["strategy", "regime"]):
        gains = g.loc[g.pnl_pct > 0, "pnl_pct"].sum()
        losses = abs(g.loc[g.pnl_pct < 0, "pnl_pct"].sum())
        pf = round(float(gains / losses), 2) if losses > 0 else 99.0
        rows.append({"strategy": strat, "regime": reg, "n_closed": int(len(g)),
                     "n_wins": int((g.pnl_pct > 0).sum()),
                     "win_rate": round(float((g.pnl_pct > 0).mean() * 100), 1),
                     "avg_pnl": round(float(g.pnl_pct.mean()), 2),
                     "profit_factor": pf})
    if rows:
        sb.table("nx_strategy_stats").upsert(rows, on_conflict="strategy,regime").execute()
    return len(rows)


def load_live_stats() -> Dict[str, Dict]:
    try:
        res = client(True).table("nx_strategy_stats").select("*").execute()
        return {f"{r['strategy']}|{r['regime']}": r for r in (res.data or [])}
    except Exception as e:
        logger.warning(f"live stats unavailable: {e}")
        return {}


# ------------------------------------------------------------------- config --
def get_config(key: str) -> Optional[str]:
    try:
        res = client(True).table("nx_app_config").select("value").eq("key", key).execute()
        return res.data[0]["value"] if res.data else None
    except Exception:
        return None


def set_config(key: str, value: str) -> bool:
    try:
        client(True).table("nx_app_config").upsert(
            {"key": key, "value": value}, on_conflict="key").execute()
        return True
    except Exception as e:
        logger.warning(f"set_config failed: {e}")
        return False
