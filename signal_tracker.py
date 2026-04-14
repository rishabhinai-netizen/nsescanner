"""
Signal Tracker v2 — Supabase primary, CSV fallback
===================================================
All signal writes go to Supabase first.
CSV files remain as a local cache / offline fallback.
Works in both GitHub Actions and Streamlit Cloud environments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List
import os, logging, json

logger = logging.getLogger(__name__)

SIGNALS_DIR = Path(__file__).parent / "signals"

# ============================================================
# SUPABASE CLIENT — lazy init, graceful degradation
# ============================================================

_supabase_client = None

def _get_supabase():
    """Return a Supabase client if credentials are available, else None."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    try:
        url = _get_secret("SUPABASE_URL")
        key = _get_secret("SUPABASE_SERVICE_KEY") or _get_secret("SUPABASE_ANON_KEY")
        if not url or not key:
            return None
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as e:
        logger.debug(f"Supabase init failed: {e}")
        return None


def _get_secret(key: str) -> str:
    """Read from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, "")


# ============================================================
# WRITE — save signals to Supabase + local CSV
# ============================================================

def save_signals_today(signals_list: list, regime: dict = None) -> int:
    """Save scan signals. Writes to Supabase (primary) and CSV (fallback cache)."""
    SIGNALS_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    now_str = datetime.now().strftime("%H:%M")
    regime_name = regime.get("regime", "") if regime else ""
    regime_score = regime.get("score", 0) if regime else 0

    new_rows = []
    for r in signals_list:
        row = {
            "date":          today,
            "time":          now_str,
            "strategy":      getattr(r, "strategy", ""),
            "strategy_name": getattr(r, "strategy", ""),
            "symbol":        r.symbol,
            "signal":        r.signal,
            "cmp":           round(r.cmp, 2),
            "entry":         round(r.entry, 2),
            "sl":            round(r.stop_loss, 2),
            "t1":            round(r.target_1, 2),
            "t2":            round(r.target_2, 2),
            "rr":            round(r.risk_reward, 1),
            "confidence":    int(r.confidence),
            "rs":            round(r.rs_rating, 1),
            "sqi":           round(getattr(r, "sqi", 0), 1) or None,
            "sqi_grade":     getattr(r, "sqi_grade", "") or None,
            "sqi_breakdown": getattr(r, "sqi_breakdown", "") or None,
            "sector":        getattr(r, "sector", ""),
            "regime":        regime_name,
            "regime_score":  int(regime_score),
            "regime_fit":    getattr(r, "regime_fit", ""),
            "status":        "OPEN",
        }
        new_rows.append(row)

    if not new_rows:
        return 0

    saved = 0

    # --- Supabase write (smart upsert: preserves first_seen_ist, increments scan_count) ---
    sb = _get_supabase()
    if sb:
        try:
            import pytz
            IST = pytz.timezone('Asia/Kolkata')
            ist_now = datetime.now(IST).isoformat()

            # Fetch ALL currently OPEN signals (any date) to prevent cross-day duplicates
            # A signal is only "new" if no OPEN row exists for same symbol+strategy
            existing_res = sb.table("signals").select("id,symbol,strategy,scan_count,date").eq("status", "OPEN").execute()
            existing_open = {(r["symbol"], r["strategy"]): r for r in (existing_res.data or [])}
            
            # Also check today's signals (even closed ones) for same-day re-detection
            today_res = sb.table("signals").select("id,symbol,strategy,scan_count").eq("date", today).execute()
            today_map = {(r["symbol"], r["strategy"]): r for r in (today_res.data or [])}

            new_inserts, updates = [], []
            for row in new_rows:
                key = (row["symbol"], row["strategy"])
                if key in today_map:
                    # Already recorded today — just update CMP + scan_count
                    updates.append({
                        "id":           today_map[key]["id"],
                        "cmp":          row["cmp"],
                        "scan_count":   (today_map[key].get("scan_count") or 1) + 1,
                        "last_seen_ist": ist_now,
                    })
                elif key in existing_open:
                    # Already OPEN from a prior day — update CMP + last_seen only
                    # Do NOT create a new row — this is the cross-day dedup fix
                    updates.append({
                        "id":           existing_open[key]["id"],
                        "cmp":          row["cmp"],
                        "scan_count":   (existing_open[key].get("scan_count") or 1) + 1,
                        "last_seen_ist": ist_now,
                    })
                else:
                    # Genuinely new signal — no open row exists for this symbol+strategy
                    row["first_seen_ist"] = ist_now
                    row["last_seen_ist"]  = ist_now
                    row["scan_count"]     = 1
                    new_inserts.append(row)

            if new_inserts:
                sb.table("signals").insert(new_inserts).execute()
                saved += len(new_inserts)
            for upd in updates:
                _id = upd.pop("id")
                sb.table("signals").update(upd).eq("id", _id).execute()

            logger.info(f"Supabase: {len(new_inserts)} new + {len(updates)} re-detected signals for {today}")
        except Exception as e:
            logger.warning(f"Supabase write failed: {e} — falling back to CSV")

    # --- CSV write (always, as local cache) ---
    today_file = SIGNALS_DIR / f"{today}_signals.csv"
    existing_keys = set()
    df_old = pd.DataFrame()
    if today_file.exists():
        try:
            df_old = pd.read_csv(today_file)
            for _, row in df_old.iterrows():
                existing_keys.add(f"{row.get('strategy','')}_{row.get('symbol','')}")
        except Exception:
            pass

    csv_rows = []
    for row in new_rows:
        key = f"{row['strategy']}_{row['symbol']}"
        if key not in existing_keys:
            csv_rows.append({
                "Date":         row["date"],
                "Time":         row["time"],
                "Strategy":     row["strategy"],
                "Strategy_Name":row["strategy_name"],
                "Symbol":       row["symbol"],
                "Signal":       row["signal"],
                "CMP":          row["cmp"],
                "Entry":        row["entry"],
                "SL":           row["sl"],
                "T1":           row["t1"],
                "T2":           row["t2"],
                "RR":           row["rr"],
                "Confidence":   row["confidence"],
                "RS":           row["rs"],
                "SQI":          row["sqi"],
                "SQI_Grade":    row["sqi_grade"],
                "Sector":       row["sector"],
                "Regime":       row["regime"],
                "Regime_Score": row["regime_score"],
                "Regime_Fit":   row["regime_fit"],
                "Status":       "OPEN",
                "Exit_Date":    "",
                "Exit_Price":   "",
                "Exit_Reason":  "",
                "PnL_Pct":      "",
            })

    if csv_rows:
        df_new = pd.DataFrame(csv_rows)
        combined = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new
        combined.to_csv(today_file, index=False)
        if saved == 0:
            saved = len(csv_rows)

    return saved


# ============================================================
# READ — signals
# ============================================================

def load_signals(date_str: str = None) -> Optional[pd.DataFrame]:
    """Load signals for a date (or all). Tries Supabase first, CSV fallback."""
    sb = _get_supabase()

    if sb:
        try:
            q = sb.table("signals").select("*").order("created_at", desc=False)
            if date_str:
                q = q.eq("date", date_str)
            resp = q.execute()
            if resp.data:
                df = pd.DataFrame(resp.data)
                # Normalize column names to match legacy code expectations
                rename = {
                    "sl": "SL", "t1": "T1", "t2": "T2", "rr": "RR",
                    "date": "Date", "time": "Time", "strategy": "Strategy",
                    "strategy_name": "Strategy_Name", "symbol": "Symbol",
                    "signal": "Signal", "cmp": "CMP", "entry": "Entry",
                    "confidence": "Confidence", "rs": "RS", "sqi": "SQI",
                    "sqi_grade": "SQI_Grade", "sector": "Sector",
                    "regime": "Regime", "regime_score": "Regime_Score",
                    "regime_fit": "Regime_Fit", "status": "Status",
                    "exit_date": "Exit_Date", "exit_price": "Exit_Price",
                    "exit_reason": "Exit_Reason", "pnl_pct": "PnL_Pct",
                }
                df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
                return df
        except Exception as e:
            logger.warning(f"Supabase read failed: {e}")

    # CSV fallback
    SIGNALS_DIR.mkdir(exist_ok=True)
    if date_str:
        filepath = SIGNALS_DIR / f"{date_str}_signals.csv"
        if filepath.exists():
            return pd.read_csv(filepath)
        return None

    files = sorted(SIGNALS_DIR.glob("*_signals.csv"))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None


def get_signal_dates() -> List[str]:
    """Get all distinct dates that have signals."""
    sb = _get_supabase()
    if sb:
        try:
            resp = sb.table("signals").select("date").execute()
            if resp.data:
                dates = sorted(set(r["date"] for r in resp.data), reverse=True)
                return dates
        except Exception:
            pass

    SIGNALS_DIR.mkdir(exist_ok=True)
    files = sorted(SIGNALS_DIR.glob("*_signals.csv"), reverse=True)
    return [f.stem.replace("_signals", "") for f in files]


# ============================================================
# TRACKER — outcome tracking (SL/T1 hit)
# ============================================================

def load_tracker() -> Optional[pd.DataFrame]:
    """Load all signals with status info (the tracker view)."""
    return load_signals()  # Same table — status column tracks outcomes


def update_open_signals_live(data_dict: dict) -> int:
    """
    Check OPEN signals against current prices in data_dict.
    Updates Supabase + CSV. Returns count of updated signals.
    """
    today = date.today().isoformat()
    all_signals = load_signals()
    if all_signals is None or all_signals.empty:
        return 0

    open_mask = all_signals.get("Status", all_signals.get("status", pd.Series())) == "OPEN"
    open_signals = all_signals[open_mask]
    if open_signals.empty:
        return 0

    updated = 0
    updates_for_sb = []

    for _, row in open_signals.iterrows():
        sym = row.get("Symbol", row.get("symbol", ""))
        if sym not in data_dict:
            continue

        stock_df = data_dict[sym]
        if stock_df is None or stock_df.empty:
            continue

        # Use price history SINCE the signal date (not just last 5 bars)
        sig_date_str = str(row.get("Date", row.get("date", today)))
        try:
            sig_date = pd.Timestamp(sig_date_str)
            since_signal = stock_df[stock_df.index >= sig_date] if not stock_df.empty else stock_df
            if since_signal.empty:
                since_signal = stock_df.iloc[-5:]   # fallback
        except Exception:
            since_signal = stock_df.iloc[-5:]

        recent_high = float(since_signal["high"].max()) if "high" in since_signal.columns and not since_signal.empty else 0
        recent_low  = float(since_signal["low"].min())  if "low"  in since_signal.columns and not since_signal.empty else 0
        close       = float(stock_df.iloc[-1].get("close", 0))

        try:
            entry  = float(row.get("Entry", row.get("entry", 0)) or 0)
            sl     = float(row.get("SL",    row.get("sl",    0)) or 0)
            t1     = float(row.get("T1",    row.get("t1",    0)) or 0)
            signal = row.get("Signal", row.get("signal", "BUY"))
        except (ValueError, TypeError):
            continue

        if entry <= 0:
            continue

        # Days held — for expiry logic
        try:
            days_held = (date.today() - pd.Timestamp(sig_date_str).date()).days
        except Exception:
            days_held = 0

        # Strategy-specific max hold (EMA bounce = 15d, breakout/short = 25d)
        strategy_name = str(row.get("Strategy", row.get("strategy", "")))
        max_hold = 15 if "EMA21" in strategy_name or "Last30" in strategy_name else 25

        update = None
        if signal == "BUY":
            # Check T1 FIRST — if both SL and T1 were touched, T1 wins (assume best execution)
            if recent_high >= t1 > 0:
                update = dict(status="TARGET", exit_date=today,
                              exit_price=round(t1, 2),
                              exit_reason="Target 1 Hit",
                              pnl_pct=round((t1 / entry - 1) * 100, 2))
            elif recent_low <= sl > 0:
                update = dict(status="STOPPED", exit_date=today,
                              exit_price=round(sl, 2),
                              exit_reason="Stop Loss Hit",
                              pnl_pct=round((sl / entry - 1) * 100, 2))
            elif days_held > max_hold:
                update = dict(status="EXPIRED", exit_date=today,
                              exit_price=round(close, 2),
                              exit_reason=f"Expired ({days_held}d > {max_hold}d max)",
                              pnl_pct=round((close / entry - 1) * 100, 2))
        elif signal == "SHORT":
            # T1 check first (if both triggered, T1 wins)
            if recent_low <= t1 > 0:
                update = dict(status="TARGET", exit_date=today,
                              exit_price=round(t1, 2),
                              exit_reason="Target 1 Hit",
                              pnl_pct=round((entry - t1) / entry * 100, 2))   # positive = profit
            elif recent_high >= sl > 0:
                update = dict(status="STOPPED", exit_date=today,
                              exit_price=round(sl, 2),
                              exit_reason="Stop Loss Hit",
                              pnl_pct=round((entry - sl) / entry * 100, 2))   # negative = loss
            elif days_held > max_hold:
                update = dict(status="EXPIRED", exit_date=today,
                              exit_price=round(close, 2),
                              exit_reason=f"Expired ({days_held}d > {max_hold}d max)",
                              pnl_pct=round((entry - close) / entry * 100, 2))

        if update:
            sig_date = str(row.get("Date", row.get("date", today)))
            strategy = row.get("Strategy", row.get("strategy", ""))
            updates_for_sb.append({"date": sig_date, "strategy": strategy,
                                   "symbol": sym, **update})
            updated += 1

    # Write updates to Supabase
    sb = _get_supabase()
    if sb and updates_for_sb:
        for upd in updates_for_sb:
            try:
                sb.table("signals").update({
                    "status":       upd["status"],
                    "exit_date":    upd["exit_date"],
                    "exit_price":   upd["exit_price"],
                    "exit_reason":  upd["exit_reason"],
                    "pnl_pct":      upd["pnl_pct"],
                }).eq("date", upd["date"]).eq("strategy", upd["strategy"]).eq("symbol", upd["symbol"]).execute()
            except Exception as e:
                logger.warning(f"Supabase update failed for {upd['symbol']}: {e}")

    # Also update CSV files
    _update_csv_outcomes(updates_for_sb)

    return updated


def _update_csv_outcomes(updates: list):
    """Sync outcome updates back to CSV files."""
    if not updates:
        return
    for upd in updates:
        sig_date = upd.get("date", date.today().isoformat())
        filepath = SIGNALS_DIR / f"{sig_date}_signals.csv"
        if not filepath.exists():
            continue
        try:
            df = pd.read_csv(filepath)
            mask = (df.get("Strategy", df.get("strategy", pd.Series())) == upd.get("strategy", "")) & \
                   (df.get("Symbol",   df.get("symbol",   pd.Series())) == upd.get("symbol", ""))
            if mask.any():
                df.loc[mask, "Status"]       = upd["status"]
                df.loc[mask, "Exit_Date"]    = upd["exit_date"]
                df.loc[mask, "Exit_Price"]   = upd["exit_price"]
                df.loc[mask, "Exit_Reason"]  = upd["exit_reason"]
                df.loc[mask, "PnL_Pct"]      = upd["pnl_pct"]
                df.to_csv(filepath, index=False)
        except Exception:
            pass


# ============================================================
# STATS
# ============================================================

def compute_tracker_stats(df: pd.DataFrame = None) -> Dict:
    """Compute forward-test statistics.
    
    Deduplication policy: for same symbol+strategy, keep the LATEST row only.
    This prevents cross-day duplicate signals from inflating counts.
    """
    if df is None:
        df = load_tracker()
    if df is None or df.empty:
        return {}

    strat_col  = "Strategy" if "Strategy" in df.columns else "strategy"
    sym_col    = "Symbol"   if "Symbol"   in df.columns else "symbol"
    date_col   = "Date"     if "Date"     in df.columns else "date"
    status_col = "Status"   if "Status"   in df.columns else "status"

    # Deduplicate: per (symbol, strategy) keep latest non-OPEN outcome if any,
    # else keep the most recent OPEN row. This ensures each trade idea counted once.
    try:
        df = df.copy()
        # Sort by date descending so .first() picks most recent
        df["_date_sort"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values("_date_sort", ascending=False)
        
        # Within each symbol+strategy, prefer closed outcomes (TARGET/STOPPED/EXPIRED)
        # over OPEN — so if same stock was signalled 3 times and hit target once, 
        # we count it as 1 TARGET.
        def _pick_best_row(group):
            closed = group[group[status_col].isin(["TARGET","STOPPED","EXPIRED"])]
            if not closed.empty:
                return closed.iloc[0]   # most recent closed outcome
            return group.iloc[0]        # most recent OPEN
        
        df = df.groupby([sym_col, strat_col], as_index=False).apply(_pick_best_row)
        df = df.reset_index(drop=True)
    except Exception:
        pass  # If dedup fails, continue with raw data (safer than crashing)

    # Exclude DUPLICATE and REGIME_BLOCKED from all counts
    # These are noise rows — DUPLICATE = cross-day same signal,
    # REGIME_BLOCKED = signal generated during a blocked market regime
    EXCLUDED = ["DUPLICATE", "REGIME_BLOCKED"]
    df = df[~df[status_col].isin(EXCLUDED)].copy()

    targets  = len(df[df[status_col] == "TARGET"])
    stopped  = len(df[df[status_col] == "STOPPED"])
    open_c   = len(df[df[status_col] == "OPEN"])
    expired  = len(df[df[status_col] == "EXPIRED"])
    closed   = targets + stopped
    total    = targets + stopped + open_c + expired  # meaningful total only

    win_rate = round(targets / closed * 100, 1) if closed > 0 else 0

    pnl_col_name = "PnL_Pct" if "PnL_Pct" in df.columns else "pnl_pct"
    pnl = pd.to_numeric(df.get(pnl_col_name, pd.Series()), errors="coerce")
    closed_pnl = pnl[df[status_col].isin(["TARGET", "STOPPED"])].dropna()

    strat_col = "Strategy" if "Strategy" in df.columns else "strategy"
    strategy_stats = {}
    for strat in df[strat_col].unique():
        sdf  = df[df[strat_col] == strat]
        t_c  = len(sdf[sdf[status_col] == "TARGET"])
        s_c  = len(sdf[sdf[status_col] == "STOPPED"])
        tot  = t_c + s_c
        sdf_pnl = pd.to_numeric(sdf.get(pnl_col_name, pd.Series()), errors="coerce")
        wins_pnl = sdf_pnl[sdf[status_col] == "TARGET"].dropna()
        loss_pnl = sdf_pnl[sdf[status_col] == "STOPPED"].dropna()
        strategy_stats[strat] = {
            "total":    len(sdf),
            "open":     len(sdf[sdf[status_col] == "OPEN"]),
            "targets":  t_c,
            "stopped":  s_c,
            "win_rate": round(t_c / tot * 100, 1) if tot > 0 else 0,
            "avg_win":  round(float(wins_pnl.mean()), 2) if len(wins_pnl) > 0 else 0,
            "avg_loss": round(float(loss_pnl.mean()), 2) if len(loss_pnl) > 0 else 0,
        }

    dates = pd.to_datetime(df.get("Date", df.get("date", pd.Series())), errors="coerce").dropna()

    return {
        "total":          total,
        "targets":        targets,
        "stopped":        stopped,
        "open":           open_c,
        "expired":        expired,
        "closed":         closed,
        "win_rate":       win_rate,
        # total_pnl = avg P&L per closed trade (expectancy metric). SUM was wrong.
        "total_pnl":      round(float(closed_pnl.mean()), 2) if len(closed_pnl) > 0 else 0,
        "avg_win":        round(float(closed_pnl[closed_pnl > 0].mean()), 2) if (closed_pnl > 0).any() else 0,
        "avg_loss":       round(float(closed_pnl[closed_pnl < 0].mean()), 2) if (closed_pnl < 0).any() else 0,
        "expectancy":     round(
                              (win_rate/100) * (round(float(closed_pnl[closed_pnl>0].mean()),2) if (closed_pnl>0).any() else 0)
                            + (1-win_rate/100) * (round(float(closed_pnl[closed_pnl<0].mean()),2) if (closed_pnl<0).any() else 0)
                          , 2) if closed > 0 else 0,
        "strategy_stats": strategy_stats,
        "first_date":     dates.min().strftime("%Y-%m-%d") if len(dates) > 0 else "",
        "last_date":      dates.max().strftime("%Y-%m-%d") if len(dates) > 0 else "",
        "trading_days":   len(df.get("Date", df.get("date", pd.Series())).unique()),
    }


def generate_csv_download() -> Optional[bytes]:
    df = load_tracker()
    if df is None or df.empty:
        return None
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# PORTFOLIO helpers — open position tracking
# ============================================================

def save_position(position: dict) -> bool:
    """Save an open position to Supabase portfolio table."""
    sb = _get_supabase()
    if sb:
        try:
            sb.table("portfolio").insert(position).execute()
            return True
        except Exception as e:
            logger.warning(f"Portfolio save failed: {e}")
    return False


def load_portfolio() -> Optional[pd.DataFrame]:
    """Load open portfolio positions from Supabase."""
    sb = _get_supabase()
    if sb:
        try:
            resp = sb.table("portfolio").select("*").eq("status", "OPEN").order("entry_date", desc=True).execute()
            if resp.data:
                return pd.DataFrame(resp.data)
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Portfolio load failed: {e}")
    return pd.DataFrame()


def close_position(position_id: str, exit_price: float, exit_reason: str = "Manual") -> bool:
    """Close a portfolio position."""
    sb = _get_supabase()
    if sb:
        try:
            resp = sb.table("portfolio").select("*").eq("id", position_id).execute()
            if resp.data:
                pos = resp.data[0]
                entry = float(pos.get("entry_price", 0))
                qty   = int(pos.get("qty", 1))
                pnl_pct = round((exit_price / entry - 1) * 100, 2) if entry > 0 else 0
                if pos.get("signal") == "SHORT":
                    pnl_pct = round((entry / exit_price - 1) * 100, 2) if exit_price > 0 else 0
                pnl_abs = round((exit_price - entry) * qty, 2)

                sb.table("portfolio").update({
                    "status":       "CLOSED",
                    "exit_price":   round(exit_price, 2),
                    "exit_date":    date.today().isoformat(),
                    "exit_reason":  exit_reason,
                    "pnl":          pnl_abs,
                    "pnl_pct":      pnl_pct,
                }).eq("id", position_id).execute()
                return True
        except Exception as e:
            logger.warning(f"Close position failed: {e}")
    return False


def get_portfolio_pnl(data_dict: dict) -> dict:
    """Compute live P&L for all open positions using current prices."""
    portfolio_df = load_portfolio()
    if portfolio_df is None or portfolio_df.empty:
        return {"positions": [], "total_pnl": 0, "total_pnl_pct": 0, "heat_pct": 0}

    positions = []
    total_pnl = 0.0
    total_invested = 0.0

    for _, pos in portfolio_df.iterrows():
        sym        = pos.get("symbol", "")
        entry      = float(pos.get("entry_price", 0) or 0)
        qty        = int(pos.get("qty", 1) or 1)
        sl         = float(pos.get("stop_loss", 0) or 0)
        signal     = pos.get("signal", "BUY")

        cmp = entry  # fallback
        if sym in data_dict and not data_dict[sym].empty:
            cmp = float(data_dict[sym].iloc[-1].get("close", entry))

        if signal == "BUY":
            pnl_pct = round((cmp / entry - 1) * 100, 2) if entry > 0 else 0
        else:
            pnl_pct = round((entry / cmp - 1) * 100, 2) if cmp > 0 else 0

        pnl_abs = round((cmp - entry) * qty, 2) if signal == "BUY" else round((entry - cmp) * qty, 2)
        total_pnl += pnl_abs

        position_value = entry * qty
        total_invested += position_value

        risk_per_share = abs(entry - sl) if sl > 0 else entry * 0.05
        risk_amount = risk_per_share * qty

        positions.append({
            "id":            str(pos.get("id", "")),
            "symbol":        sym,
            "strategy":      pos.get("strategy", ""),
            "signal":        signal,
            "entry":         entry,
            "cmp":           cmp,
            "sl":            sl,
            "target1":       float(pos.get("target1", 0) or 0),
            "qty":           qty,
            "pnl_pct":       pnl_pct,
            "pnl_abs":       pnl_abs,
            "position_value":position_value,
            "risk_amount":   round(risk_amount, 2),
            "entry_date":    str(pos.get("entry_date", "")),
            "sector":        pos.get("sector", ""),
        })

    return {
        "positions":      positions,
        "total_pnl":      round(total_pnl, 2),
        "total_pnl_pct":  round((total_pnl / total_invested * 100), 2) if total_invested > 0 else 0,
        "total_invested": round(total_invested, 2),
        "position_count": len(positions),
    }


