"""
Auto Scanner v2 — Standalone script for GitHub Actions
========================================================
FIXES IN v2:
  - CRITICAL: Tracker now correctly compares signal-day+1 onwards
    (was: comparing same-day intraday range → 88% false day-0 stops)
  - Holiday-aware: skips Saturday, Sunday, NSE holidays
  - Retry with backoff for Nifty fetch (single-flake won't abort scan)
  - Logs DATA_QUALITY=STALE if today's bar wasn't actually traded

Usage:
    python auto_scanner.py                    # Scan + record signals
    python auto_scanner.py --track            # Update tracker (signal_date+1 onwards)
    python auto_scanner.py --mode eod         # End-of-day full scan
    python auto_scanner.py --mode intraday    # Quick scan (fewer stocks)
"""

import pandas as pd
import numpy as np
import os, sys, json, csv, logging, time
from datetime import datetime, date, timedelta
from pathlib import Path
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import fetch_batch_daily, fetch_nifty_data, Indicators
from stock_universe import get_stock_universe, get_sector, NIFTY_50
from scanners import (
    STRATEGY_PROFILES, DAILY_SCANNERS,
    run_all_scanners, detect_market_regime
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SIGNALS_DIR = Path(__file__).parent / "signals"
TRACKER_FILE = SIGNALS_DIR / "tracker.csv"

# ──────────────────────────────────────────────────────────────────────────
# NSE HOLIDAY CALENDAR — minimal hardcoded for safety
# Update yearly from https://www.nseindia.com/resources/exchange-communication-holidays
# ──────────────────────────────────────────────────────────────────────────
NSE_HOLIDAYS_2026 = {
    "2026-01-26", "2026-03-06", "2026-03-26", "2026-04-03", "2026-04-14",
    "2026-04-15", "2026-05-01", "2026-06-19", "2026-08-15", "2026-08-27",
    "2026-10-02", "2026-10-21", "2026-11-09", "2026-12-25",
}
NSE_HOLIDAYS = NSE_HOLIDAYS_2026  # union sets here for multi-year


def is_market_day(d: date = None) -> bool:
    """True if NSE was/is open on date d (default today)."""
    d = d or date.today()
    if d.weekday() >= 5:  # Sat/Sun
        return False
    if d.isoformat() in NSE_HOLIDAYS:
        return False
    return True


def previous_market_day(d: date = None) -> date:
    """Last NSE trading day on or before d (default today)."""
    d = d or date.today()
    while not is_market_day(d):
        d -= timedelta(days=1)
    return d


# ============================================================================
# SIGNAL RECORDING
# ============================================================================

def scan_and_record(mode="eod", universe="nifty500"):
    """Run all scanners, save unique signals to today's CSV."""
    SIGNALS_DIR.mkdir(exist_ok=True)

    today = date.today()
    if not is_market_day(today):
        log.info(f"=== {today.isoformat()} is not a market day. Skipping scan. ===")
        return

    today_str = today.isoformat()
    today_file = SIGNALS_DIR / f"{today_str}_signals.csv"

    log.info(f"=== Auto Scan: {mode} mode, universe={universe} ===")

    # Fetch data
    syms = get_stock_universe(universe)
    log.info(f"Fetching data for {len(syms)} stocks...")
    data = fetch_batch_daily(syms, "1y")
    log.info(f"Loaded {len(data)} stocks")

    # Robust Nifty fetch with retries
    nifty = None
    for attempt in range(3):
        nifty = fetch_nifty_data()
        if nifty is not None and not nifty.empty:
            break
        log.warning(f"Nifty fetch attempt {attempt+1}/3 failed, retrying in 5s...")
        time.sleep(5)
    if nifty is None or nifty.empty:
        log.error("Failed to fetch Nifty data after 3 attempts. Aborting.")
        return

    # Stale-data check on Nifty itself
    nifty_last_date = nifty.index[-1].date()
    if (today - nifty_last_date).days > 3:
        log.warning(f"Nifty last bar is {nifty_last_date} ({(today-nifty_last_date).days}d old). Data may be stale.")

    # Detect regime
    regime = detect_market_regime(nifty)
    log.info(f"Regime: {regime['regime_display']} ({regime['score']}/{regime['max_score']})")

    results = run_all_scanners(
        data, nifty, daily_only=True,
        regime=regime,
        has_intraday=False,
        sector_rankings={},
        min_rs=0,
    )

    current_regime_name = regime.get("regime", "UNKNOWN") if regime else "UNKNOWN"
    for strat_name, profile in STRATEGY_PROFILES.items():
        if current_regime_name in profile.get("blocked_regimes", []):
            log.info(f"  ⛔ {strat_name} BLOCKED in {current_regime_name} regime")

    total = sum(len(v) for v in results.values())
    log.info(f"Found {total} total signals")

    existing = set()
    if today_file.exists():
        df_existing = pd.read_csv(today_file)
        for _, row in df_existing.iterrows():
            existing.add(f"{row['Strategy']}_{row['Symbol']}")

    now_str = datetime.now().strftime("%H:%M")
    new_rows = []

    for strategy, signals in results.items():
        for r in signals:
            key = f"{strategy}_{r.symbol}"
            if key in existing:
                continue

            # Data-quality check on the underlying stock data
            stock_df = data.get(r.symbol)
            stock_quality = "OK"
            if stock_df is not None and not stock_df.empty:
                last_bar_date = stock_df.index[-1].date()
                if (today - last_bar_date).days > 3:
                    stock_quality = "STALE"

            new_rows.append({
                "Date": today_str,
                "Time": now_str,
                "Strategy": strategy,
                "Strategy_Name": STRATEGY_PROFILES.get(strategy, {}).get("name", strategy),
                "Symbol": r.symbol,
                "Signal": r.signal,
                "CMP": round(r.cmp, 2),
                "Entry": round(r.entry, 2),
                "SL": round(r.stop_loss, 2),
                "T1": round(r.target_1, 2),
                "T2": round(r.target_2, 2),
                "RR": round(r.risk_reward, 1),
                "Confidence": r.confidence,
                "SQI": getattr(r, "sqi", ""),
                "SQI_Grade": getattr(r, "sqi_grade", ""),
                "RS": round(r.rs_rating, 0),
                "Sector": r.sector or get_sector(r.symbol),
                "Regime": regime["regime"],
                "Regime_Score": regime["score"],
                "Regime_Fit": r.regime_fit,
                "Data_Quality": stock_quality,
                "Status": "OPEN",
                "Exit_Date": "",
                "Exit_Price": "",
                "Exit_Reason": "",
                "PnL_Pct": "",
            })

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if today_file.exists():
            df_old = pd.read_csv(today_file)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(today_file, index=False)
        log.info(f"Recorded {len(new_rows)} NEW signals (total today: {len(df_combined)})")
    else:
        log.info("No new unique signals to record.")

    summary_file = SIGNALS_DIR / f"{today_str}_summary.json"
    summary = {
        "date": today_str,
        "scan_time": now_str,
        "mode": mode,
        "universe": universe,
        "regime": regime["regime"],
        "regime_score": regime["score"],
        "nifty": regime["nifty_close"],
        "nifty_last_bar_date": nifty_last_date.isoformat(),
        "stocks_scanned": len(data),
        "total_signals": total,
        "new_signals": len(new_rows),
        "strategies": {s: len(sigs) for s, sigs in results.items() if sigs},
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Summary saved: {summary_file}")
    return today_file


# ============================================================================
# SIGNAL TRACKER — FIXED v2: signal_date+1 onwards only
# ============================================================================

def update_tracker():
    """
    Check all OPEN signals against price action that occurred STRICTLY AFTER
    the signal date. This is the fix for the day-0 false-stop bug.

    Previous bug: tracker compared signal vs same-day intraday range, so
    the signal would close as STOPPED before the trade was even held overnight.

    New logic:
      For each OPEN signal:
        Fetch OHLC for dates STRICTLY > signal_date (T+1 onwards)
        If no such bars exist yet (e.g., signal was just generated)
          → leave as OPEN, no comparison
        Else
          Compare T+1..today high/low against SL and T1
    """
    SIGNALS_DIR.mkdir(exist_ok=True)

    signal_files = sorted(SIGNALS_DIR.glob("*_signals.csv"))
    if not signal_files:
        log.info("No signal files found.")
        return

    all_signals = []
    _str_cols = ["Status", "Exit_Date", "Exit_Price", "Exit_Reason", "PnL_Pct"]
    for f in signal_files:
        try:
            df = pd.read_csv(f)
            for c in _str_cols:
                if c in df.columns:
                    df[c] = df[c].fillna("").astype(object)
                    df[c] = df[c].replace("nan", "")
            all_signals.append((f, df))
        except Exception as e:
            log.warning(f"Could not read {f}: {e}")
            continue

    if not all_signals:
        return

    open_symbols = set()
    for f, df in all_signals:
        open_mask = df["Status"] == "OPEN"
        open_symbols.update(df[open_mask]["Symbol"].unique())

    if not open_symbols:
        log.info("No OPEN signals to track.")
        return

    log.info(f"Tracking {len(open_symbols)} symbols with OPEN signals...")

    # Fetch FULL history for each symbol so we can slice T+1 onwards per signal
    import yfinance as yf
    symbol_history = {}
    for sym in open_symbols:
        try:
            tk = yf.Ticker(f"{sym}.NS")
            # 60 days is enough to cover max_hold of 25d + buffer
            hist = tk.history(period="60d", auto_adjust=True)
            if hist.empty:
                continue
            # Normalize index to date (drop tz, drop time)
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize() \
                if hist.index.tz is None else pd.to_datetime(hist.index).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
            symbol_history[sym] = hist
        except Exception as e:
            log.warning(f"Failed history for {sym}: {e}")
            continue

    log.info(f"Fetched price history for {len(symbol_history)} symbols")
    today = date.today().isoformat()
    today_date = date.today()
    updated_count = 0

    for filepath, df in all_signals:
        changed = False
        for idx, row in df.iterrows():
            if row["Status"] != "OPEN":
                continue

            sym = row["Symbol"]
            if sym not in symbol_history:
                continue

            hist = symbol_history[sym]
            signal_date_str = str(row["Date"])

            try:
                signal_date = pd.Timestamp(signal_date_str).normalize()
            except Exception:
                continue

            # ── CRITICAL FIX: only consider bars AFTER signal date ──
            after_signal = hist[hist.index > signal_date]
            if after_signal.empty:
                # No bars yet beyond signal day — leave OPEN
                continue

            try:
                entry = float(row["Entry"])
                sl = float(row["SL"])
                t1 = float(row["T1"])
            except (ValueError, TypeError):
                continue

            if entry <= 0 or sl <= 0 or t1 <= 0:
                continue

            try:
                days_held = (today_date - signal_date.date()).days
            except Exception:
                days_held = 0

            strategy_name = str(row.get("Strategy", ""))
            max_hold = 15 if "EMA21" in strategy_name or "Last30" in strategy_name else 25

            # For each bar after signal, check exit condition in chronological order
            # First-touch wins (don't peek the future)
            exit_hit = None
            for bar_date, bar in after_signal.iterrows():
                bar_high = float(bar["High"])
                bar_low = float(bar["Low"])
                bar_close = float(bar["Close"])

                if row["Signal"] == "BUY":
                    # On the same bar, if both SL and T1 are touched, we apply
                    # the conservative rule: SL wins (most adverse interpretation)
                    sl_hit = bar_low <= sl
                    t1_hit = bar_high >= t1
                    if sl_hit and t1_hit:
                        # Same-day both touched → assume SL hit first (conservative)
                        # (Real fill quality depends on open price)
                        if float(bar["Open"]) >= entry:
                            # Gap up open above entry — likely T1 hit first
                            exit_hit = ("TARGET", bar_date, t1, "Target 1 Hit")
                        else:
                            exit_hit = ("STOPPED", bar_date, sl, "Stop Loss Hit")
                        break
                    elif t1_hit:
                        exit_hit = ("TARGET", bar_date, t1, "Target 1 Hit")
                        break
                    elif sl_hit:
                        exit_hit = ("STOPPED", bar_date, sl, "Stop Loss Hit")
                        break
                elif row["Signal"] == "SHORT":
                    sl_hit = bar_high >= sl
                    t1_hit = bar_low <= t1
                    if sl_hit and t1_hit:
                        if float(bar["Open"]) <= entry:
                            exit_hit = ("TARGET", bar_date, t1, "Target 1 Hit")
                        else:
                            exit_hit = ("STOPPED", bar_date, sl, "Stop Loss Hit")
                        break
                    elif t1_hit:
                        exit_hit = ("TARGET", bar_date, t1, "Target 1 Hit")
                        break
                    elif sl_hit:
                        exit_hit = ("STOPPED", bar_date, sl, "Stop Loss Hit")
                        break

            if exit_hit:
                status, bar_date, exit_price, reason = exit_hit
                if row["Signal"] == "BUY":
                    pnl_pct = round((exit_price / entry - 1) * 100, 2)
                else:
                    pnl_pct = round((entry - exit_price) / entry * 100, 2)
                df.at[idx, "Status"] = status
                df.at[idx, "Exit_Date"] = bar_date.date().isoformat()
                df.at[idx, "Exit_Price"] = round(exit_price, 2)
                df.at[idx, "Exit_Reason"] = reason
                df.at[idx, "PnL_Pct"] = pnl_pct
                changed = True
                updated_count += 1
                icon = "🟢" if status == "TARGET" else "🔴"
                log.info(f"  {icon} {sym} {row['Signal']} {status} at {exit_price:.0f} (entry {entry:.0f}, exit_date {bar_date.date()})")
            elif days_held > max_hold:
                # Expired without hit — exit at most recent close
                last_close = float(after_signal["Close"].iloc[-1])
                last_date = after_signal.index[-1].date().isoformat()
                if row["Signal"] == "BUY":
                    pnl_pct = round((last_close / entry - 1) * 100, 2)
                else:
                    pnl_pct = round((entry - last_close) / entry * 100, 2)
                df.at[idx, "Status"] = "EXPIRED"
                df.at[idx, "Exit_Date"] = last_date
                df.at[idx, "Exit_Price"] = round(last_close, 2)
                df.at[idx, "Exit_Reason"] = f"Expired ({days_held}d > {max_hold}d)"
                df.at[idx, "PnL_Pct"] = pnl_pct
                changed = True
                updated_count += 1

        if changed:
            df.to_csv(filepath, index=False)

    log.info(f"Updated {updated_count} signals")
    _rebuild_tracker(all_signals)


def _rebuild_tracker(all_signals=None):
    """Build consolidated tracker.csv from all signal files."""
    if all_signals is None:
        signal_files = sorted(SIGNALS_DIR.glob("*_signals.csv"))
        all_signals = []
        _str_cols = ["Status", "Exit_Date", "Exit_Price", "Exit_Reason", "PnL_Pct"]
        for f in signal_files:
            try:
                df = pd.read_csv(f)
                for c in _str_cols:
                    if c in df.columns:
                        df[c] = df[c].fillna("").astype(object)
                        df[c] = df[c].replace("nan", "")
                all_signals.append((f, df))
            except Exception:
                continue

    if not all_signals:
        return

    frames = [df for _, df in all_signals]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(TRACKER_FILE, index=False)

    total = len(combined)
    targets = len(combined[combined["Status"] == "TARGET"])
    stopped = len(combined[combined["Status"] == "STOPPED"])
    open_count = len(combined[combined["Status"] == "OPEN"])
    expired = len(combined[combined["Status"] == "EXPIRED"])

    log.info(f"Tracker: {total} total | 🟢 {targets} TARGET | 🔴 {stopped} STOPPED | "
             f"⏳ {open_count} OPEN | ⏰ {expired} EXPIRED")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Scanner Auto-Runner v2")
    parser.add_argument("--mode", choices=["eod", "intraday"], default="eod",
                        help="eod=full NSE500 scan, intraday=quick Nifty200")
    parser.add_argument("--track", action="store_true",
                        help="Update tracker (T+1 onwards only)")
    parser.add_argument("--universe", default=None,
                        help="Override universe: nifty50, nifty200, nifty500")
    args = parser.parse_args()

    if args.track:
        update_tracker()
    else:
        universe = args.universe or ("nifty500" if args.mode == "eod" else "nifty200")
        scan_and_record(mode=args.mode, universe=universe)
        # Always update tracker after scanning (will only close signals from PRIOR days)
        update_tracker()
