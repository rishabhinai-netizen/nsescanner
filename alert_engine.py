"""
Alert Engine — NSE Scanner Pro
================================
Runs every 15 minutes during NSE market hours (9:15 AM - 3:30 PM IST).
Triggered by GitHub Actions on schedule.

Does 3 things:
1. Checks if any approaching setup just crossed its trigger level → fires "ENTRY ALERT"
2. Checks open signals for SL/T1 hits → fires "EXIT ALERT"
3. Checks for confluence (same stock flagged by 2+ strategies) → fires "CONFLUENCE ALERT"

Uses Supabase alert_state table to prevent duplicate alerts within the same day.
Requires: SUPABASE_URL, SUPABASE_SERVICE_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
           BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN (optional)
"""

import sys, os, logging, requests, json
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import fetch_batch_daily, fetch_nifty_data, Indicators
from stock_universe import get_stock_universe, get_sector
from scanners import (
    DAILY_SCANNERS, STRATEGY_PROFILES,
    detect_market_regime, collect_approaching_setups,
    run_all_scanners, ScanResult
)
from signal_tracker import _get_secret, _get_supabase, load_signals, update_open_signals_live
from fno_list import get_fno_tag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(msg: str) -> bool:
    token = _get_secret("TELEGRAM_BOT_TOKEN")
    chat  = _get_secret("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.warning("Telegram not configured")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return resp.status_code == 200
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False


# ============================================================
# DEDUP — prevent same alert twice in one day
# ============================================================

def already_alerted(symbol: str, strategy: str, alert_type: str) -> bool:
    sb = _get_supabase()
    if sb:
        try:
            resp = sb.table("alert_state")\
                .select("id")\
                .eq("symbol", symbol)\
                .eq("strategy", strategy)\
                .eq("alert_type", alert_type)\
                .eq("alerted_date", date.today().isoformat())\
                .execute()
            return len(resp.data) > 0
        except Exception:
            pass
    # Fallback: local file dedup
    flag_file = Path(f"/tmp/alerts_{date.today().isoformat()}.json")
    key = f"{symbol}_{strategy}_{alert_type}"
    if flag_file.exists():
        try:
            flags = json.loads(flag_file.read_text())
            if key in flags:
                return True
        except Exception:
            pass
    return False


def mark_alerted(symbol: str, strategy: str, alert_type: str, trigger_price: float = 0):
    sb = _get_supabase()
    if sb:
        try:
            sb.table("alert_state").upsert({
                "symbol":        symbol,
                "strategy":      strategy,
                "alert_type":    alert_type,
                "trigger_price": round(trigger_price, 2),
                "alerted_date":  date.today().isoformat(),
            }, on_conflict="symbol,strategy,alert_type,alerted_date").execute()
        except Exception:
            pass
    # Also write local fallback
    flag_file = Path(f"/tmp/alerts_{date.today().isoformat()}.json")
    key = f"{symbol}_{strategy}_{alert_type}"
    flags = {}
    if flag_file.exists():
        try:
            flags = json.loads(flag_file.read_text())
        except Exception:
            pass
    flags[key] = datetime.now().isoformat()
    flag_file.write_text(json.dumps(flags))


# ============================================================
# ALERT FORMATTERS
# ============================================================

def fmt_entry_alert(r: ScanResult, trigger_pct_move: float = 0) -> str:
    fno  = get_fno_tag(r.symbol)
    p    = STRATEGY_PROFILES.get(r.strategy, {})
    icon = p.get("icon", "📊")
    sqi_val = getattr(r, "sqi", None)
    sqi_str = f"SQI: {sqi_val:.0f} ({getattr(r, 'sqi_grade', '')})" if sqi_val else f"Conf: {r.confidence}%"
    trigger_note = f"\n🔔 Moved {trigger_pct_move:+.1f}% to cross pivot!" if abs(trigger_pct_move) > 0 else ""
    return (
        f"🚨 <b>ENTRY ALERT — {r.symbol}</b> [{fno}]{trigger_note}\n"
        f"{icon} Strategy: <b>{p.get('name', r.strategy)}</b> | {r.signal}\n"
        f"💰 CMP: ₹{r.cmp:,.1f} | Entry: ₹{r.entry:,.1f}\n"
        f"🛑 SL: ₹{r.stop_loss:,.1f} | T1: ₹{r.target_1:,.1f} | R:R 1:{r.risk_reward:.1f}\n"
        f"📊 {sqi_str} | RS: {r.rs_rating:.0f}\n"
        f"📂 Sector: {r.sector} | Regime: {r.regime_fit}"
    )


def fmt_exit_alert(symbol: str, strategy: str, exit_type: str,
                   entry: float, exit_price: float, pnl_pct: float) -> str:
    icon = "🟢" if exit_type == "TARGET" else "🔴"
    p = STRATEGY_PROFILES.get(strategy, {})
    return (
        f"{icon} <b>{'TARGET HIT' if exit_type == 'TARGET' else 'STOP HIT'} — {symbol}</b>\n"
        f"Strategy: {p.get('name', strategy)}\n"
        f"Entry: ₹{entry:,.1f} → Exit: ₹{exit_price:,.1f}\n"
        f"P&L: {pnl_pct:+.1f}%"
    )


def fmt_confluence_alert(symbol: str, strategies: List[str], cmp: float) -> str:
    names = [STRATEGY_PROFILES.get(s, {}).get("name", s) for s in strategies]
    fno   = get_fno_tag(symbol)
    return (
        f"🔥🔥 <b>CONFLUENCE — {symbol}</b> [{fno}]\n"
        f"CMP: ₹{cmp:,.1f}\n"
        f"{len(strategies)} strategies: {', '.join(names)}"
    )


# ============================================================
# CORE ALERT LOGIC
# ============================================================

def check_exit_alerts(data_dict: dict) -> int:
    """Check open signals for SL/T1 hits and fire Telegram."""
    all_signals = load_signals()
    if all_signals is None or all_signals.empty:
        return 0

    status_col = "Status" if "Status" in all_signals.columns else "status"
    open_sigs  = all_signals[all_signals[status_col] == "OPEN"]
    fired = 0

    for _, row in open_sigs.iterrows():
        sym      = row.get("Symbol", row.get("symbol", ""))
        strategy = row.get("Strategy", row.get("strategy", ""))
        signal   = row.get("Signal",   row.get("signal", "BUY"))

        if sym not in data_dict:
            continue

        stock_df = data_dict[sym]
        if stock_df is None or stock_df.empty:
            continue

        recent    = stock_df.iloc[-2:]
        high_rec  = float(recent["high"].max()) if "high" in recent.columns else 0
        low_rec   = float(recent["low"].min())  if "low"  in recent.columns else 0

        try:
            entry = float(row.get("Entry", row.get("entry", 0)) or 0)
            sl    = float(row.get("SL",    row.get("sl",    0)) or 0)
            t1    = float(row.get("T1",    row.get("t1",    0)) or 0)
        except (ValueError, TypeError):
            continue

        if entry <= 0:
            continue

        exit_type  = None
        exit_price = 0.0
        pnl_pct    = 0.0

        if signal == "BUY":
            if sl > 0 and low_rec <= sl and not already_alerted(sym, strategy, "SL_HIT"):
                exit_type  = "STOP"
                exit_price = sl
                pnl_pct    = round((sl / entry - 1) * 100, 2) if entry > 0 else 0
            elif t1 > 0 and high_rec >= t1 and not already_alerted(sym, strategy, "T1_HIT"):
                exit_type  = "TARGET"
                exit_price = t1
                pnl_pct    = round((t1 / entry - 1) * 100, 2) if entry > 0 else 0
        elif signal == "SHORT":
            if sl > 0 and high_rec >= sl and not already_alerted(sym, strategy, "SL_HIT"):
                exit_type  = "STOP"
                exit_price = sl
                pnl_pct    = round((entry / sl - 1) * 100, 2) if sl > 0 else 0
            elif t1 > 0 and low_rec <= t1 and not already_alerted(sym, strategy, "T1_HIT"):
                exit_type  = "TARGET"
                exit_price = t1
                pnl_pct    = round((entry / t1 - 1) * 100, 2) if t1 > 0 else 0

        if exit_type:
            alert_key = "SL_HIT" if exit_type == "STOP" else "T1_HIT"
            msg = fmt_exit_alert(sym, strategy, exit_type, entry, exit_price, pnl_pct)
            if send_telegram(msg):
                mark_alerted(sym, strategy, alert_key, exit_price)
                fired += 1
                log.info(f"EXIT alert sent: {sym} {exit_type} @ {exit_price}")

    return fired


def check_entry_alerts(data_dict: dict, nifty_df, regime: dict) -> int:
    """Check all stocks for fresh scan signals and fire entry Telegram alerts."""
    results = run_all_scanners(
        data_dict, nifty_df,
        daily_only=True,
        regime=regime,
        has_intraday=False,
        sector_rankings={},
        min_rs=65,
        compute_sqi_flag=True,
    )

    if not results:
        return 0

    # Detect confluence
    symbol_strategies: Dict[str, List] = {}
    for strategy, signals in results.items():
        for r in signals:
            if r.symbol not in symbol_strategies:
                symbol_strategies[r.symbol] = []
            symbol_strategies[r.symbol].append((strategy, r))

    confluence = {sym: pairs for sym, pairs in symbol_strategies.items() if len(pairs) >= 2}
    fired = 0

    # Send confluence alerts first (highest priority)
    for sym, pairs in confluence.items():
        strats = [p[0] for p in pairs]
        cmp    = pairs[0][1].cmp
        if not already_alerted(sym, "CONFLUENCE", "TRIGGER"):
            msg = fmt_confluence_alert(sym, strats, cmp)
            if send_telegram(msg):
                mark_alerted(sym, "CONFLUENCE", "TRIGGER", cmp)
                fired += 1
                log.info(f"CONFLUENCE alert: {sym} — {strats}")

    # Individual strategy alerts
    for strategy, signals in results.items():
        for r in signals:
            # Only fire for STRONG/ELITE SQI signals
            sqi_val = getattr(r, "sqi", 50)
            if sqi_val < 50:
                continue
            if already_alerted(r.symbol, strategy, "TRIGGER"):
                continue
            msg = fmt_entry_alert(r)
            if send_telegram(msg):
                mark_alerted(r.symbol, strategy, "TRIGGER", r.entry)
                fired += 1
                log.info(f"ENTRY alert: {r.symbol} {strategy} SQI:{sqi_val}")

    return fired


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def run_market_hours_scan():
    """Full alert scan — called every 15 min by GitHub Actions."""
    now_ist = datetime.now(IST)
    log.info(f"Alert engine starting at {now_ist.strftime('%H:%M IST')}")

    # Check we're in market hours (9:15 AM – 3:30 PM IST, Mon-Fri)
    t = now_ist.time()
    if now_ist.weekday() >= 5:
        log.info("Weekend — skipping")
        return

    from datetime import time as dtime
    if not (dtime(9, 15) <= t <= dtime(15, 30)):
        log.info(f"Outside market hours ({t}) — skipping")
        return

    # Fetch data — use Nifty 200 for speed (runs every 15 min)
    universe = _get_secret("ALERT_UNIVERSE") or "nifty200"
    syms = get_stock_universe(universe)
    log.info(f"Fetching {len(syms)} stocks...")

    data_dict = fetch_batch_daily(syms, "1y")
    if not data_dict:
        log.error("No data fetched — aborting")
        return

    log.info(f"Loaded {len(data_dict)} stocks")

    nifty_df = fetch_nifty_data("1y")

    # Enrich
    enriched = {}
    for sym, df in data_dict.items():
        try:
            enriched[sym] = Indicators.enrich_dataframe(df)
        except Exception:
            enriched[sym] = df

    # Detect regime
    regime = detect_market_regime(nifty_df) if nifty_df is not None else {}
    log.info(f"Regime: {regime.get('regime', 'UNKNOWN')}")

    # 1. Check exit alerts
    exits = check_exit_alerts(enriched)
    log.info(f"Exit alerts fired: {exits}")

    # 2. Check entry alerts  
    entries = check_entry_alerts(enriched, nifty_df, regime)
    log.info(f"Entry alerts fired: {entries}")

    # 3. Daily heartbeat at market open (9:15-9:30 AM IST) — always fires once per day
    if dtime(9, 15) <= t <= dtime(9, 30):
        regime_display = regime.get("regime_display", "Unknown") if regime else "Unknown"
        nifty_close = regime.get("nifty_close", 0) if regime else 0
        nifty_str = f"₹{nifty_close:,.0f}" if nifty_close else "—"
        if exits + entries == 0:
            signal_line = "🔍 Watching for setups — no signals yet"
        else:
            signal_line = f"✅ {entries} entry + {exits} exit alert(s) fired this window"
        send_telegram(
            f"🌅 <b>NSE Scanner — Market Open</b>\n"
            f"📅 {now_ist.strftime('%d %b %Y, %I:%M %p IST')}\n"
            f"📊 Regime: <b>{regime_display}</b> | Nifty: {nifty_str}\n"
            f"🔭 Watching {len(enriched)} stocks\n"
            f"{signal_line}"
        )

    log.info(f"Alert engine done. Fired: exits={exits} entries={entries}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Send a test Telegram message")
    args = parser.parse_args()

    if args.test:
        ok = send_telegram("🔔 NSE Scanner alert engine test — working correctly!")
        print("Telegram test:", "OK" if ok else "FAILED")
    else:
        run_market_hours_scan()
