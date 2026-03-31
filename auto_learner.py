"""
Auto Learner — NSE Scanner Pro
=================================
Runs weekly (Sunday 8 PM IST) via GitHub Actions.

Algorithm:
1. Read all closed trades from Supabase signals table
2. Group by strategy × regime (need ≥ 15 trades to update)
3. Compute actual profit factor
4. Bayesian blend: new_pf = (1 - weight) * prior + weight * actual
   where weight = min(trade_count / 50, 0.5) — caps at 50% influence
5. Write blended PF to strategy_performance table
6. Fire Telegram alert for any strategy in danger (PF < 0.8 for 20+ trades)
7. Generate weekly performance report

Result: signal_quality.py reads live PF from Supabase on every scan.
        The scanner literally gets smarter every week.
"""

import sys, os, logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_quality import STRATEGY_REGIME_PF, reset_pf_cache
from signal_tracker import _get_secret, _get_supabase, load_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REGIMES    = ["EXPANSION", "ACCUMULATION", "DISTRIBUTION", "PANIC"]
MIN_TRADES = 15   # minimum closed trades before we update PF
MAX_WEIGHT = 0.50 # maximum weight given to new data (50% → Bayesian conservatism)
DANGER_PF  = 0.80 # below this = strategy degraded → Telegram alert
AUTO_BLOCK_PF  = 0.65  # below this + 20 trades = auto-block recommendation
MIN_TRADES_BLOCK = 20  # minimum trades before issuing block recommendation


def send_telegram(msg: str) -> bool:
    token = _get_secret("TELEGRAM_BOT_TOKEN")
    chat  = _get_secret("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
        return resp.status_code == 200
    except Exception:
        return False


def compute_actual_pf(trades_df: pd.DataFrame) -> Tuple[float, int, int, int]:
    """
    Compute profit factor from a set of closed trades.
    Returns (pf, total_trades, wins, losses)
    """
    pnl_col = "PnL_Pct" if "PnL_Pct" in trades_df.columns else "pnl_pct"
    pnl = pd.to_numeric(trades_df[pnl_col], errors="coerce").dropna()

    wins_sum   = pnl[pnl > 0].sum()
    losses_sum = abs(pnl[pnl < 0].sum())
    wins       = int((pnl > 0).sum())
    losses     = int((pnl < 0).sum())
    total      = wins + losses

    if losses_sum > 0:
        pf = round(wins_sum / losses_sum, 3)
    elif wins_sum > 0:
        pf = 3.0  # no losses at all — cap at 3.0
    else:
        pf = 1.0  # no data

    return pf, total, wins, losses


def bayesian_blend(prior: float, actual: float, trade_count: int) -> float:
    """
    Bayesian blend: weight actual data by how much we trust it.
    More trades → more weight on actual.
    Caps at MAX_WEIGHT to prevent overreaction to small samples.
    """
    weight = min(trade_count / 50, MAX_WEIGHT)
    blended = prior * (1 - weight) + actual * weight
    return round(blended, 3)


def run_auto_learning(dry_run: bool = False) -> Dict:
    """
    Main auto-learning loop. Returns summary dict.
    dry_run=True: compute but don't write to DB.
    """
    sb = _get_supabase()
    if sb is None:
        log.error("Supabase not configured — cannot run auto-learning")
        return {"error": "No Supabase connection"}

    log.info("Auto-learner starting...")

    # Load all closed trades
    all_signals = load_signals()
    if all_signals is None or all_signals.empty:
        log.info("No signals found — nothing to learn from yet")
        return {"status": "no_data", "updates": 0}

    status_col   = "Status" if "Status"   in all_signals.columns else "status"
    strategy_col = "Strategy" if "Strategy" in all_signals.columns else "strategy"
    regime_col   = "Regime" if "Regime"   in all_signals.columns else "regime"

    closed = all_signals[all_signals[status_col].isin(["TARGET", "STOPPED"])]
    log.info(f"Total closed trades: {len(closed)}")

    if len(closed) < MIN_TRADES:
        log.info(f"Only {len(closed)} closed trades — need ≥ {MIN_TRADES} to start learning")
        return {"status": "insufficient_data", "closed_trades": len(closed)}

    updates = []
    alerts  = []
    summary_rows = []

    for strategy in STRATEGY_REGIME_PF.keys():
        for regime in REGIMES:
            # Filter trades for this strategy × regime
            mask = (
                (closed[strategy_col] == strategy) &
                (closed[regime_col] == regime)
            )
            subset = closed[mask]

            if len(subset) < MIN_TRADES:
                log.debug(f"{strategy}/{regime}: only {len(subset)} trades — skipping")
                continue

            actual_pf, total, wins, losses = compute_actual_pf(subset)
            prior_pf  = STRATEGY_REGIME_PF.get(strategy, {}).get(regime, 1.0)
            blended   = bayesian_blend(prior_pf, actual_pf, total)
            win_rate  = round(wins / total * 100, 1) if total > 0 else 0

            log.info(f"{strategy}/{regime}: actual={actual_pf:.2f} prior={prior_pf:.2f} "
                     f"blended={blended:.2f} trades={total} wr={win_rate}%")

            row = {
                "strategy":    strategy,
                "regime":      regime,
                "prior_pf":    prior_pf,
                "actual_pf":   actual_pf,
                "blended_pf":  blended,
                "trade_count": total,
                "win_count":   wins,
                "loss_count":  losses,
                "last_updated": datetime.utcnow().isoformat(),
            }
            updates.append(row)
            summary_rows.append({
                "Strategy": strategy, "Regime": regime,
                "Prior PF": prior_pf, "Actual PF": actual_pf,
                "Blended PF": blended, "Trades": total,
                "Win Rate": f"{win_rate}%",
                "Status": _pf_status(blended),
            })

            # Check for degradation
            if blended < DANGER_PF and total >= MIN_TRADES:
                direction = "better" if blended > actual_pf else "worse"
                alerts.append(
                    f"⚠️ <b>{strategy} / {regime}</b>: PF = {blended:.2f} "
                    f"({total} trades, WR {win_rate}%) — DEGRADED"
                )
            if blended < AUTO_BLOCK_PF and total >= MIN_TRADES_BLOCK:
                alerts.append(
                    f"🔴 <b>BLOCK RECOMMENDATION: {strategy} / {regime}</b>\n"
                    f"PF {blended:.2f} over {total} real trades. Consider blocking in Settings."
                )

    # Write to Supabase
    if not dry_run and updates and sb:
        try:
            sb.table("strategy_performance").upsert(
                updates, on_conflict="strategy,regime"
            ).execute()
            log.info(f"Updated {len(updates)} strategy×regime entries in Supabase")
            reset_pf_cache()  # force reload on next scan
        except Exception as e:
            log.error(f"Supabase write failed: {e}")

    # Write weekly report
    report = _generate_weekly_report(closed, summary_rows)
    if not dry_run and sb:
        try:
            sb.table("weekly_reports").upsert({
                "week_ending":    date.today().isoformat(),
                "total_signals":  int(len(all_signals)),
                "closed_trades":  int(len(closed)),
                "wins":           int(closed[status_col].eq("TARGET").sum()),
                "losses":         int(closed[status_col].eq("STOPPED").sum()),
                "win_rate":       round(closed[status_col].eq("TARGET").sum() / max(len(closed), 1) * 100, 1),
                "report_json":    report,
            }, on_conflict="week_ending").execute()
        except Exception as e:
            log.warning(f"Report write failed: {e}")

    # Send Telegram summary
    _send_weekly_telegram(closed, summary_rows, alerts, report)

    return {
        "status":       "ok",
        "updates":      len(updates),
        "closed_trades":len(closed),
        "alerts":       len(alerts),
        "dry_run":      dry_run,
    }


def _pf_status(pf: float) -> str:
    if pf >= 1.8: return "🟢 Strong"
    if pf >= 1.3: return "🟡 OK"
    if pf >= 0.8: return "🟠 Watch"
    return "🔴 Danger"


def _generate_weekly_report(closed: pd.DataFrame, summary_rows: list) -> dict:
    """Generate a JSON report dict for storage."""
    status_col   = "Status"   if "Status"   in closed.columns else "status"
    strategy_col = "Strategy" if "Strategy" in closed.columns else "strategy"
    pnl_col      = "PnL_Pct"  if "PnL_Pct"  in closed.columns else "pnl_pct"

    wins   = int(closed[status_col].eq("TARGET").sum())
    losses = int(closed[status_col].eq("STOPPED").sum())
    total  = wins + losses
    pnl    = pd.to_numeric(closed.get(pnl_col, pd.Series()), errors="coerce")

    # Best and worst strategies
    strat_pf = {}
    for strat in closed[strategy_col].unique():
        sub = closed[closed[strategy_col] == strat]
        pf, _, _, _ = compute_actual_pf(sub)
        strat_pf[strat] = pf

    best_strat = max(strat_pf, key=strat_pf.get) if strat_pf else ""

    return {
        "week_ending":   date.today().isoformat(),
        "total_signals": 0,
        "closed":        total,
        "wins":          wins,
        "losses":        losses,
        "win_rate":      round(wins / total * 100, 1) if total > 0 else 0,
        "total_pnl_pct": round(float(pnl.sum()), 2),
        "avg_pnl_pct":   round(float(pnl.mean()), 2) if len(pnl) > 0 else 0,
        "best_strategy": best_strat,
        "strategy_pf":   strat_pf,
        "pf_matrix":     summary_rows,
    }


def _send_weekly_telegram(closed: pd.DataFrame, summary_rows: list,
                           alerts: list, report: dict):
    """Send the weekly learning report to Telegram."""
    wins     = report.get("wins", 0)
    losses   = report.get("losses", 0)
    total    = wins + losses
    win_rate = report.get("win_rate", 0)
    total_pnl = report.get("total_pnl_pct", 0)
    best_strat = report.get("best_strategy", "")

    msg = (
        f"📊 <b>Weekly Auto-Learning Report</b>\n"
        f"Week ending: {date.today().strftime('%d %b %Y')}\n\n"
        f"🏁 Closed trades: {total}\n"
        f"🟢 Wins: {wins} | 🔴 Losses: {losses}\n"
        f"📈 Win Rate: {win_rate}%\n"
        f"💰 Total P&L: {total_pnl:+.1f}%\n"
        f"⭐ Best strategy: {best_strat}\n\n"
    )

    # Add PF changes
    changed = [r for r in summary_rows if abs(r["Actual PF"] - r["Prior PF"]) > 0.1]
    if changed:
        msg += "📉 <b>PF Changes (vs prior):</b>\n"
        for r in changed[:8]:
            delta = r["Blended PF"] - r["Prior PF"]
            arrow = "↑" if delta > 0 else "↓"
            msg += f"  {r['Strategy'][:12]}/{r['Regime'][:4]}: {r['Prior PF']:.2f} {arrow} {r['Blended PF']:.2f}\n"

    if alerts:
        msg += "\n⚠️ <b>Alerts:</b>\n"
        for a in alerts[:5]:
            msg += a + "\n"

    send_telegram(msg)
    log.info("Weekly report sent to Telegram")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NSE Scanner Auto-Learner")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't write to DB")
    parser.add_argument("--force",   action="store_true", help="Ignore MIN_TRADES threshold")
    args = parser.parse_args()

    if args.force:
        MIN_TRADES = 5

    result = run_auto_learning(dry_run=args.dry_run)
    print(f"Auto-learning result: {result}")
