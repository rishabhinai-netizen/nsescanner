"""Alerts — Telegram. Only LIVE gate + Tier A/B signals ever alert.
Incubating signals are tracked silently; noise is the enemy of trust."""
import logging
import requests

from core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger("nx.alerts")


def send_telegram(msg: str, chat_id: str = None) -> bool:
    token = TELEGRAM_BOT_TOKEN()
    chat = chat_id or TELEGRAM_CHAT_ID()
    if not token or not chat:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": msg, "parse_mode": "HTML"},
                          timeout=10)
        return r.ok
    except Exception as e:
        logger.warning(f"telegram failed: {e}")
        return False


def fmt_signal(row: dict, regime: str) -> str:
    return (f"🎯 <b>NX Signal — {row['strategy']}</b>\n"
            f"<b>{row['symbol']}</b> {row['side']} | SQI {row['sqi']} ({row['sqi_tier']})\n"
            f"Entry ₹{row['entry']} | Stop ₹{row['stop']}\n"
            f"T1 ₹{row['target1']} | T2 ₹{row.get('target2','—')} | RR {row['rr']}\n"
            f"RS {row.get('rs_rank','—')} | Regime: {regime}\n"
            f"⚠️ Risk 1–2% of capital max")


def fmt_summary(n_live: int, n_incubating: int, n_blocked: int, regime: dict) -> str:
    return (f"📊 <b>NX Scan Complete</b>\n"
            f"Regime: <b>{regime['regime']}</b> (score {regime['score']})\n"
            f"LIVE: {n_live} | Incubating: {n_incubating} | Blocked: {n_blocked}")
