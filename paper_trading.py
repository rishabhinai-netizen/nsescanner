"""
paper_trading.py — Virtual Trading Game for NSE Scanner Pro
===========================================================
A gamified paper trading system linked to live scanner signals.
Virtual ₹5L capital, XP/leveling, achievements, streaks.
"""

import streamlit as st
from supabase import create_client, Client
from datetime import date, datetime
import yfinance as yf
import os

SUPABASE_URL = os.environ.get("SUPABASE_URL", st.secrets.get("SUPABASE_URL", ""))
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", st.secrets.get("SUPABASE_SERVICE_KEY", ""))
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

CAPITAL_PER_TRADE = 50000   # ₹50K virtual deployed per trade
STARTING_CAPITAL  = 500000  # ₹5 lakh total

# ── XP rewards ─────────────────────────────────────────────────────────────────
XP_RULES = {
    "trade_win":          100,
    "trade_loss":          20,   # still learn from losses
    "win_streak_3_bonus": 150,
    "win_streak_5_bonus": 300,
    "elite_signal_win":   250,
    "rr_above_3":          50,   # disciplined trade
    "beat_scanner":       100,   # your exit beat the scanner's T1
}

LEVELS = [
    (0,     "Rookie Trader"),
    (500,   "Apprentice"),
    (1500,  "Journeyman"),
    (3000,  "Seasoned Trader"),
    (6000,  "Expert"),
    (12000, "Market Pro"),
    (25000, "Elite Operator"),
    (50000, "Legend"),
]

def get_level(xp: int) -> tuple[int, str, int, int]:
    """Returns (level_num, title, xp_start, xp_next)"""
    for i, (threshold, title) in enumerate(LEVELS):
        if i + 1 < len(LEVELS):
            next_threshold = LEVELS[i + 1][0]
            if xp < next_threshold:
                return i + 1, title, threshold, next_threshold
    return len(LEVELS), LEVELS[-1][1], LEVELS[-1][0], LEVELS[-1][0]


# ── Portfolio helpers ───────────────────────────────────────────────────────────

def get_paper_portfolio() -> dict:
    try:
        res = supabase.table("paper_portfolio").select("*").limit(1).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        st.error(f"Error loading paper portfolio: {e}")
    return {}


def get_open_paper_trades() -> list:
    try:
        res = (supabase.table("paper_trades")
               .select("*")
               .eq("status", "OPEN")
               .order("created_at", desc=True)
               .execute())
        return res.data or []
    except Exception:
        return []


def get_closed_paper_trades(limit: int = 30) -> list:
    try:
        res = (supabase.table("paper_trades")
               .select("*")
               .neq("status", "OPEN")
               .order("updated_at", desc=True)
               .limit(limit)
               .execute())
        return res.data or []
    except Exception:
        return []


def get_achievements() -> list:
    try:
        res = supabase.table("game_achievements").select("*").order("xp_reward").execute()
        return res.data or []
    except Exception:
        return []


# ── Enter a paper trade from a live signal ─────────────────────────────────────

def enter_paper_trade(signal: dict) -> dict | None:
    """
    Accept a signal dict (from signals table) and open a paper trade.
    Returns the new paper_trade row or None on failure.
    """
    portfolio = get_paper_portfolio()
    if not portfolio:
        st.error("Paper portfolio not found.")
        return None

    current_cap = float(portfolio.get("current_capital", STARTING_CAPITAL))
    if current_cap < CAPITAL_PER_TRADE:
        st.warning("Insufficient virtual capital. Close some trades or reset.")
        return None

    entry = float(signal.get("entry") or signal.get("cmp", 0))
    sl    = float(signal.get("sl", 0))
    t1    = float(signal.get("t1", 0))
    t2    = signal.get("t2")
    qty   = int(CAPITAL_PER_TRADE / entry) if entry > 0 else 1

    trade_data = {
        "signal_id":           signal.get("id"),
        "symbol":              signal["symbol"],
        "strategy":            signal["strategy"],
        "signal":              signal["signal"],
        "entry_price":         entry,
        "stop_loss":           sl,
        "target1":             t1,
        "target2":             float(t2) if t2 else None,
        "rr":                  float(signal.get("rr") or 0),
        "sqi":                 float(signal.get("sqi") or 0),
        "sqi_grade":           signal.get("sqi_grade"),
        "sector":              signal.get("sector"),
        "regime":              signal.get("regime"),
        "virtual_capital_used": CAPITAL_PER_TRADE,
        "qty":                 qty,
        "status":              "OPEN",
        "trade_date":          str(date.today()),
    }

    try:
        res = supabase.table("paper_trades").insert(trade_data).execute()
        if res.data:
            # Deduct capital from paper_portfolio
            new_cap = current_cap - CAPITAL_PER_TRADE
            supabase.table("paper_portfolio").update({
                "current_capital": new_cap,
                "last_trade_at": datetime.now().isoformat(),
            }).eq("id", portfolio["id"]).execute()
            _check_first_trade_achievement(portfolio)
            return res.data[0]
    except Exception as e:
        st.error(f"Failed to enter paper trade: {e}")
    return None


# ── Exit a paper trade ─────────────────────────────────────────────────────────

def exit_paper_trade(trade_id: str, exit_price: float, exit_reason: str = "MANUAL_EXIT") -> bool:
    """
    Close an open paper trade, calculate P&L, award XP.
    """
    try:
        trade_res = supabase.table("paper_trades").select("*").eq("id", trade_id).single().execute()
        trade = trade_res.data
        if not trade:
            return False

        portfolio = get_paper_portfolio()
        entry     = float(trade["entry_price"])
        qty       = int(trade["qty"])
        direction = trade["signal"]  # BUY or SHORT

        if direction == "BUY":
            pnl     = (exit_price - entry) * qty
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:  # SHORT
            pnl     = (entry - exit_price) * qty
            pnl_pct = ((entry - exit_price) / entry) * 100

        won = pnl > 0
        xp  = _calculate_xp(trade, pnl_pct, won)

        # Update trade
        supabase.table("paper_trades").update({
            "status":      exit_reason,
            "exit_price":  exit_price,
            "exit_reason": exit_reason,
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(pnl_pct, 2),
            "xp_earned":   xp,
            "exit_date":   str(date.today()),
            "updated_at":  datetime.now().isoformat(),
        }).eq("id", trade_id).execute()

        # Update portfolio stats
        pid        = portfolio["id"]
        new_cap    = float(portfolio.get("current_capital", 0)) + float(trade["virtual_capital_used"]) + pnl
        total_pnl  = float(portfolio.get("total_pnl", 0)) + pnl
        total_trades = int(portfolio.get("total_trades", 0)) + 1
        wins       = int(portfolio.get("winning_trades", 0)) + (1 if won else 0)
        losses     = int(portfolio.get("losing_trades", 0)) + (0 if won else 1)
        streak     = int(portfolio.get("current_streak", 0))
        streak     = streak + 1 if won else (streak - 1 if streak >= 0 else streak - 1)
        if won and streak < 0:
            streak = 1
        if not won and streak > 0:
            streak = -1

        max_win  = max(int(portfolio.get("max_win_streak", 0)), streak if streak > 0 else 0)
        max_loss = max(int(portfolio.get("max_loss_streak", 0)), abs(streak) if streak < 0 else 0)
        new_xp   = int(portfolio.get("xp_points", 0)) + xp
        _, rank, _, _ = get_level(new_xp)

        supabase.table("paper_portfolio").update({
            "current_capital":  round(new_cap, 2),
            "total_pnl":        round(total_pnl, 2),
            "total_pnl_pct":    round((total_pnl / STARTING_CAPITAL) * 100, 2),
            "total_trades":     total_trades,
            "winning_trades":   wins,
            "losing_trades":    losses,
            "current_streak":   streak,
            "max_win_streak":   max_win,
            "max_loss_streak":  max_loss,
            "xp_points":        new_xp,
            "rank_title":       rank,
            "last_trade_at":    datetime.now().isoformat(),
        }).eq("id", pid).execute()

        # Log XP
        supabase.table("xp_log").insert({
            "action":      "TRADE_WIN" if won else "TRADE_LOSS",
            "xp_gained":   xp,
            "description": f"{'Won' if won else 'Lost'} {trade['symbol']} {trade['strategy']} ({pnl_pct:+.1f}%)",
        }).execute()

        _check_achievements(portfolio, trade, pnl_pct, won, streak, wins)
        return True

    except Exception as e:
        st.error(f"Failed to exit paper trade: {e}")
        return False


def _calculate_xp(trade: dict, pnl_pct: float, won: bool) -> int:
    xp = XP_RULES["trade_win"] if won else XP_RULES["trade_loss"]
    if won and trade.get("sqi_grade") == "ELITE":
        xp += XP_RULES["elite_signal_win"]
    if won and float(trade.get("rr") or 0) >= 3:
        xp += XP_RULES["rr_above_3"]
    # beat scanner: exit above T1 on a BUY
    if trade["signal"] == "BUY" and won and trade.get("target1"):
        exit_p = trade.get("exit_price") or 0
        if float(exit_p) > float(trade["target1"]):
            xp += XP_RULES["beat_scanner"]
    return xp


def _unlock_achievement(key: str):
    try:
        supabase.table("game_achievements").update({
            "is_unlocked": True,
            "unlocked_at": datetime.now().isoformat(),
        }).eq("achievement_key", key).eq("is_unlocked", False).execute()
    except Exception:
        pass


def _check_first_trade_achievement(portfolio: dict):
    if int(portfolio.get("total_trades", 0)) == 0:
        _unlock_achievement("first_trade")


def _check_achievements(portfolio: dict, trade: dict, pnl_pct: float, won: bool, streak: int, wins: int):
    if wins == 1:
        _unlock_achievement("first_win")
    if streak >= 3:
        _unlock_achievement("streak_3")
    if streak >= 5:
        _unlock_achievement("streak_5")
    if streak >= 10:
        _unlock_achievement("streak_10")
    if float(portfolio.get("current_capital", 0)) >= STARTING_CAPITAL * 2:
        _unlock_achievement("doubled_capital")


# ── Auto-update open trades against live prices ─────────────────────────────────

def auto_update_paper_trades():
    """
    Call this once when the game page loads.
    Checks all open trades against live yfinance prices and auto-closes if SL/T1 hit.
    """
    open_trades = get_open_paper_trades()
    if not open_trades:
        return

    symbols = list(set(t["symbol"] for t in open_trades))
    prices  = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            hist   = ticker.history(period="1d")
            if not hist.empty:
                prices[sym] = {
                    "current": float(hist["Close"].iloc[-1]),
                    "high":    float(hist["High"].iloc[-1]),
                    "low":     float(hist["Low"].iloc[-1]),
                }
        except Exception:
            pass

    for trade in open_trades:
        sym   = trade["symbol"]
        if sym not in prices:
            continue
        lp    = prices[sym]["current"]
        high  = prices[sym]["high"]
        low   = prices[sym]["low"]
        sl    = float(trade["stop_loss"])
        t1    = float(trade["target1"])
        direction = trade["signal"]

        if direction == "BUY":
            if low <= sl:
                exit_paper_trade(trade["id"], sl, "STOPPED")
            elif high >= t1:
                exit_paper_trade(trade["id"], t1, "TARGET1")
        else:  # SHORT
            if high >= sl:
                exit_paper_trade(trade["id"], sl, "STOPPED")
            elif low <= t1:
                exit_paper_trade(trade["id"], t1, "TARGET1")


# ── Streamlit page ─────────────────────────────────────────────────────────────

def render_paper_trading_page():
    st.title("🎮 Virtual Trading Game")
    st.caption("Paper trade every scanner signal with virtual ₹5L. No real money, real market prices.")

    # Auto-update open trades silently
    with st.spinner("Syncing live prices..."):
        auto_update_paper_trades()

    portfolio = get_paper_portfolio()
    if not portfolio:
        st.error("Could not load paper portfolio. Check Supabase connection.")
        return

    xp        = int(portfolio.get("xp_points", 0))
    level_num, rank, xp_start, xp_next = get_level(xp)
    xp_progress = (xp - xp_start) / max(xp_next - xp_start, 1) if xp_next > xp_start else 1.0

    # ── Header card ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    starting = float(portfolio.get("starting_capital", STARTING_CAPITAL))
    current  = float(portfolio.get("current_capital", STARTING_CAPITAL))
    total_pnl_pct = float(portfolio.get("total_pnl_pct", 0))
    wins  = int(portfolio.get("winning_trades", 0))
    losses = int(portfolio.get("losing_trades", 0))
    total = int(portfolio.get("total_trades", 0))
    win_rate = (wins / total * 100) if total > 0 else 0
    streak = int(portfolio.get("current_streak", 0))

    with col1:
        delta_pct = ((current - starting) / starting) * 100
        st.metric("Virtual Capital", f"₹{current:,.0f}", f"{delta_pct:+.1f}%")
    with col2:
        st.metric("Win Rate", f"{win_rate:.0f}%", f"{wins}W / {losses}L")
    with col3:
        streak_label = f"🔥 {streak} win streak" if streak > 0 else (f"📉 {abs(streak)} loss streak" if streak < 0 else "—")
        st.metric("Streak", streak_label)
    with col4:
        st.metric("Total P&L", f"{total_pnl_pct:+.1f}%", f"₹{float(portfolio.get('total_pnl', 0)):+,.0f}")

    # ── XP / Level bar ───────────────────────────────────────────────────────
    st.markdown(f"**Level {level_num} — {rank}** &nbsp;&nbsp; `{xp} XP`")
    st.progress(xp_progress, text=f"{xp - xp_start} / {xp_next - xp_start} XP to Level {level_num + 1}")
    st.divider()

    open_trades_check = get_open_paper_trades()
    default_tab = 3 if not open_trades_check else 0
    tabs = st.tabs(["📋 Open Positions", "📜 History", "🏆 Achievements", "⚡ Take a Trade"])
    if default_tab == 3:
        st.info("👆 No open positions yet. Click **⚡ Take a Trade** tab above to accept a signal and start playing.")

    # ── OPEN POSITIONS ────────────────────────────────────────────────────────
    with tabs[0]:
        open_trades = open_trades_check  # already fetched above
        if not open_trades:
            st.info("No open paper positions. Go to 'Take a Trade' to accept a signal.")
        for t in open_trades:
            sym   = t["symbol"]
            entry = float(t["entry_price"])
            sl    = float(t["stop_loss"])
            t1    = float(t["target1"])
            qty   = int(t["qty"])
            direction = t["signal"]
            sqi_g = t.get("sqi_grade", "—")

            # Get live price
            try:
                ticker = yf.Ticker(f"{sym}.NS")
                lp     = float(ticker.fast_info["last_price"])
            except Exception:
                lp = entry

            live_pnl_pct = ((lp - entry) / entry * 100) if direction == "BUY" else ((entry - lp) / entry * 100)
            live_pnl_rs  = (lp - entry) * qty if direction == "BUY" else (entry - lp) * qty

            with st.expander(f"{sym}  {'🟢 BUY' if direction=='BUY' else '🔴 SHORT'}  |  Live: ₹{lp:,.1f}  |  P&L: {live_pnl_pct:+.1f}%  |  {sqi_g}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Entry", f"₹{entry:,.1f}")
                c2.metric("Stop Loss", f"₹{sl:,.1f}", f"Risk: ₹{abs(entry-sl)*qty:,.0f}")
                c3.metric("Target 1", f"₹{t1:,.1f}", f"Reward: ₹{abs(t1-entry)*qty:,.0f}")
                st.metric("Live P&L", f"₹{live_pnl_rs:+,.0f}", f"{live_pnl_pct:+.1f}%",
                          delta_color="normal" if live_pnl_rs >= 0 else "inverse")

                exit_col, price_col = st.columns([3, 1])
                custom_exit = exit_col.number_input("Manual exit price", value=float(lp), key=f"exit_{t['id']}", step=0.5)
                if price_col.button("Exit", key=f"btn_{t['id']}", type="primary"):
                    if exit_paper_trade(t["id"], custom_exit, "MANUAL_EXIT"):
                        st.success(f"Trade closed. P&L: {live_pnl_pct:+.1f}%")
                        st.rerun()

    # ── HISTORY ───────────────────────────────────────────────────────────────
    with tabs[1]:
        closed = get_closed_paper_trades()
        if not closed:
            st.info("No closed trades yet.")
        for t in closed:
            pnl_pct = float(t.get("pnl_pct") or 0)
            icon    = "✅" if pnl_pct > 0 else "❌"
            xp_e    = int(t.get("xp_earned") or 0)
            st.write(f"{icon} **{t['symbol']}** ({t['strategy']}) &nbsp; {pnl_pct:+.1f}% &nbsp; ₹{float(t.get('pnl') or 0):+,.0f} &nbsp; +{xp_e} XP &nbsp; *{t['exit_reason']}* &nbsp; {t.get('exit_date','')}")

    # ── ACHIEVEMENTS ──────────────────────────────────────────────────────────
    with tabs[2]:
        achievements = get_achievements()
        locked   = [a for a in achievements if not a["is_unlocked"]]
        unlocked = [a for a in achievements if a["is_unlocked"]]

        if unlocked:
            st.subheader(f"Unlocked ({len(unlocked)})")
            cols = st.columns(3)
            for i, a in enumerate(unlocked):
                with cols[i % 3]:
                    st.success(f"{a['icon']} **{a['title']}**\n\n{a['description']}\n\n+{a['xp_reward']} XP")

        st.subheader(f"Locked ({len(locked)})")
        cols = st.columns(3)
        for i, a in enumerate(locked):
            with cols[i % 3]:
                st.info(f"🔒 **{a['title']}**\n\n{a['description']}\n\n+{a['xp_reward']} XP")

    # ── TAKE A TRADE ──────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Accept a Signal")
        st.caption(f"Each trade deploys ₹{CAPITAL_PER_TRADE:,} virtual. Available: ₹{float(portfolio.get('current_capital', 0)):,.0f}")

        # Load today's open scanner signals
        try:
            sigs = (supabase.table("signals")
                    .select("*")
                    .eq("date", str(date.today()))
                    .eq("status", "OPEN")
                    .order("sqi", desc=True)
                    .limit(20)
                    .execute())
            today_signals = sigs.data or []
        except Exception:
            today_signals = []

        if not today_signals:
            st.warning("No open signals today yet. Scanner runs at 4:30 PM and 7:00 PM IST.")
        else:
            # Check which ones are already paper-traded today
            already = {t["symbol"] + t["strategy"] for t in open_trades}

            for sig in today_signals:
                key = sig["symbol"] + sig["strategy"]
                if key in already:
                    st.write(f"✓ Already in paper position: **{sig['symbol']}** ({sig['strategy']})")
                    continue

                sqi_g = sig.get("sqi_grade", "—")
                grade_color = {"ELITE": "🥇", "STRONG": "🥈", "MODERATE": "🥉", "WEAK": "⚠️"}.get(sqi_g, "")
                rr    = float(sig.get("rr") or 0)

                with st.expander(f"{grade_color} **{sig['symbol']}** — {sig['strategy']} | {sig['signal']} | SQI: {sqi_g} | R:R {rr:.1f}"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Entry", f"₹{float(sig.get('entry') or sig.get('cmp', 0)):,.1f}")
                    c2.metric("Stop Loss", f"₹{float(sig.get('sl', 0)):,.1f}")
                    c3.metric("Target 1", f"₹{float(sig.get('t1', 0)):,.1f}")
                    c4.metric("R:R", f"1:{rr:.1f}")
                    st.write(f"Sector: **{sig.get('sector','—')}** | Regime: **{sig.get('regime','—')}** | Fit: **{sig.get('regime_fit','—')}**")

                    if st.button(f"🎯 Accept Trade — Deploy ₹{CAPITAL_PER_TRADE:,}", key=f"accept_{sig['id']}", type="primary"):
                        result = enter_paper_trade(sig)
                        if result:
                            st.success(f"Paper trade opened! {sig['symbol']} entered at ₹{float(sig.get('entry', 0)):,.1f}")
                            st.rerun()

    # ── Reset option ─────────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚠️ Reset Paper Portfolio"):
        st.warning("This will reset your virtual capital to ₹5L and clear all paper trades. XP and achievements are kept.")
        if st.button("Reset Capital", type="secondary"):
            portfolio = get_paper_portfolio()
            supabase.table("paper_trades").delete().eq("status", "OPEN").execute()
            supabase.table("paper_portfolio").update({
                "current_capital":  STARTING_CAPITAL,
                "total_trades":     0,
                "winning_trades":   0,
                "losing_trades":    0,
                "total_pnl":        0,
                "total_pnl_pct":    0,
                "current_streak":   0,
                "reset_count":      int(portfolio.get("reset_count", 0)) + 1,
            }).eq("id", portfolio["id"]).execute()
            st.success("Portfolio reset to ₹5L. Your XP and achievements remain.")
            st.rerun()
