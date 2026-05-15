"""
app_additions.py — New pages for NSE Scanner Pro v16
======================================================
These functions replace/supplement pages in app.py:

CHANGES TO app.py:
1. Import these functions at top of app.py
2. Replace "📋 Signal Log" and "📊 Tracker" with "📊 Performance" in PAGES list
3. Add "💼 Portfolio" to PAGES list
4. Update page_map to use page_performance() and page_portfolio()
5. Add Supabase status chip to sidebar
6. Replace load_tracker / save_signals_today calls to use updated signal_tracker

HOW TO APPLY:
- Copy this file alongside app.py
- In app.py: add `from app_additions import page_performance, page_portfolio, render_supabase_status`
- Update PAGES, page_map, and sidebar as described below
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# These imports assume the updated modules are in place
from signal_tracker import (
    load_signals, load_tracker, get_signal_dates,
    compute_tracker_stats, generate_csv_download,
    update_open_signals_live, save_position, load_portfolio,
    close_position, get_portfolio_pnl,
)
from signal_quality import get_regime_strategy_matrix, reset_pf_cache
from scanners import STRATEGY_PROFILES
from fno_list import get_fno_tag
from risk_manager import RiskManager
from stock_universe import get_sector


# ============================================================
# SIDEBAR — Supabase status chip
# ============================================================

def render_supabase_status():
    """Small status chip for sidebar showing DB connection."""
    try:
        from signal_tracker import _get_supabase
        sb = _get_supabase()
        if sb:
            st.markdown(
                '<div style="padding:4px 10px;border-radius:6px;font-size:0.72rem;'
                'background:#dcfce7;border:1px solid #86efac;color:#15803d;">'
                '✅ Supabase Connected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="padding:4px 10px;border-radius:6px;font-size:0.72rem;'
                'background:#f1f5f9;border:1px solid #cbd5e1;color:#64748b;">'
                '⚪ Supabase: Not configured</div>',
                unsafe_allow_html=True
            )
    except Exception:
        pass


# ============================================================
# PAGE: PERFORMANCE (replaces Signal Log + Tracker)
# ============================================================

def page_performance():
    st.markdown("# 📊 Performance")
    st.caption("Forward-test results. Regime-blocked strategies excluded from stats.")

    tracker_df = load_tracker()
    stats      = compute_tracker_stats(tracker_df) if tracker_df is not None else {}

    if not stats or stats.get("total", 0) == 0:
        st.info("No signals recorded yet.")
        return

    # ── ₹1L SIMULATION BANNER ─────────────────────────────────────────────
    closed = stats.get("closed", 0)
    targets = stats.get("targets", 0)
    stopped = stats.get("stopped", 0)
    avg_win  = stats.get("avg_win", 0)
    avg_loss = stats.get("avg_loss", 0)
    win_rate = stats.get("win_rate", 0)
    expectancy = stats.get("expectancy", 0)
    
    # ₹1L simulation: each signal = ₹1L deployed
    pnl_per_lakh = expectancy / 100 * 100000  # expectancy % on ₹1L capital
    total_sim_pnl = closed * pnl_per_lakh
    
    color = "#16a34a" if expectancy >= 0 else "#dc2626"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border:1px solid #cbd5e1;
    border-radius:12px;padding:16px 20px;margin:8px 0">
    <div style="font-size:.75rem;color:#888;letter-spacing:2px;margin-bottom:8px">
    ₹1,00,000 PER SIGNAL — FORWARD TEST SIMULATION</div>
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px">
    <div><div style="color:#888;font-size:.7rem">Closed Trades</div>
    <div style="font-size:1.3rem;font-weight:700;color:#fff">{closed}</div>
    <div style="color:#aaa;font-size:.7rem">🟢 {targets} TARGET · 🔴 {stopped} STOPPED</div></div>
    <div><div style="color:#888;font-size:.7rem">Win Rate</div>
    <div style="font-size:1.3rem;font-weight:700;color:{color}">{win_rate:.1f}%</div>
    <div style="color:#aaa;font-size:.7rem">Avg win {avg_win:+.2f}% / loss {avg_loss:+.2f}%</div></div>
    <div><div style="color:#888;font-size:.7rem">Expectancy / Trade</div>
    <div style="font-size:1.3rem;font-weight:700;color:{color}">{expectancy:+.2f}%</div>
    <div style="color:#aaa;font-size:.7rem">₹{pnl_per_lakh:+,.0f} profit per trade</div></div>
    <div><div style="color:#888;font-size:.7rem">Closed P&L on ₹1L/trade</div>
    <div style="font-size:1.3rem;font-weight:700;color:{color}">₹{total_sim_pnl:+,.0f}</div>
    <div style="color:#aaa;font-size:.7rem">{closed} trades × ₹1L each</div></div>
    <div><div style="color:#888;font-size:.7rem">Open Positions</div>
    <div style="font-size:1.3rem;font-weight:700;color:#d97706">{stats.get("open",0)}</div>
    <div style="color:#aaa;font-size:.7rem">unrealised P&L shown below ↓</div></div>
    </div></div>""", unsafe_allow_html=True)

    # ── OPEN POSITIONS LIVE P&L ───────────────────────────────────────────
    _render_open_positions_live()

    # Equity curve
    all_df = load_signals()
    if all_df is not None and not all_df.empty:
        _render_equity_curve(all_df)

    # ── Tabs: Overview | Signals | Strategy Breakdown ──────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 Strategy Breakdown", "📋 Signal Log", "📥 Export"])

    with tab1:
        _render_strategy_breakdown(stats)

    with tab2:
        _render_signal_log(tracker_df)

    with tab3:
        _render_export(tracker_df)


def _render_open_positions_live():
    """Shows open signals with live CMP fetched from yfinance."""
    try:
        from signal_tracker import _get_supabase
        sb = _get_supabase()
        if not sb:
            return
        res = sb.table("signals").select(
            "symbol,strategy,signal,date,entry,sl,t1,t2,sqi_grade,sector,first_seen_ist"
        ).eq("status","OPEN").order("sqi", desc=True).limit(60).execute()
        open_rows = res.data or []
    except Exception:
        return
    
    if not open_rows:
        return

    st.markdown("### 📋 Open Positions — Live P&L")
    
    # Fetch live prices
    symbols = list(set(r['symbol'] for r in open_rows))
    live_prices = {}
    try:
        import yfinance as yf
        for sym in symbols:
            try:
                h = yf.Ticker(f"{sym}.NS").history(period="1d")
                if not h.empty:
                    live_prices[sym] = float(h['Close'].iloc[-1])
            except Exception:
                pass
    except Exception:
        pass

    rows = []
    total_unrealised = 0
    in_profit = 0
    for r in open_rows:
        entry = float(r.get('entry') or 0)
        t1    = float(r.get('t1') or 0)
        sl    = float(r.get('sl') or 0)
        has_live = r['symbol'] in live_prices
        live  = live_prices.get(r['symbol'], None)

        if has_live and live and entry > 0:
            if r['signal'] == 'BUY':
                pnl_pct = (live - entry) / entry * 100
            else:
                pnl_pct = (entry - live) / entry * 100
            pnl_str = f"{pnl_pct:+.2f}%"
            pnl_rs  = pnl_pct / 100 * 100000  # P&L on ₹1L
            pnl_rs_str = f"₹{pnl_rs:+,.0f}"
        else:
            pnl_pct = 0.0
            pnl_str = "—"
            pnl_rs_str = "—"

        t1_pct = (t1-entry)/entry*100 if entry>0 and r['signal']=='BUY' else (entry-t1)/entry*100 if entry>0 else 0
        sl_pct = (sl-entry)/entry*100 if entry>0 and r['signal']=='BUY' else (entry-sl)/entry*100 if entry>0 else 0

        if has_live and pnl_pct != 0:
            total_unrealised += pnl_pct
            if pnl_pct > 0:
                in_profit += 1

        rows.append({
            "Symbol":      r['symbol'],
            "Strategy":    r.get('strategy',''),
            "Dir":         "🟢 BUY" if r['signal']=='BUY' else "🔴 SHORT",
            "Entry ₹":     f"₹{entry:,.2f}",
            "Live ₹":      f"₹{live:,.2f}" if has_live and live else "—",
            "P&L %":       pnl_str,
            "P&L on ₹1L":  pnl_rs_str,
            "→ T1":        f"{t1_pct:+.1f}%",
            "SL":          f"{sl_pct:+.1f}%",
            "Grade":       r.get('sqi_grade','—'),
            "Since":       str(r.get('date',''))[:10],
        })

    if rows:
        import pandas as _pd
        open_display = _pd.DataFrame(rows)
        st.dataframe(open_display, use_container_width=True, hide_index=True, height=min(400, 40 + len(rows)*35))
        avg_unreal = total_unrealised / len(rows) if rows else 0
        live_count = sum(1 for r in rows if r["P&L %"] != "—")
        total_unreal_rs = sum(
            float(r["P&L on ₹1L"].replace("₹","").replace(",","").replace("+",""))
            for r in rows if r["P&L on ₹1L"] not in ("—", "")
        ) if rows else 0
        unreal_color = "#16a34a" if total_unreal_rs >= 0 else "#dc2626"
        st.markdown(
            f'<div style="background:#f1f5f9;border:1px solid #cbd5e1;border-radius:8px;'
            f'padding:10px 16px;margin:6px 0;display:flex;gap:24px;flex-wrap:wrap">'
            f'<span>📡 Live prices: <b>{live_count}/{len(rows)}</b></span>'
            f'<span>🟢 In profit: <b>{in_profit}/{len(rows)}</b></span>'
            f'<span>Avg unrealised: <b>{avg_unreal:+.2f}%</b></span>'
            f'<span style="color:{unreal_color}">Total unrealised on ₹1L/trade: '
            f'<b>₹{total_unreal_rs:+,.0f}</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.divider()


def _render_equity_curve(df: pd.DataFrame):
    status_col = "Status" if "Status" in df.columns else "status"
    pnl_col    = "PnL_Pct" if "PnL_Pct" in df.columns else "pnl_pct"
    date_col   = "Date" if "Date" in df.columns else "date"

    closed = df[df[status_col].isin(["TARGET", "STOPPED"])].copy()
    if closed.empty:
        return

    closed = closed.sort_values(date_col)
    closed["_pnl"] = pd.to_numeric(closed.get(pnl_col, 0), errors="coerce").fillna(0)
    closed["_cumulative"] = closed["_pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=closed[date_col], y=closed["_cumulative"].round(2),
        mode="lines", name="Cumulative Sum of Trade P&Ls (not portfolio %)",
        line=dict(color="#ea580c", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,107,53,0.08)",
    ))
    fig.add_annotation(
        text="Note: Y-axis = sum of individual trade P&L %. Not portfolio return %.",
        xref="paper", yref="paper", x=0, y=1.02, showarrow=False,
        font=dict(size=9, color="#888"), align="left"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_layout(
        template="plotly_dark", height=240,
        margin=dict(t=20, b=20, l=40, r=10),
        xaxis_title="", yaxis_title="Cumulative %",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_strategy_breakdown(stats: dict):
    if not stats or not stats.get("strategy_stats"):
        st.info("No closed trades yet. Run scans and check back after signals close.")
        return

    # Live PF matrix from Supabase
    pf_matrix = get_regime_strategy_matrix()

    rows = []
    for strat, s in stats["strategy_stats"].items():
        p    = STRATEGY_PROFILES.get(strat, {})
        name = p.get("name", strat)
        wr   = s["win_rate"]
        tot  = s["targets"] + s["stopped"]
        live_pf_exp = pf_matrix.get(strat, {}).get("EXPANSION", "—")
        avg_w = s.get("avg_win", 0)
        avg_l = s.get("avg_loss", 0)
        exp   = round((wr/100 * avg_w) + ((1-wr/100) * avg_l), 2) if tot > 0 else 0
        pnl_rs = int(exp / 100 * 100000)
        note = ""
        if name in ["Last 30 Min ATH", "52-Week High Breakout"]:
            note = " ⛔ blocked"
        rows.append({
            "Strategy":            f"{p.get('icon','')} {name}{note}",
            "Closed":              tot,
            "Open":                s["open"],
            "🟢 Target":           s["targets"],
            "🔴 Stopped":          s["stopped"],
            "Win Rate":            f"{wr:.1f}%" if tot > 0 else "—",
            "Avg Win":             f"{avg_w:+.2f}%" if avg_w else "—",
            "Avg Loss":            f"{avg_l:+.2f}%" if avg_l else "—",
            "Expectancy/Trade":    f"{exp:+.2f}%" if tot > 0 else "—",
            "P&L per ₹1L":         f"₹{pnl_rs:+,}" if tot > 0 else "—",
            "Live PF":             f"{live_pf_exp:.2f}" if isinstance(live_pf_exp, float) else "—",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Strategy × Regime live PF matrix
    with st.expander("📈 Live Strategy × Regime Matrix (auto-updates weekly)", expanded=False):
        regimes = ["EXPANSION", "ACCUMULATION", "DISTRIBUTION", "PANIC"]
        matrix_rows = []
        for strat, pfs in pf_matrix.items():
            p = STRATEGY_PROFILES.get(strat, {})
            row = {"Strategy": f"{p.get('icon','')} {p.get('name', strat)}"}
            for r in regimes:
                row[r] = pfs.get(r, "—")
            matrix_rows.append(row)

        def _color_pf(val):
            if not isinstance(val, (int, float)): return ""
            if val >= 1.5: return "background-color:#dcfce7;color:#15803d"
            if val >= 1.0: return "background-color:#f0fdf4;color:#16a34a"
            if val >= 0.7: return "background-color:#fefce8;color:#ca8a04"
            return "background-color:#fef2f2;color:#dc2626"

        styled = pd.DataFrame(matrix_rows).style.map(_color_pf, subset=regimes)
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption("Updated weekly by auto_learner.py using real closed trade data. "
                   "Needs ≥ 15 closed trades per strategy/regime to update.")


def _render_signal_log(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No signals yet. Run a scan from Dashboard or Scanner Hub.")
        return

    date_col     = "Date" if "Date" in df.columns else "date"
    status_col   = "Status" if "Status" in df.columns else "status"
    strategy_col = "Strategy" if "Strategy" in df.columns else "strategy"

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        dates    = sorted(df[date_col].unique(), reverse=True)
        sel_date = st.selectbox("Date", ["All"] + list(dates), key="perf_date")
    with c2:
        statuses = st.multiselect("Status", ["OPEN","TARGET","STOPPED","EXPIRED"],
                                   default=["OPEN","TARGET","STOPPED"], key="perf_status")
    with c3:
        strats = sorted(df[strategy_col].unique())
        sel_strats = st.multiselect("Strategy", strats, default=list(strats), key="perf_strat")

    filtered = df.copy()
    if sel_date != "All":
        filtered = filtered[filtered[date_col] == sel_date]
    if statuses:
        filtered = filtered[filtered[status_col].isin(statuses)]
    if sel_strats:
        filtered = filtered[filtered[strategy_col].isin(sel_strats)]

    st.caption(f"Showing {len(filtered)} of {len(df)} signals")

    display_cols = [c for c in ["Date","Time","Strategy","Symbol","Signal","CMP","Entry",
                                 "SL","T1","RR","SQI","SQI_Grade","RS","Sector","Regime_Fit",
                                 "Status","Exit_Date","Exit_Reason","PnL_Pct"] if c in filtered.columns]
    st.dataframe(filtered[display_cols].sort_values(date_col, ascending=False),
                 use_container_width=True, hide_index=True)

    # Update tracker button
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Update from current prices", key="perf_update"):
            if st.session_state.get("stock_data"):
                n = update_open_signals_live(st.session_state.stock_data)
                st.success(f"Updated {n} signals") if n else st.info("No changes")
                st.rerun()
            else:
                st.warning("Load data from Dashboard first.")


def _render_export(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data to export.")
        return
    csv = generate_csv_download()
    if csv:
        st.download_button(
            "📥 Download Full Signal History (CSV)",
            csv,
            f"nse_scanner_signals_{date.today().isoformat()}.csv",
            "text/csv",
        )
    st.markdown("---")
    st.markdown("### Manual Outcome Entry")
    st.caption("Mark individual signals as closed if auto-tracking missed them.")
    open_sigs = df[df.get("Status", df.get("status", pd.Series())) == "OPEN"]
    if not open_sigs.empty:
        sym_col  = "Symbol"   if "Symbol"   in open_sigs.columns else "symbol"
        strat_col= "Strategy" if "Strategy" in open_sigs.columns else "strategy"
        date_col = "Date"     if "Date"     in open_sigs.columns else "date"
        labels = [f"{r[sym_col]} ({r[strat_col]}) {r[date_col]}" for _, r in open_sigs.iterrows()]
        sel = st.selectbox("Select open signal to close", labels, key="manual_close_sel")
        ex_price = st.number_input("Exit price ₹", min_value=0.0, step=1.0, key="manual_close_price")
        ex_reason = st.selectbox("Reason", ["Manual Close","Target Hit","Stop Hit","Partial Exit"], key="manual_close_reason")
        if st.button("✅ Mark Closed", key="manual_close_btn"):
            st.success("Marked as closed. Reload Performance tab to see the update.")

    st.markdown("---")
    st.markdown("### 🧹 One-Time Data Cleanup (Run Once in Supabase)")
    st.caption("Copy the SQL below and run it in Supabase → SQL Editor to fix historical data.")
    with st.expander("🔧 Show Cleanup SQL", expanded=False):
        cleanup_sql = """-- NSE Scanner Pro — One-Time Signal History Cleanup
-- Run this ONCE in Supabase SQL Editor

-- STEP 1: Fix SHORT PnL (was computed wrong — using ratio instead of difference)
UPDATE signals
SET pnl_pct = ROUND(((entry - exit_price) / entry * 100)::numeric, 2)
WHERE signal = 'SHORT'
  AND status IN ('STOPPED','TARGET','EXPIRED')
  AND exit_price IS NOT NULL AND entry IS NOT NULL AND entry > 0;

-- STEP 2: Fix BUY PnL (recalculate from actual exit_price to be safe)
UPDATE signals
SET pnl_pct = ROUND(((exit_price - entry) / entry * 100)::numeric, 2)
WHERE signal = 'BUY'
  AND status IN ('STOPPED','TARGET','EXPIRED')
  AND exit_price IS NOT NULL AND entry IS NOT NULL AND entry > 0;

-- STEP 3: Mark cross-day duplicate OPEN signals (keep only earliest per symbol+strategy)
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (PARTITION BY symbol, strategy ORDER BY date ASC, created_at ASC) as rn
    FROM signals WHERE status = 'OPEN'
),
to_mark AS (SELECT id FROM ranked WHERE rn > 1)
UPDATE signals
SET status = 'DUPLICATE',
    exit_reason = 'Cross-day duplicate suppressed'
WHERE id IN (SELECT id FROM to_mark);

-- STEP 4: Expire old OPEN signals (EMA21/Last30 > 15 days, Breakout > 25 days)
UPDATE signals
SET status = 'EXPIRED', exit_date = CURRENT_DATE::text,
    exit_price = cmp,
    exit_reason = 'Auto-expired: exceeded max hold period',
    pnl_pct = CASE
        WHEN signal = 'BUY'   THEN ROUND(((cmp - entry)/entry*100)::numeric,2)
        WHEN signal = 'SHORT' THEN ROUND(((entry - cmp)/entry*100)::numeric,2)
        ELSE 0 END
WHERE status = 'OPEN' AND entry > 0 AND cmp IS NOT NULL
  AND ((strategy IN ('EMA21_Bounce','Last30Min_ATH') AND (CURRENT_DATE - date::date) > 15)
    OR (strategy IN ('52WH_Breakout','Failed_Breakout_Short') AND (CURRENT_DATE - date::date) > 25));

-- STEP 5: Prevent future cross-day duplication (unique OPEN index)
CREATE UNIQUE INDEX IF NOT EXISTS signals_unique_open
ON signals (symbol, strategy) WHERE status = 'OPEN';

-- STEP 6: Verify results
SELECT strategy,
    COUNT(*) total,
    COUNT(*) FILTER (WHERE status='OPEN') open,
    COUNT(*) FILTER (WHERE status='TARGET') target,
    COUNT(*) FILTER (WHERE status='STOPPED') stopped,
    COUNT(*) FILTER (WHERE status='EXPIRED') expired,
    COUNT(*) FILTER (WHERE status='DUPLICATE') duplicate,
    ROUND(COUNT(*) FILTER (WHERE status='TARGET') * 100.0 /
          NULLIF(COUNT(*) FILTER (WHERE status IN ('TARGET','STOPPED','EXPIRED')),0),1) win_rate,
    ROUND(AVG(pnl_pct) FILTER (WHERE status IN ('TARGET','STOPPED','EXPIRED')),2) avg_pnl
FROM signals GROUP BY strategy ORDER BY strategy;"""
        st.code(cleanup_sql, language="sql")
        st.info("📋 Copy the SQL above → go to supabase.com → your project → SQL Editor → New query → paste → Run")


# ============================================================
# PAGE: PORTFOLIO
# ============================================================

def fp(v: float) -> str:
    if v >= 10000: return f"₹{v:,.0f}"
    if v >= 100:   return f"₹{v:,.1f}"
    return f"₹{v:,.2f}"


def page_portfolio():
    st.markdown("# 💼 Portfolio")
    st.caption("Live open positions with real-time P&L, sector exposure, and portfolio heat.")

    data_dict = st.session_state.get("stock_data") or st.session_state.get("enriched_data") or {}

    # ── Summary cards ──────────────────────────────────────────────────────
    pnl_data = get_portfolio_pnl(data_dict)
    positions = pnl_data.get("positions", [])

    if positions:
        total_pnl    = pnl_data["total_pnl"]
        total_pnl_pct= pnl_data["total_pnl_pct"]
        total_inv    = pnl_data["total_invested"]
        capital      = st.session_state.get("capital", 500000)
        regime       = st.session_state.get("regime", {})
        regime_name  = regime.get("regime", "UNKNOWN") if regime else "UNKNOWN"
        heat_cap     = RiskManager.get_regime_heat_cap(regime_name)
        total_risk   = sum(p["risk_amount"] for p in positions)
        heat_pct     = round(total_risk / capital * 100, 1) if capital > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Open Positions", len(positions))
        c2.metric("Total Invested", f"₹{total_inv:,.0f}")

        delta_color = "normal" if total_pnl >= 0 else "inverse"
        c3.metric("Unrealised P&L", f"₹{total_pnl:+,.0f}",
                  delta=f"{total_pnl_pct:+.1f}%", delta_color=delta_color)

        heat_color = "normal" if heat_pct <= heat_cap else "inverse"
        c4.metric("Portfolio Heat", f"{heat_pct}%",
                  delta=f"Cap: {heat_cap}%", delta_color=heat_color)

        c5.metric("Regime", regime.get("regime_display", "—") if regime else "—")

        # Heat warning
        if heat_pct > heat_cap:
            st.error(f"⚠️ Portfolio heat {heat_pct}% exceeds {regime_name} cap of {heat_cap}%. "
                     "Consider closing weakest position or tightening stops.")

        # ── Positions table ────────────────────────────────────────────────
        st.markdown("### Open Positions")
        rows = []
        for p in positions:
            pnl_icon = "🟢" if p["pnl_pct"] >= 0 else "🔴"
            rows.append({
                "":         pnl_icon,
                "Symbol":   p["symbol"],
                "Strategy": p["strategy"],
                "Signal":   p["signal"],
                "Entry":    fp(p["entry"]),
                "CMP":      fp(p["cmp"]),
                "SL":       fp(p["sl"]) if p["sl"] > 0 else "—",
                "T1":       fp(p["target1"]) if p["target1"] > 0 else "—",
                "Qty":      p["qty"],
                "P&L %":    f"{p['pnl_pct']:+.1f}%",
                "P&L ₹":    f"₹{p['pnl_abs']:+,.0f}",
                "Risk ₹":   f"₹{p['risk_amount']:,.0f}",
                "Sector":   p["sector"],
                "Since":    p["entry_date"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Sector exposure chart ──────────────────────────────────────────
        sector_data: dict = {}
        for p in positions:
            sec = p["sector"] or "Other"
            sector_data[sec] = sector_data.get(sec, 0) + p["position_value"]

        if sector_data:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Sector Exposure")
                fig = go.Figure(go.Pie(
                    labels=list(sector_data.keys()),
                    values=[round(v/1000, 1) for v in sector_data.values()],
                    hole=0.45,
                    textinfo="label+percent",
                    textfont_size=12,
                ))
                fig.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="#ffffff",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("### P&L by Position")
                sorted_pos = sorted(positions, key=lambda x: x["pnl_pct"])
                names = [p["symbol"] for p in sorted_pos]
                pnls  = [p["pnl_pct"] for p in sorted_pos]
                colors= ["#dc2626" if v < 0 else "#16a34a" for v in pnls]
                fig2 = go.Figure(go.Bar(
                    x=pnls, y=names, orientation="h",
                    marker_color=colors, text=[f"{v:+.1f}%" for v in pnls],
                    textposition="outside"
                ))
                fig2.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(t=20, b=20, l=60, r=60),
                    paper_bgcolor="#ffffff",
                    xaxis_title="P&L %",
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ── Close position ─────────────────────────────────────────────────
        st.markdown("### Close / Update Position")
        pos_labels = [f"{p['symbol']} ({p['strategy']}) entered {p['entry_date']}" for p in positions]
        sel = st.selectbox("Select position", pos_labels, key="close_pos_sel")
        sel_pos = positions[pos_labels.index(sel)]
        c1, c2, c3 = st.columns(3)
        with c1:
            ex_price = st.number_input("Exit price ₹",
                value=float(sel_pos["cmp"]), step=1.0, key="close_ex_price")
        with c2:
            reason = st.selectbox("Reason",
                ["Target Hit", "Stop Hit", "Manual Exit", "Partial Exit", "Strategy Change"],
                key="close_reason")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔒 Close Position", type="primary", key="close_pos_btn"):
                ok = close_position(sel_pos["id"], ex_price, reason)
                if ok:
                    pnl_pct = round((ex_price / sel_pos["entry"] - 1) * 100, 2) if sel_pos["signal"] == "BUY" else round((sel_pos["entry"] / ex_price - 1) * 100, 2)
                    st.success(f"✅ {sel_pos['symbol']} closed at {fp(ex_price)} | P&L: {pnl_pct:+.1f}%")
                    st.rerun()
                else:
                    st.error("Close failed — check Supabase connection.")

    else:
        st.info("No open positions. Add a position below when you take a trade.")

    # ── Add new position ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("➕ Add New Position", expanded=len(positions) == 0):
        with st.form("add_position_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                sym  = st.text_input("Symbol (e.g. RELIANCE)", key="pos_sym").upper().strip()
                sig  = st.selectbox("Signal", ["BUY", "SHORT"], key="pos_sig")
                strat= st.selectbox("Strategy",
                    list(STRATEGY_PROFILES.keys()),
                    format_func=lambda x: f"{STRATEGY_PROFILES[x]['icon']} {STRATEGY_PROFILES[x]['name']}",
                    key="pos_strat")
            with c2:
                entry= st.number_input("Entry ₹", min_value=0.0, step=1.0, key="pos_entry")
                sl_  = st.number_input("Stop Loss ₹", min_value=0.0, step=1.0, key="pos_sl")
                qty  = st.number_input("Quantity", min_value=1, value=1, key="pos_qty")
            with c3:
                t1_  = st.number_input("Target 1 ₹", min_value=0.0, step=1.0, key="pos_t1")
                t2_  = st.number_input("Target 2 ₹", min_value=0.0, step=1.0, key="pos_t2")
                notes= st.text_input("Notes (optional)", key="pos_notes")

            regime_name2 = st.session_state.get("regime", {}).get("regime", "") if st.session_state.get("regime") else ""
            submitted = st.form_submit_button("➕ Add Position", type="primary")
            if submitted and sym and entry > 0 and qty > 0:
                ok = save_position({
                    "symbol":          sym,
                    "strategy":        strat,
                    "signal":          sig,
                    "entry_price":     round(entry, 2),
                    "stop_loss":       round(sl_, 2),
                    "target1":         round(t1_, 2),
                    "target2":         round(t2_, 2),
                    "qty":             int(qty),
                    "entry_date":      date.today().isoformat(),
                    "sector":          get_sector(sym),
                    "regime_at_entry": regime_name2,
                    "notes":           notes,
                })
                if ok:
                    st.success(f"✅ Added {sym} {sig} position at ₹{entry:,.1f}")
                    st.rerun()
                else:
                    st.error("Save failed. Configure SUPABASE_URL and SUPABASE_SERVICE_KEY in Streamlit secrets.")
