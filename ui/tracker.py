"""Tracker — forward-test truth. Win rates, expectancy, equity curve,
expiry health. This page is why v2 can never lie to itself."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.db import load_signals


@st.cache_data(ttl=300, show_spinner=False)
def _closed(days: int) -> pd.DataFrame:
    try:
        df = load_signals(days=days, service=True)
        if df.empty:
            return df
        return df[df["status"].isin(["TARGET", "STOPPED", "EXPIRED"])].copy()
    except Exception:
        return pd.DataFrame()


def render():
    st.title("Tracker")
    days = st.select_slider("Window", [30, 60, 90, 180, 365], value=90)
    df = _closed(days)
    if df.empty:
        st.info("No closed signals in this window yet.")
        return

    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    df = df.dropna(subset=["pnl_pct"])
    df["win"] = df["pnl_pct"] > 0

    wr = df["win"].mean() * 100
    avg = df["pnl_pct"].mean()
    gains = df.loc[df.win, "pnl_pct"].sum()
    losses = abs(df.loc[~df.win, "pnl_pct"].sum())
    pf = gains / losses if losses else float("inf")
    exp_rate = (df["status"] == "EXPIRED").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Closed trades", len(df))
    c2.metric("Win rate", f"{wr:.1f}%")
    c3.metric("Profit factor", f"{pf:.2f}")
    c4.metric("Expiry rate", f"{exp_rate:.0f}%",
              help="v15 ran at 40% — v2 targets <25% via tuned time-stop")

    # Equity curve (1R-normalized cumulative pnl%)
    eq = df.sort_values("exit_date").copy()
    eq["cum"] = eq["pnl_pct"].cumsum()
    fig = go.Figure(go.Scatter(x=eq["exit_date"], y=eq["cum"], mode="lines",
                               line=dict(color="#1A1A18", width=2.2),
                               fill="tozeroy", fillcolor="rgba(26,26,24,.06)"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="#F9F8F6", paper_bgcolor="#F9F8F6",
                      yaxis_title="Cumulative PnL %", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### By strategy × gate")
    g = (df.groupby(["strategy", "gate"])
           .agg(n=("win", "size"), win_rate=("win", "mean"), avg_pnl=("pnl_pct", "mean"))
           .reset_index())
    g["win_rate"] = (g["win_rate"] * 100).round(1)
    g["avg_pnl"] = g["avg_pnl"].round(2)
    st.dataframe(g, use_container_width=True, hide_index=True)

    st.markdown("### SQI calibration check")
    st.caption("If SQI works, Tier A must outperform Tier B. If it doesn't, the gate weights get revisited.")
    t = (df.groupby("sqi_tier")
           .agg(n=("win", "size"), win_rate=("win", "mean"), avg_pnl=("pnl_pct", "mean"))
           .reset_index())
    t["win_rate"] = (t["win_rate"] * 100).round(1)
    t["avg_pnl"] = t["avg_pnl"].round(2)
    st.dataframe(t, use_container_width=True, hide_index=True)
