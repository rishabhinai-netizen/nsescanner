"""Dashboard — today's regime, LIVE signals, incubator progress."""
import pandas as pd
import streamlit as st

from core.db import load_signals, load_live_stats


def _card(label: str, value: str, sub: str = ""):
    st.markdown(f"""<div class="nx-card">
      <div class="nx-label">{label}</div>
      <div class="nx-metric">{value}</div>
      <div style="font-size:.8rem;color:#8A8781">{sub}</div></div>""",
      unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def _signals(days: int) -> pd.DataFrame:
    try:
        return load_signals(days=days, service=True)
    except Exception:
        return pd.DataFrame()


def render():
    st.title("Dashboard")
    df = _signals(30)
    if df.empty:
        st.info("No signals yet. Run the scan job or wait for the 7 PM IST GitHub Action.")
        return

    latest_date = df["signal_date"].max()
    today = df[df["signal_date"] == latest_date]
    regime = today["regime"].iloc[0] if len(today) else "—"
    rscore = today["regime_score"].iloc[0] if len(today) else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="nx-card"><div class="nx-label">Market Regime</div>
          <span class="nx-badge nx-{regime}" style="font-size:1.05rem;padding:6px 16px">{regime}</span>
          <div style="font-size:.8rem;color:#8A8781;margin-top:6px">score {rscore} · {latest_date}</div></div>""",
          unsafe_allow_html=True)
    live = today[today["gate"] == "LIVE"]
    inc = today[today["gate"] == "INCUBATING"]
    with c2: _card("Live signals today", str(len(live)), "alerted via Telegram")
    with c3: _card("Incubating", str(len(inc)), "tracked, not alerted")
    with c4:
        a = len(today[today["sqi_tier"] == "A"])
        _card("Tier A setups", str(a), "SQI ≥ 75")

    st.markdown("### Live signals")
    if live.empty:
        st.caption("No LIVE signals in the latest scan — the gate held. That is the system working.")
    else:
        show = live.sort_values("sqi", ascending=False)[
            ["symbol", "strategy", "side", "entry", "stop", "target1", "target2",
             "rr", "rs_rank", "sqi", "sqi_tier", "sector"]]
        st.dataframe(show, use_container_width=True, hide_index=True)

    with st.expander(f"Incubating signals ({len(inc)}) — building live sample, do not trade"):
        if not inc.empty:
            st.dataframe(inc[["symbol", "strategy", "entry", "stop", "target1",
                              "rs_rank", "sqi", "sqi_tier"]],
                         use_container_width=True, hide_index=True)

    st.markdown("### Strategy × regime — live profit factor")
    stats = load_live_stats()
    if stats:
        sdf = pd.DataFrame(stats.values())[
            ["strategy", "regime", "n_closed", "win_rate", "avg_pnl", "profit_factor"]]
        st.dataframe(sdf.sort_values(["strategy", "regime"]),
                     use_container_width=True, hide_index=True)
        st.caption("Promotion: PF ≥ 1.2 on n ≥ 30. Demotion: PF < 0.8 on n ≥ 30. "
                   "No hardcoded priors — this table IS the gate.")
    else:
        st.caption("Stats build as signals close. The gate uses defaults until n ≥ 30 per cell.")
