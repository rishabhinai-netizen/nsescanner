"""Settings — Breeze daily token refresh (→ nx_app_config), connection checks.
No secrets are ever displayed back; write-only fields."""
import streamlit as st

from core.db import set_config, get_config, client
from core.config import (SUPABASE_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                         BREEZE_API_KEY)
from core.alerts import send_telegram


def render():
    st.title("Settings")

    st.markdown("### Breeze session token")
    st.caption("Expires daily. Grab it each morning from the ICICI Direct API portal. "
               "Stored in nx_app_config (admin-only RLS) — never in git, never displayed.")
    tok = st.text_input("Paste today's session token", type="password",
                        placeholder="••••••••")
    if st.button("Save token") and tok:
        st.success("Token saved.") if set_config("breeze_session_token", tok.strip()) \
            else st.error("Save failed — check Supabase service key.")
    have = get_config("breeze_session_token")
    st.caption(f"Token on file: {'yes' if have else 'no'}")

    st.divider()
    st.markdown("### Connection checks")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Test Supabase"):
            try:
                client(True).table("nx_app_config").select("key").limit(1).execute()
                st.success("Supabase OK")
            except Exception as e:
                st.error(f"Supabase: {e}")
    with c2:
        if st.button("Test Telegram"):
            st.success("Sent") if send_telegram("✅ NX connection test") \
                else st.error("Telegram failed — check bot token / chat id")
    with c3:
        st.write("Breeze key set:", "✅" if BREEZE_API_KEY() else "❌")
        st.write("Supabase URL set:", "✅" if SUPABASE_URL() else "❌")
        st.write("Telegram set:", "✅" if TELEGRAM_BOT_TOKEN() and TELEGRAM_CHAT_ID() else "❌")

    st.divider()
    st.markdown("### Gate policy (read-only)")
    st.code("""Promotion:  INCUBATING → LIVE  when PF ≥ 1.2 on n ≥ 30 closed (per regime)
Demotion:   LIVE → INCUBATING when PF < 0.8 on n ≥ 30 closed
Hard block: LONG in DISTRIBUTION unless RS ≥ 90
Alert cap:  top 8 by SQI per scan — trust > volume""", language="text")
