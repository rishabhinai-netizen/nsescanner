"""Central config — single source for secrets. st.secrets → env fallback.
Never hardcode keys. Works identically in Streamlit Cloud and GitHub Actions."""
import os

def secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)

SUPABASE_URL         = lambda: secret("SUPABASE_URL")
SUPABASE_ANON_KEY    = lambda: secret("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = lambda: secret("SUPABASE_SERVICE_KEY")
TELEGRAM_BOT_TOKEN   = lambda: secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID     = lambda: secret("TELEGRAM_CHAT_ID")
BREEZE_API_KEY       = lambda: secret("BREEZE_API_KEY")
BREEZE_API_SECRET    = lambda: secret("BREEZE_API_SECRET")
ANTHROPIC_API_KEY    = lambda: secret("ANTHROPIC_API_KEY")

IST = "Asia/Kolkata"
APP_NAME = "NSE Scanner NX"
APP_VERSION = "2.0.0"
