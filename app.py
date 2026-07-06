"""NSE Scanner NX — thin router. All page logic lives in ui/.
Design: T09 Charcoal & Ivory (ivory #F9F8F6, charcoal #1A1A18 / #444441)."""
import streamlit as st

st.set_page_config(page_title="NSE Scanner NX", page_icon="◆",
                   layout="wide", initial_sidebar_state="expanded")

# ── T09 Charcoal & Ivory design system ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F9F8F6; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #1A1A18 !important; letter-spacing: -0.01em; }
p, span, label, div { color: #444441; }

[data-testid="stSidebar"] { background: #1A1A18; }
[data-testid="stSidebar"] * { color: #F9F8F6 !important; }
[data-testid="stSidebar"] .stRadio label:hover { opacity: .8; }

.nx-card { background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 14px;
  padding: 1.1rem 1.3rem; box-shadow: 0 1px 3px rgba(26,26,24,.05); }
.nx-metric { font-family: 'DM Serif Display', serif; font-size: 1.9rem; color: #1A1A18; }
.nx-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #8A8781; }

.nx-badge { display:inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: .72rem; font-weight: 600; letter-spacing:.03em; }
.nx-live { background:#E7F2EA; color:#1F6B3A; }
.nx-inc  { background:#FDF3E3; color:#8A5A00; }
.nx-a    { background:#1A1A18; color:#F9F8F6; }
.nx-b    { background:#E8E6E1; color:#444441; }

.nx-EXPANSION    { background:#E7F2EA; color:#1F6B3A; }
.nx-ACCUMULATION { background:#EAF1F8; color:#0C447C; }
.nx-DISTRIBUTION { background:#FDF3E3; color:#8A5A00; }
.nx-PANIC        { background:#FBEAEA; color:#9B2C2C; }

div[data-testid="stDataFrame"] { border: 1px solid #E8E6E1; border-radius: 12px; }
.stButton>button { background:#1A1A18; color:#F9F8F6; border:none; border-radius:10px;
  font-weight:600; } .stButton>button:hover { background:#444441; color:#F9F8F6; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Navigation ───────────────────────────────────────────────────────────────
from ui import dashboard, tracker, settings

PAGES = {"◆ Dashboard": dashboard.render,
         "▤ Tracker": tracker.render,
         "⚙ Settings": settings.render}

with st.sidebar:
    st.markdown("## NSE Scanner **NX**")
    st.caption("Evidence over opinion · v2.0")
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("One live strategy that works beats\neight that don't. — the v2 thesis")

PAGES[page]()
