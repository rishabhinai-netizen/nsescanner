"""
NSE SCANNER PRO v15 — Complete Professional Trading Intelligence Suite
=======================================================================
v15 merges the best of all previous versions:

CORE ENGINE (from v5.2):
- Signal Quality Index (SQI) — multi-factor scoring (Backtest Edge 30%,
  RS Acceleration 25%, Regime Fit 20%, Vol Contraction 15%, Volume 10%)
- Fundamental Quality Gate — CANSLIM-inspired (EPS, revenue, PE, D/E)
- Strategy×Regime Profit Factor Matrix — auto-blocks PF < 0.7 combos
- Regime-Adaptive heat caps (EXPANSION 6%, PANIC 1%)
- Approaching Setup Watchlist — stocks 50-95% through setups
- Broker Basket Export — Zerodha Kite CSV + generic broker CSV
- Strategy Health Tracker — auto-dim failing strategies
- VCP rewrite with std_10/std_50 < 0.50 volatility compression ratio

RESTORED (from v11):
- RRG Sector Rotation — 4-quadrant panel, +8 LEADING / -15 LAGGING
- Net P&L / CostModel — slippage, STT, brokerage, GST in backtests
- Volume Profile (POC, VAH/VAL, HVN/LVN) via Breeze API
- Stock Lookup page — deep-dive indicator + signal history per stock
- Signal History page — first-seen table, entry drift detection
- Walk-Forward Validation — optional toggle in Backtest page
- Breeze Retry button + threading fix (20s timeout, no SIGALRM)

NEW MODULES (v15 original):
- Option Chain Overlay — 7-factor scoring via Breeze API
  (OI Change 25%, UOA 20%, IVP 15%, PCR 15%, Max Pain 10%, Skew 7.5%, IVS 7.5%)
  Academic foundation: Cremers & Weinbaum (2010) IV Spread 50bps/week
- IPO Base Scanner — listing date + subscription auto-scrape
  O'Neil: 57% hit rate, 2.79% alpha on high-volume breakouts ≥150% avg volume
  Lock-up alerts: Day 30/90/180/540 with 5-day buffer warnings
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime, date
import pytz, json, os, logging

from stock_universe import get_stock_universe, get_sector, NIFTY_50
from data_engine import (
    fetch_batch_daily, fetch_nifty_data,
    Indicators, BreezeEngine, now_ist, IST,
    update_breeze_token_in_supabase, _get_breeze_token_from_supabase
)
from scanners import (
    STRATEGY_PROFILES, DAILY_SCANNERS, INTRADAY_SCANNERS,
    run_scanner, run_all_scanners, detect_market_regime, ScanResult,
    collect_approaching_setups, strategy_health, ApproachingSetup,
    compute_sector_rrg,
)
from risk_manager import RiskManager
from enhancements import (
    plot_candlestick, compute_sector_performance, plot_sector_heatmap,
    compute_rs_rankings, plot_rs_scatter,
    load_journal, save_journal, add_journal_entry, compute_journal_analytics, plot_equity_curve,
    check_weekly_alignment, compute_market_breadth, plot_breadth_gauge,
)
from backtester import (
    backtest_strategy, backtest_multi_stock, backtest_walk_forward,
    BacktestResult, CostModel, DEFAULT_COSTS
)
from tooltips import TIPS, tip, tip_md, tip_badge
from signal_tracker import (
    load_signals, load_tracker, save_signals_today,
    compute_tracker_stats, update_open_signals_live,
    generate_csv_download, get_signal_dates,
    save_position, load_portfolio, close_position, get_portfolio_pnl,
    _get_supabase,
)
from fno_list import is_fno, get_fno_tag
from signal_quality import compute_sqi, get_regime_strategy_matrix, STRATEGY_REGIME_PF, reset_pf_cache

# v16 additions
from app_additions import page_performance, render_supabase_status

# v18 — AI Deep Dive (Multi-Agent Swarm Analysis) [v3-exhaustive]

# v19 — Auth manager (multi-user, Google OAuth, Telegram/email alerts)
try:
    from auth_manager import (
        require_auth, get_current_user, get_current_profile,
        is_admin, logout, render_alert_preferences, render_admin_dashboard
    )
    AUTH_AVAILABLE = True
except Exception as _auth_e:
    AUTH_AVAILABLE = False
    print(f"[Auth] auth_manager not loaded: {_auth_e}")
    def require_auth(): return True
    def get_current_user(): return None
    def get_current_profile(): return None
    def is_admin(): return False
    def logout(): pass
    def render_alert_preferences(): st.info("Auth not configured.")
    def render_admin_dashboard(): st.info("Auth not configured.")
_AI_DEEP_DIVE_ERROR = ""
try:
    from ai_deep_dive import page_ai_deep_dive
    AI_DEEP_DIVE_AVAILABLE = True
    print("[AI Deep Dive] ✅ Loaded successfully")
except Exception as _e:
    import traceback
    AI_DEEP_DIVE_AVAILABLE = False
    _AI_DEEP_DIVE_ERROR = str(_e)
    print(f"[AI Deep Dive] ❌ Import failed: {_e}")
    print(traceback.format_exc())

# v17 additions
try:
    from paper_trading import render_paper_trading_page, enter_paper_trade
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False

try:
    from perplexity_enrichment import render_signal_context_card
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
from fundamental_gate import check_fundamental_quality, batch_fundamental_check
from basket_export import generate_zerodha_basket, generate_generic_basket

# New v15 modules
try:
    from option_chain import analyze_option_chain, OptionChainResult, format_oc_signal
    OPTION_CHAIN_AVAILABLE = True
except ImportError:
    OPTION_CHAIN_AVAILABLE = False

try:
    from ipo_scanner import (
        scan_ipo_universe, IPOResult, get_upcoming_lock_up_alerts,
        fetch_nse_ipo_list, format_ipo_alert
    )
    IPO_SCANNER_AVAILABLE = True
except ImportError:
    IPO_SCANNER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NSE Scanner Pro", page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# ============================================================================
# AUTH GATE — Multi-user authentication (v19)
# ============================================================================
if not require_auth():
    st.stop()  # Block rest of app until user is logged in

# ============================================================================
# PASSWORD GATE (optional — set APP_PASSWORD in Streamlit secrets to enable)
# ============================================================================
try:
    _app_pw = st.secrets.get("APP_PASSWORD", "")
except Exception:
    _app_pw = ""
if _app_pw and _app_pw != "your_password":
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.markdown("## 🎯 NSE Scanner Pro")
        st.markdown("---")
        pw = st.text_input("Enter access password", type="password", key="pw_input")
        if st.button("Login", key="pw_btn"):
            if pw == _app_pw:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.stop()
else:
    st.session_state.authenticated = True

# ============================================================================
# CSS — Mobile-friendly, overflow-safe, branding hidden
# ============================================================================
st.markdown("""
<style>
    /* === NUCLEAR BRANDING HIDE — covers all Streamlit versions === */
    #MainMenu, footer, header, 
    .stDeployButton,
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stHeader"],
    [data-testid="manage-app-button"],
    #stDecoration,
    /* Class-based (varies by Streamlit version) */
    .viewerBadge_container__r5tak, .styles_viewerBadge__CvC9N,
    ._profileContainer_51w34_53, ._profilePreview_51w34_63,
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_link__qRIco,
    .styles_viewerBadge__1yB5_, .viewerBadge_text__1JaDK,
    /* Catch-all for ANY link to github or streamlit */
    a[href*="github.com"],
    a[href*="streamlit.io"],
    a[href*="github"],
    /* Footer and deploy elements */
    .stApp > footer,
    div[class*="stDeployButton"],
    div[class*="viewerBadge"],
    div[class*="StatusWidget"],
    button[kind="header"],
    /* Mobile-specific toolbar */
    section[data-testid="stSidebar"] a[href*="github"],
    .stApp header,
    .stApp [data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        top: -9999px !important;
    }
    
    /* LOCK sidebar open — hide the collapse arrow button */
    [data-testid="collapsedControl"],
    button[kind="header"],
    [data-testid="stSidebarCollapseButton"],
    .st-emotion-cache-1ibsh2c,
    section[data-testid="stSidebar"] button[aria-label="Close sidebar"],
    section[data-testid="stSidebar"] > div > div > button { 
        display: none !important; 
    }
    /* Ensure sidebar is always visible */
    section[data-testid="stSidebar"] {
        min-width: 260px !important;
        transform: none !important;
    }
    /* Also hide the top padding that header leaves behind */
    .stApp > header + div { padding-top: 0 !important; }
    .block-container { padding-top: 1rem !important; }
    
    /* Metric boxes: compact, no overflow */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333; border-radius: 8px; padding: 8px 10px; overflow: hidden;
    }
    div[data-testid="stMetric"] label { font-size: 0.7rem !important; color: #888 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1rem !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    
    /* Custom cards */
    .pc { background: linear-gradient(135deg,#1a1d23,#252830); border:1px solid #333;
          border-radius:8px; padding:8px 12px; margin:3px 0; }
    .pc .lb { font-size:0.68rem; color:#888; margin-bottom:1px; }
    .pc .vl { font-size:0.95rem; font-weight:600; color:#fafafa; }
    .pc .vl.g { color:#00d26a; } .pc .vl.r { color:#ff4757; } .pc .vl.o { color:#FF6B35; }
    .pc .vl.y { color:#ffd700; }
    
    /* Focus panel */
    .focus { background:linear-gradient(135deg,#0d1117,#161b22); border:1px solid #FF6B35;
             border-radius:12px; padding:16px 20px; margin:10px 0; }
    .focus h3 { margin:0 0 8px 0; color:#FF6B35; font-size:1.1rem; }
    .focus .regime { font-size:1.3rem; font-weight:700; margin:4px 0; }
    .focus .tip { color:#aaa; font-size:0.82rem; margin-top:6px; }
    
    /* Strategy cards */
    .sc { background:#1a1d23; border:1px solid #333; border-radius:10px; padding:12px; margin:5px 0; }
    .sc:hover { border-color:#FF6B35; }
    .sc.blocked { opacity:0.4; border-color:#ff4757; }
    .sc.ideal { border-left:3px solid #00d26a; }
    .sc.caution { border-left:3px solid #ffd700; }
    
    /* Badges */
    .bg { display:inline-block; padding:2px 7px; border-radius:10px; font-size:0.65em; }
    .bg-s { background:#1e3a5f; color:#5dade2; } .bg-i { background:#3e2723; color:#ff8a65; }
    .bg-p { background:#1b5e20; color:#81c784; } .bg-o { background:#4a148c; color:#ce93d8; }
    .bg-blocked { background:#3d1a1a; color:#ff4757; }
    .bg-ideal { background:#0d3320; color:#00d26a; }
    .bg-caution { background:#3d3a1a; color:#ffd700; }
    
    /* SQI badges */
    .sqi-elite { background:#0d3320; color:#00d26a; font-weight:700; }
    .sqi-strong { background:#1e3a5f; color:#5dade2; font-weight:600; }
    .sqi-moderate { background:#3d3a1a; color:#ffd700; }
    .sqi-weak { background:#3d1a1a; color:#ff4757; }
    
    /* Stale data warning */
    .stale { background:#3d2a1a; border:1px solid #ff8a65; border-radius:8px;
             padding:8px 14px; color:#ff8a65; font-size:0.85rem; }
    
    /* Workflow */
    .ws { border-left:3px solid #FF6B35; padding:7px 12px; margin:5px 0;
          background:#1a1d23; border-radius:0 8px 8px 0; }
    
    /* Breeze banner */
    .bb { padding:6px 12px; border-radius:8px; margin:6px 0; font-size:0.8rem; }
    .bb-on { background:#0d3320; border:1px solid #1b5e20; color:#81c784; }
    .bb-off { background:#1a1d23; border:1px solid #333; color:#888; }
    
    .dataframe { font-size:0.78rem !important; }
    
    /* Mobile: ensure sidebar works */
    @media (max-width: 768px) {
        .pc .vl { font-size:0.85rem; }
        .pc .lb { font-size:0.62rem; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size:0.85rem !important; }
        .focus { padding:12px 14px; }
        .sc { padding:10px; }
        /* Extra mobile branding kill */
        header[data-testid="stHeader"] { display:none !important; }
        .stApp > header { display:none !important; }
    }
</style>
""", unsafe_allow_html=True)

# PWA — Add to Home Screen support for mobile
st.markdown("""
<link rel="manifest" href="data:application/json;base64,eyJuYW1lIjogIk5TRSBTY2FubmVyIFBybyIsICJzaG9ydF9uYW1lIjogIlNjYW5uZXIiLCAic3RhcnRfdXJsIjogIi4iLCAiZGlzcGxheSI6ICJzdGFuZGFsb25lIiwgImJhY2tncm91bmRfY29sb3IiOiAiIzBFMTExNyIsICJ0aGVtZV9jb2xvciI6ICIjRkY2QjAwIiwgIm9yaWVudGF0aW9uIjogInBvcnRyYWl0IiwgImljb25zIjogW3sic3JjIjogImh0dHBzOi8vZW0tY29udGVudC56b2JqLm5ldC9zb3VyY2UvYXBwbGUvMzU0L2RpcmVjdC1oaXRfMWYzYWYucG5nIiwgInNpemVzIjogIjEyMHgxMjAiLCAidHlwZSI6ICJpbWFnZS9wbmcifV19" />
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="NSE Scanner">
<meta name="theme-color" content="#FF6B00">
""", unsafe_allow_html=True)
# ============================================================================
for k, v in {
    "watchlist":[], "scan_results":{}, "regime":None,
    "data_loaded":False, "stock_data":{}, "enriched_data":{},
    "nifty_data":None, "capital":500000,
    "breeze_connected":False, "breeze_engine":None, "breeze_msg":"",
    "workflow_checks":{}, "universe_size":"nifty200",
    "telegram_token":"", "telegram_chat_id":"",
    "journal":None, "last_scan_time":None, "sector_rankings":{},
    "rrg_data": {},  # v15: RRG sector rotation data
    "rs_filter": 70, "regime_filter": True,
    "fundamental_filter": False, "fundamental_cache": {},
    "approaching_setups": [],
    "oc_results": {},    # v15: Option Chain results cache
    "ipo_results": [],   # v15: IPO Scanner results
}.items():
    if k not in st.session_state: st.session_state[k] = v

if st.session_state.journal is None:
    st.session_state.journal = load_journal()

# Breeze auto-connect — reads all credentials in MAIN thread, connects in worker thread
def try_breeze():
    """
    Auto-connect Breeze.
    KEY FIX: st.secrets and Supabase are accessed in the MAIN thread before
    the worker thread starts. This avoids silent failures with st.secrets and
    supabase client inside threading.Thread contexts.

    Session token priority: Supabase app_config → Streamlit secrets.
    """
    if st.session_state.breeze_connected: return
    try:
        # ── Step 1: Read ALL credentials in the MAIN thread ──────────────
        api_key    = ""
        api_secret = ""
        try:
            api_key    = st.secrets.get("BREEZE_API_KEY", "")
            api_secret = st.secrets.get("BREEZE_API_SECRET", "")
        except Exception:
            pass

        if not api_key or "your_" in api_key:
            st.session_state.breeze_msg = "BREEZE_API_KEY not set in Streamlit secrets."
            return
        if not api_secret:
            st.session_state.breeze_msg = "BREEZE_API_SECRET not set in Streamlit secrets."
            return

        # ── Step 2: Get session token (main thread — Supabase first) ─────
        session_token = _get_breeze_token_from_supabase()   # reads Supabase in main thread
        if not session_token:
            try:
                session_token = st.secrets.get("BREEZE_SESSION_TOKEN", "")
            except Exception:
                pass
        if not session_token:
            st.session_state.breeze_msg = (
                "Breeze session token not found. "
                "Paste today's token in Settings → Update Session Token."
            )
            return

        # ── Step 3: Connect in worker thread (only network I/O now) ──────
        import threading
        result = {"ok": False, "msg": "Breeze connection timed out (20s)"}
        def _connect():
            try:
                e = BreezeEngine()
                ok, msg = e.connect(api_key, api_secret, session_token)
                result["ok"] = ok
                result["msg"] = msg
                if ok:
                    result["engine"] = e
            except Exception as ex:
                result["msg"] = f"Breeze: {type(ex).__name__}: {str(ex)[:80]}"
        t = threading.Thread(target=_connect, daemon=True)
        t.start()
        t.join(timeout=20)
        st.session_state.breeze_connected = result.get("ok", False)
        st.session_state.breeze_msg = result.get("msg", "")
        if result.get("ok"):
            st.session_state.breeze_engine = result.get("engine")
    except Exception as ex:
        st.session_state.breeze_msg = f"Breeze init: {type(ex).__name__}: {str(ex)[:80]}"
try_breeze()

# ============================================================================
# HELPERS
# ============================================================================
def pc(label, value, css=""):
    if isinstance(value,(int,float)):
        v = f"₹{value:,.0f}" if value>=10000 else (f"₹{value:,.1f}" if value>=100 else f"₹{value:,.2f}")
    else: v = str(value)
    st.markdown(f'<div class="pc"><div class="lb">{label}</div><div class="vl {css}">{v}</div></div>', unsafe_allow_html=True)

def fp(v):
    if v>=10000: return f"₹{v:,.0f}"
    elif v>=100: return f"₹{v:,.1f}"
    else: return f"₹{v:,.2f}"

def send_tg(msg):
    try:
        tk = st.session_state.telegram_token or st.secrets.get("TELEGRAM_BOT_TOKEN","")
        ci = st.session_state.telegram_chat_id or st.secrets.get("TELEGRAM_CHAT_ID","")
    except Exception:
        tk, ci = st.session_state.get("telegram_token",""), st.session_state.get("telegram_chat_id","")
    if not tk or not ci: return False
    try:
        import requests
        return requests.post(f"https://api.telegram.org/bot{tk}/sendMessage",
                             json={"chat_id":ci,"text":msg,"parse_mode":"HTML"},timeout=10).status_code == 200
    except: return False

def fmt_alert(r, is_confluence=False):
    fno = "F&O ✓" if is_fno(r.symbol) else "Cash"
    sec_perf = _get_sector_perf_str(r.sector)
    conf_tag = "🔥 CONFLUENCE — " if is_confluence else ""
    sqi_val = getattr(r, 'sqi', None)
    sqi_str = f"SQI: {sqi_val:.0f} ({getattr(r, 'sqi_grade', '')})" if sqi_val else f"Conf: {r.confidence}%"
    return (f"{conf_tag}🎯 <b>{r.strategy}</b> — {r.signal}\n"
            f"📈 <b>{r.symbol}</b> ({r.sector}) [{fno}]\n"
            f"💰 CMP: {fp(r.cmp)} | Entry: {fp(r.entry)}\n"
            f"🛑 SL: {fp(r.stop_loss)} | T1: {fp(r.target_1)} | R:R 1:{r.risk_reward:.1f}\n"
            f"📊 {sqi_str} | RS: {int(r.rs_rating)} | {sec_perf}\n"
            f"🏷️ Regime: {r.regime_fit}")

def _get_sector_perf_str(sector):
    """Get sector performance string for display."""
    ranks = st.session_state.get("sector_rankings", {})
    if sector and sector in ranks:
        rank_pct = ranks[sector]
        return f"Sector: #{int(100-rank_pct)+1}"
    return "Sector: -"

def results_df(results):
    rows = []
    for r in results:
        fno = "F&O" if is_fno(r.symbol) else "Cash"
        sqi_val = getattr(r, 'sqi', None)
        sqi_grade = getattr(r, 'sqi_grade', '')
        sqi_icon = getattr(r, 'sqi_icon', '')
        row = {
            "Symbol": r.symbol, "Signal": r.signal, "CMP": fp(r.cmp), "Entry": fp(r.entry),
            "SL": fp(r.stop_loss), "T1": fp(r.target_1), "R:R": f"1:{r.risk_reward:.1f}",
            "RS": int(r.rs_rating), "Mkt": fno,
            "Regime": r.regime_fit, "Sector": r.sector, "Hold": r.hold_type,
        }
        if sqi_val is not None:
            row["SQI"] = f"{sqi_icon} {sqi_val:.0f}"
            row["Grade"] = sqi_grade
        else:
            row["Conf"] = f"{r.confidence}%"
        rows.append(row)
    return pd.DataFrame(rows)


def detect_confluence(scan_results: dict) -> dict:
    """Find stocks appearing in 2+ strategies. Returns {symbol: [strategies]}."""
    symbol_strategies = {}
    for strategy, signals in scan_results.items():
        for r in signals:
            if r.symbol not in symbol_strategies:
                symbol_strategies[r.symbol] = []
            symbol_strategies[r.symbol].append(strategy)
    return {sym: strats for sym, strats in symbol_strategies.items() if len(strats) >= 2}


def send_scan_alerts(scan_results: dict):
    """Send telegram alerts with confluence awareness."""
    confluence = detect_confluence(scan_results)
    confluence_symbols = set(confluence.keys())
    
    # Send confluence alerts first (higher priority)
    if confluence:
        msg = "🔥🔥 <b>CONFLUENCE ALERT</b> 🔥🔥\n"
        for sym, strats in confluence.items():
            strat_names = [STRATEGY_PROFILES.get(s, {}).get("name", s) for s in strats]
            fno = "F&O ✓" if is_fno(sym) else "Cash"
            msg += f"\n📈 <b>{sym}</b> [{fno}] — {len(strats)} strategies:\n"
            msg += ", ".join(strat_names) + "\n"
        send_tg(msg)
    
    # Then individual signals
    for strat, signals in scan_results.items():
        for r in signals:
            is_conf = r.symbol in confluence_symbols
            send_tg(fmt_alert(r, is_confluence=is_conf))

def is_data_stale():
    """Check if scan data is stale (>15 min old)."""
    if not st.session_state.last_scan_time: return True
    age = (now_ist() - st.session_state.last_scan_time).total_seconds() / 60
    return age > 15

def compute_sector_ranks():
    """Build sector ranking dict for filtering."""
    data = st.session_state.enriched_data or st.session_state.stock_data
    if not data: return {}
    perf = compute_sector_performance(data, get_sector)
    if perf.empty: return {}
    perf["rank"] = perf["avg_1m"].rank(pct=True) * 100
    return perf["rank"].to_dict()

def load_data():
    syms = get_stock_universe(st.session_state.universe_size)
    pb = st.progress(0, "Starting...")
    def cb(p, t): pb.progress(min(p, 0.95), t)
    data = fetch_batch_daily(syms, "1y", cb)
    pb.progress(0.96, "Fetching Nifty...")
    nifty = fetch_nifty_data()
    pb.progress(0.97, "Computing indicators...")
    enriched = {}
    for s, df in data.items():
        try: enriched[s] = Indicators.enrich_dataframe(df)
        except: enriched[s] = df
    pb.progress(1.0, f"✅ {len(data)} stocks loaded!")
    return data, nifty, enriched

# ============================================================================
# SIDEBAR — with query_params persistence (fixes page jump on interaction)
# ============================================================================
# Navigation groups — order defines user workflow
# ① Market Intelligence  ② Scan & Find  ③ Analyse  ④ Track  ⑤ System
PAGES = [
    "📊 Dashboard",
    "🧠 AI Deep Dive",
    "🔎 Stock Lookup",
    "── SCAN ──",
    "🔍 Scanner Hub",
    "📜 Signal History",
    "📊 Performance",
    "── ANALYSE ──",
    "📈 Charts & RS",
    "🔗 Option Chain",
    "🚀 IPO Scanner",
    "🧪 Backtest",
    "── TRACK ──",
    "🎮 Virtual Game",
    "── ──",
    "⚙️ Settings",
    "👑 Admin",
]
# Pages that are section dividers (not navigable)
_NAV_DIVIDERS = {"── SCAN ──", "── ANALYSE ──", "── TRACK ──", "── ──"}

# Persist page selection in URL query params
qp = st.query_params
default_page = qp.get("page", PAGES[0])
# default_page validated below after _NAV_DIVIDERS is defined

with st.sidebar:
    st.markdown("## 🎯 NSE Scanner Pro")
    st.caption("v19 — AI · Multi-User · 1000+ Stocks · Smart Nav")
    
    # User identity chip
    _user = get_current_user()
    _profile = get_current_profile()
    if _user:
        _plan = (_profile or {}).get("plan", "free")
        _plan_badge = {"admin": "👑 Admin", "pro": "⭐ Pro", "free": "Free"}.get(_plan, "Free")
        _plan_color = {"admin": "#ffd700", "pro": "#00d26a", "free": "#5dade2"}.get(_plan, "#5dade2")
        st.markdown(
            f'<div style="background:#1a1d23;border:1px solid #333;border-radius:8px;'
            f'padding:7px 10px;margin:4px 0;display:flex;align-items:center;gap:8px;">'
            f'<div style="font-size:1.1rem">{"🖼️" if not _user.get("avatar") else ""}</div>'
            f'<div><div style="font-size:.8rem;font-weight:600;color:#fafafa">{_user.get("name","User")[:20]}</div>'
            f'<div style="font-size:.68rem;color:{_plan_color}">{_plan_badge}</div></div>'
            f'</div>', unsafe_allow_html=True
        )
    st.markdown("---")
    # Grouped navigation — section headers are visual only
    _navigable = [p for p in PAGES if p not in _NAV_DIVIDERS]
    if default_page not in _navigable:
        default_page = _navigable[0]

    # Render each item: divider as caption, page as button-style radio
    st.markdown("""
    <style>
    div[data-testid="stSidebarNav"] {display:none}
    .nav-section {
        font-size:0.62rem; font-weight:700; letter-spacing:1.5px;
        color:#FF6B35; margin:10px 0 2px 4px; text-transform:uppercase;
    }
    div[data-testid="stRadio"] label {font-size:0.82rem !important;}
    </style>""", unsafe_allow_html=True)

    page = None
    for p in PAGES:
        # Hide Admin page from non-admin users
        if p == "👑 Admin" and not is_admin():
            continue
        if p in _NAV_DIVIDERS:
            label = p.replace("──","").strip()
            if label:
                st.markdown(f'<div class="nav-section">{label}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<hr style="margin:6px 0;border-color:#333">', unsafe_allow_html=True)
        else:
            active = (p == default_page)
            btn = st.button(
                p,
                key=f"nav_{p}",
                use_container_width=True,
                type="primary" if active else "secondary",
            )
            if btn:
                page = p
                default_page = p
                st.query_params["page"] = p

    if page is None:
        page = default_page

    # Persist to URL
    if page != qp.get("page"):
        st.query_params["page"] = page
    
    # Logout button
    if get_current_user():
        if st.button("🚪 Sign Out", key="sidebar_logout", use_container_width=True):
            logout()
    
    st.markdown("---")
    ist = now_ist()
    is_mkt = dtime(9,15) <= ist.time() <= dtime(15,30) and ist.weekday() < 5
    st.caption(f"{'🟢' if is_mkt else '🔴'} {ist.strftime('%d %b, %I:%M %p IST')}")
    
    if st.session_state.regime:
        rg = st.session_state.regime
        st.markdown(f"**{rg['regime_display']}**")
        nv = rg.get("nifty_close", 0)
        if isinstance(nv,(int,float)): st.caption(f"Nifty ₹{nv:,.0f} | Pos {rg['position_multiplier']*100:.0f}%")
    
    sigs = sum(len(v) for v in st.session_state.scan_results.values())
    st.caption(f"Signals: {sigs} | Watch: {len(st.session_state.watchlist)}")
    
    # Signal tracker summary in sidebar
    tracker_df = load_tracker()
    if tracker_df is not None and not tracker_df.empty:
        open_count = len(tracker_df[tracker_df["Status"] == "OPEN"])
        target_count = len(tracker_df[tracker_df["Status"] == "TARGET"])
        stop_count = len(tracker_df[tracker_df["Status"] == "STOPPED"])
        st.caption(f"📊 Tracked: ⏳{open_count} 🟢{target_count} 🔴{stop_count}")
    
    st.markdown("---")
    if st.session_state.breeze_connected:
        st.markdown('<div class="bb bb-on">✅ Breeze Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bb bb-off">⚪ Breeze Off — intraday disabled</div>', unsafe_allow_html=True)
        if st.button("🔄 Retry Breeze", key="retry_breeze", use_container_width=True):
            st.session_state.breeze_connected = False
            st.session_state.breeze_msg = ""
            st.session_state.breeze_engine = None
            try_breeze()
            st.rerun()

    # v16: Supabase status
    render_supabase_status()

    # v16: PF cache refresh
    if st.button("🧠 Refresh Live PF", key="refresh_pf", use_container_width=True,
                 help="Reload profit factors from Supabase after auto-learning runs"):
        reset_pf_cache()
        st.toast("✅ PF cache refreshed — next scan uses latest data")
    
    # Fundamental filter toggle
    st.session_state.fundamental_filter = st.checkbox(
        "🔬 Fundamental Filter",
        value=st.session_state.fundamental_filter,
        key="sidebar_fund_toggle",
        help="Adds grade A/B/C/F to each signal: EPS growth, Revenue growth, PE < 100, D/E < 2x. A/B = trade with confidence. C/F = weak fundamentals, use caution. Best for swing trades.")

    
    # v5.2: Strategy Health Status
    tracker_for_health = load_tracker()
    if tracker_for_health is not None and not tracker_for_health.empty:
        strategy_health.update_from_tracker(tracker_for_health)
        health_items = []
        for strat_key in DAILY_SCANNERS:
            h = strategy_health.get_health(strat_key)
            if h["status"] != "UNKNOWN":
                p = STRATEGY_PROFILES.get(strat_key, {})
                health_items.append(f"{h['icon']} {p.get('name', strat_key)}: PF {h['pf']:.2f}")
        if health_items:
            with st.expander("📊 Strategy Health", expanded=False):
                for item in health_items:
                    st.caption(item)
    
    if st.session_state.last_scan_time:
        age = (now_ist() - st.session_state.last_scan_time).total_seconds() / 60
        color = "color:#00d26a" if age < 15 else "color:#ff8a65"
        st.markdown(f'<small style="{color}">Last scan: {int(age)}m ago</small>', unsafe_allow_html=True)


# ============================================================================
# DAILY FOCUS PANEL — "What matters NOW"
# ============================================================================
def render_focus_panel():
    """The #1 thing the user should see."""
    ist = now_ist()
    hour = ist.hour
    minute = ist.minute
    t = ist.time()
    is_weekend = ist.weekday() >= 5
    regime = st.session_state.regime
    
    # Determine what matters NOW based on time
    if is_weekend:
        focus_title = "📅 Weekend Review"
        focus_action = "Run full Nifty 500 VCP scan. Review journal. Analyze sector rotation."
        focus_tip = "Best time for deep analysis without market noise."
    elif t < dtime(9,15):
        focus_title = "🌅 Pre-Market Prep"
        focus_action = "Load data → Check market health → Review global cues → Plan watchlist"
        focus_tip = "Don't trade the first candle. Observe, then act."
    elif t < dtime(9,45):
        focus_title = "🔔 Market Open — OBSERVE"
        focus_action = "Watch first 15-30 min candle. DO NOT trade. Let the noise settle."
        focus_tip = "Most false breakouts happen in the first 15 minutes."
    elif t < dtime(10,30):
        if st.session_state.breeze_connected:
            focus_title = "🔓 ORB Window (LIVE)"
            focus_action = "Run ORB scanner → Enter confirmed breakouts with volume"
        else:
            focus_title = "⏳ Morning Session"
            focus_action = "ORB requires Breeze API. Focus on swing setups instead."
        focus_tip = "Only take trades with volume confirmation > 1.5x average."
    elif t < dtime(12,30):
        focus_title = "📈 Mid-Morning"
        focus_action = "VWAP Reclaim window (needs Breeze). Trail morning stops."
        focus_tip = "If morning trades are green, trail stop to breakeven."
    elif t < dtime(13,30):
        focus_title = "🍽️ Lunch Session"
        focus_action = "Lunch Low reversal window (needs Breeze). Low-volume = traps."
        focus_tip = "Lunch hour is the least reliable time. Be extra selective."
    elif t < dtime(15,0):
        focus_title = "⏳ Afternoon"
        focus_action = "Review open positions. Prepare for close."
        focus_tip = "Start identifying BTST candidates for 3:20 PM entry."
    elif t < dtime(15,30):
        focus_title = "⭐ Power Hour — Last 30 Min ATH"
        focus_action = "Run ATH scanner NOW → Buy strongest stocks at 3:25 PM"
        focus_tip = "This is the BTST window. Overnight gap-up probability is highest here."
    else:
        focus_title = "📋 Post-Market — Swing Scans"
        focus_action = "Run VCP, EMA21, 52WH, Short scanners → Build tomorrow's watchlist"
        focus_tip = "Post-market is the best time for swing analysis. No noise, clean data."
    
    # Regime context
    regime_str = ""
    if regime:
        r = regime["regime"]
        if r == "EXPANSION":
            regime_str = "🟢 EXPANSION — Go aggressive on breakouts. Full position sizing."
        elif r == "ACCUMULATION":
            regime_str = "🟡 ACCUMULATION — Be selective. Focus on VCP and EMA21 setups."
        elif r == "DISTRIBUTION":
            regime_str = "🟠 DISTRIBUTION — Defensive mode. Prefer shorts or skip breakouts."
        elif r == "PANIC":
            regime_str = "🔴 PANIC — Cash is king. Only short setups or sit out completely."
        else:
            regime_str = "⚪ Unknown regime — Load data to detect."
    
    st.markdown(f"""<div class="focus">
        <h3>🎯 {focus_title}</h3>
        <div style="color:#fafafa; font-size:0.9rem;">{focus_action}</div>
        {"<div class='regime'>" + regime_str + "</div>" if regime_str else ""}
        <div class="tip">💡 {focus_tip}</div>
    </div>""", unsafe_allow_html=True)


# ============================================================================
# DASHBOARD
# ============================================================================
def page_dashboard():
    st.markdown("# 📊 Dashboard")
    
    # FOCUS PANEL — #1 thing user sees
    render_focus_panel()
    
    # Data staleness check
    if st.session_state.data_loaded and is_data_stale():
        st.markdown('<div class="stale">⚠️ Data may be stale. Click Refresh to update.</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.universe_size = st.selectbox("Universe", ["nifty50","nifty200","nifty500"], index=1,
            format_func=lambda x: {"nifty50":"Nifty 50","nifty200":"Nifty 200","nifty500":"Nifty 500"}[x])
    with c2:
        if st.button("🔄 Load / Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            data, nifty, enriched = load_data()
            st.session_state.stock_data = data
            st.session_state.nifty_data = nifty
            st.session_state.enriched_data = enriched
            st.session_state.data_loaded = True
            st.session_state.last_scan_time = now_ist()
            # Detect regime with breadth
            breadth = compute_market_breadth(enriched)
            st.session_state.regime = detect_market_regime(nifty, breadth)
            st.session_state.sector_rankings = compute_sector_ranks()
            # Compute RRG sector rotation
            try:
                rrg = compute_sector_rrg(enriched or data, nifty)
                st.session_state.rrg_data = rrg
            except Exception as e:
                logger.warning(f"RRG computation failed: {e}")
                st.session_state.rrg_data = {}
            st.rerun()
    
    if not st.session_state.data_loaded:
        st.info("👆 Click **Load / Refresh Data** to start.")
        # Show strategies with regime compatibility
        cols = st.columns(4)
        for i, (k, p) in enumerate(STRATEGY_PROFILES.items()):
            with cols[i%4]:
                nb = " 🔌" if p.get("requires_intraday") else ""
                st.markdown(f'<div class="sc"><strong>{p["icon"]} {p["name"]}</strong>{nb}<br>'
                    f'<span class="bg bg-{"i" if p["type"]=="Intraday" else "s"}">{p["type"]}</span> '
                    f'<span style="color:#888;font-size:0.7em">{p["hold"]}</span><br>'
                    f'<span style="color:#00d26a">Win {p["win_rate"]}%</span></div>', unsafe_allow_html=True)
        return
    
    # === REGIME ===
    rg = st.session_state.regime
    if rg:
        st.markdown("### 🧠 Market Regime")
        st.caption(tip("regime"))
        c1,c2,c3,c4 = st.columns(4)
        with c1: pc("Regime", rg["regime_display"])
        with c2: pc("Regime Score", f"{rg['score']}/{rg['max_score']}")
        nv = rg.get("nifty_close", 0)
        with c3: pc("Nifty", f"₹{nv:,.0f}" if isinstance(nv,(int,float)) else str(nv))
        with c4: pc("Position Size", f"{rg['position_multiplier']*100:.0f}%",
                     "g" if rg["position_multiplier"]>=0.6 else "r")
        
        # Show allowed/blocked strategies
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**✅ Ideal now:**")
            for s in rg.get("allowed_strategies",[]):
                st.markdown(f"  {STRATEGY_PROFILES.get(s,{}).get('icon','')} {STRATEGY_PROFILES.get(s,{}).get('name',s)}")
        with c2:
            st.markdown("**⚠️ Use caution:**")
            for s in rg.get("caution_strategies",[]):
                st.markdown(f"  {STRATEGY_PROFILES.get(s,{}).get('icon','')} {STRATEGY_PROFILES.get(s,{}).get('name',s)}")
        with c3:
            st.markdown("**🚫 Blocked:**")
            for s in rg.get("blocked_strategies",[]):
                st.markdown(f"  {STRATEGY_PROFILES.get(s,{}).get('icon','')} {STRATEGY_PROFILES.get(s,{}).get('name',s)}")
        
        with st.expander("📊 Regime Scoring Breakdown"):
            for d in rg.get("details",[]):
                st.markdown(f"  {d}")
            st.caption(f"Score: {rg['score']}/{rg['max_score']} | Vol expansion: {rg.get('vol_expansion','-')}x | "
                       f"RSI: {rg.get('nifty_rsi','-')} | Dist from 52WH: {rg.get('nifty_pct_52wh','-')}%")
        
        # v5.2: Regime-Adaptive Heat Cap display
        current_regime = rg.get("regime", "UNKNOWN")
        heat_cap = RiskManager.get_regime_heat_cap(current_regime)
        risk_per_trade = RiskManager.get_regime_risk_per_trade(current_regime)
        st.markdown(f"**🔥 Portfolio Heat Cap: {heat_cap}%** (Risk/trade: {risk_per_trade}%) — *{current_regime} regime*")
        
        # v5.2: Strategy × Regime Profit Factor Matrix
        with st.expander("📈 Strategy × Regime Profit Factor Matrix", expanded=False):
            st.caption("Pre-computed profit factors from backtests on 38 NSE stocks. Green = edge, Red = avoid.")
            matrix = get_regime_strategy_matrix()
            regimes = ["EXPANSION", "ACCUMULATION", "DISTRIBUTION", "PANIC"]
            matrix_rows = []
            for strat, pfs in matrix.items():
                p = STRATEGY_PROFILES.get(strat, {})
                row = {"Strategy": f"{p.get('icon','')} {p.get('name', strat)}"}
                for regime_name in regimes:
                    pf = pfs.get(regime_name, 0)
                    row[regime_name] = pf
                matrix_rows.append(row)
            matrix_df = pd.DataFrame(matrix_rows)
            
            # Color formatting via styling
            def _color_pf(val):
                if not isinstance(val, (int, float)): return ""
                if val >= 1.5: return "background-color: #0d3320; color: #00d26a"
                elif val >= 1.0: return "background-color: #1a2d1a; color: #81c784"
                elif val >= 0.7: return "background-color: #3d3a1a; color: #ffd700"
                else: return "background-color: #3d1a1a; color: #ff4757"
            
            styled = matrix_df.style.map(_color_pf, subset=regimes)
            st.dataframe(styled, use_container_width=True, hide_index=True)
            
            # Highlight current regime column
            st.info(f"Current regime: **{current_regime}** — strategies with PF < 0.7 are auto-blocked.")
    
    # === BREADTH ===
    breadth = compute_market_breadth(st.session_state.enriched_data or st.session_state.stock_data)
    if breadth:
        st.markdown("### 📊 Market Breadth")
        st.caption(tip("ad_ratio"))
        c1,c2,c3,c4 = st.columns(4)
        with c1: pc("Advancing", str(breadth["advancing"]), "g")
        with c2: pc("Declining", str(breadth["declining"]), "r")
        with c3: pc("A/D Ratio", str(breadth["ad_ratio"]))
        with c4: pc("> 200 SMA", f"{breadth['above_200sma_pct']}%",
                     "g" if breadth["above_200sma_pct"]>50 else "r")
        fig = plot_breadth_gauge(breadth)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    # === SECTOR ===
    st.markdown("### 🗺️ Sector Rotation")
    sector_df = compute_sector_performance(st.session_state.enriched_data or st.session_state.stock_data, get_sector)
    if not sector_df.empty:
        fig = plot_sector_heatmap(sector_df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    # === RRG SECTOR ROTATION ===
    rrg = st.session_state.get("rrg_data", {})
    if rrg:
        with st.expander("🔄 RRG Sector Rotation (Relative Rotation Graph)", expanded=True):
            st.caption("Sectors classified by RS-Ratio (strength vs Nifty) and RS-Momentum (acceleration). "
                       "LEADING sectors boost scan confidence +8, LAGGING sectors penalize -15.")
            q_map = {"LEADING": [], "WEAKENING": [], "IMPROVING": [], "LAGGING": []}
            for sector_name, data in rrg.items():
                q = data.get("quadrant", "LAGGING")
                score = data.get("score", 0)
                q_map.get(q, q_map["LAGGING"]).append((sector_name, score))
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**🟢 LEADING**")
                st.caption("Strong + Improving → +8 confidence")
                for s, sc in sorted(q_map["LEADING"], key=lambda x: -x[1]):
                    st.markdown(f"**{s}** · {sc:.0f}")
                if not q_map["LEADING"]: st.markdown("*None*")
            with c2:
                st.markdown("**🟡 WEAKENING**")
                st.caption("Strong but Slowing")
                for s, sc in sorted(q_map["WEAKENING"], key=lambda x: -x[1]):
                    st.markdown(f"**{s}** · {sc:.0f}")
                if not q_map["WEAKENING"]: st.markdown("*None*")
            with c3:
                st.markdown("**🔵 IMPROVING**")
                st.caption("Weak but Accelerating")
                for s, sc in sorted(q_map["IMPROVING"], key=lambda x: -x[1]):
                    st.markdown(f"**{s}** · {sc:.0f}")
                if not q_map["IMPROVING"]: st.markdown("*None*")
            with c4:
                st.markdown("**🔴 LAGGING**")
                st.caption("Weak + Declining → -15 confidence")
                for s, sc in sorted(q_map["LAGGING"], key=lambda x: -x[1]):
                    st.markdown(f"**{s}** · {sc:.0f}")
                if not q_map["LAGGING"]: st.markdown("*None*")
    
    # === QUICK SCAN ===
    st.markdown("### ⚡ Quick Scan")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.rs_filter = st.slider("Min RS Rating (Long signals only)", 0, 95,
            st.session_state.rs_filter, 5, key="dash_rs",
            help=tip("rs_rating") + " **Note:** SHORT signals are never filtered by RS — weak stocks are ideal short candidates.")
    with c2:
        st.session_state.regime_filter = st.checkbox("Block strategies not suited for current regime",
            value=st.session_state.regime_filter, key="dash_regime",
            help=tip("regime_fit"))
    
    if st.button("🚀 Run All Swing Scanners", type="primary"):
        with st.spinner("Scanning with regime + RS filters..."):
            results = run_all_scanners(
                st.session_state.stock_data, st.session_state.nifty_data, True,
                regime=st.session_state.regime if st.session_state.regime_filter else None,
                has_intraday=st.session_state.breeze_connected,
                sector_rankings=st.session_state.sector_rankings,
                min_rs=st.session_state.rs_filter,
            )
            st.session_state.scan_results = results
            st.session_state.last_scan_time = now_ist()
            # Auto-save signals to log
            all_sigs = [r for sigs in results.values() for r in sigs]
            saved = save_signals_today(all_sigs, st.session_state.regime)
            send_scan_alerts(results)
            st.rerun()
    
    if st.session_state.scan_results:
        # Confluence section
        confluence = detect_confluence(st.session_state.scan_results)
        if confluence:
            st.markdown("### 🔥 Confluence — Multiple Strategy Confirmation")
            for sym, strats in confluence.items():
                strat_names = [STRATEGY_PROFILES.get(s, {}).get("icon", "") + " " +
                              STRATEGY_PROFILES.get(s, {}).get("name", s) for s in strats]
                fno = get_fno_tag(sym)
                st.success(f"**{sym}** [{fno}] — {len(strats)} strategies: {', '.join(strat_names)}")
        
        st.markdown("### 📋 Results")
        
        # v5.2: Basket Export buttons
        all_signals = [r for sigs in st.session_state.scan_results.values() for r in sigs]
        if all_signals:
            c1, c2 = st.columns(2)
            regime_name = rg.get("regime", "UNKNOWN") if rg else "UNKNOWN"
            pos_mult = rg.get("position_multiplier", 1.0) if rg else 1.0
            with c1:
                basket_z = generate_zerodha_basket(all_signals, st.session_state.capital,
                                                    RiskManager.get_regime_risk_per_trade(regime_name), pos_mult)
                if basket_z:
                    st.download_button("📥 Zerodha Basket CSV", basket_z,
                                       f"zerodha_basket_{date.today().isoformat()}.csv", "text/csv",
                                       key="dash_basket_z")
            with c2:
                basket_g = generate_generic_basket(all_signals, st.session_state.capital,
                                                    RiskManager.get_regime_risk_per_trade(regime_name),
                                                    pos_mult, regime_name)
                if basket_g:
                    st.download_button("📥 Trade Plan CSV", basket_g,
                                       f"trade_plan_{date.today().isoformat()}.csv", "text/csv",
                                       key="dash_basket_g")
        
        for strat, results in st.session_state.scan_results.items():
            if not results: continue
            p = STRATEGY_PROFILES.get(strat, {})
            with st.expander(f"{p.get('icon','')} {p.get('name',strat)} — {len(results)}", expanded=True):
                st.dataframe(results_df(results), use_container_width=True, hide_index=True)
        
        # v5.2: Approaching Setups section
        st.markdown("### 👀 Approaching Setups — Forming Watchlist")
        st.caption("Stocks 50-95% through a setup — not yet triggering, but getting close.")
        with st.spinner("Scanning for approaching setups..."):
            approaching = collect_approaching_setups(
                st.session_state.stock_data, st.session_state.nifty_data, min_progress=50)
            st.session_state.approaching_setups = approaching
        if approaching:
            app_rows = []
            for s in approaching[:30]:  # Show top 30
                p = STRATEGY_PROFILES.get(s.strategy, {})
                app_rows.append({
                    "Symbol": s.symbol,
                    "Strategy": f"{p.get('icon','')} {p.get('name', s.strategy)}",
                    "Progress": f"{s.progress_pct:.0f}%",
                    "CMP": fp(s.cmp),
                    "Trigger": fp(s.trigger_level),
                    "RS": int(s.rs_rating),
                    "Description": s.description,
                })
            st.dataframe(pd.DataFrame(app_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No approaching setups found. Stocks may be far from trigger levels.")


# ============================================================================
# SCANNER HUB
# ============================================================================
def page_scanner_hub():
    st.markdown("# 🔍 Scanner Hub")
    if not st.session_state.data_loaded:
        st.warning("Load data from Dashboard first.")
        return
    
    render_focus_panel()
    
    # Inline info markers shown at top of scanner
    st.markdown(
        f"**Quick Reference:** &nbsp;"
        f"RS{tip_md('rs_rating')} &nbsp;·&nbsp;"
        f"SQI{tip_md('sqi')} &nbsp;·&nbsp;"
        f"R:R{tip_md('risk_reward')} &nbsp;·&nbsp;"
        f"Regime Fit{tip_md('regime_fit')} &nbsp;·&nbsp;"
        f"Weekly Align{tip_md('weekly_aligned')} &nbsp;·&nbsp;"
        f"Sector{tip_md('sector')}",
        unsafe_allow_html=True
    )
    with st.expander("ℹ️ What do these terms mean?"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**RS Rating:** {tip('rs_rating')}")
            st.markdown(f"**Confidence:** {tip('confidence')}")
            st.markdown(f"**Regime Fit:** {tip('regime_fit')}")
        with c2:
            st.markdown(f"**R:R Ratio:** {tip('risk_reward')}")
            st.markdown(f"**Weekly Aligned:** {tip('weekly_aligned')}")
            st.markdown(f"**Sector Filter:** {tip('sector')}")
    
    # === FILTER CONTROLS (visible on Scanner Hub too) ===
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.rs_filter = st.slider(
            "Min RS Rating ℹ️ (Long signals only)", 0, 95,
            st.session_state.rs_filter, 5, key="hub_rs",
            help=tip("rs_rating") + " NOTE: Only applies to BUY signals. SHORT signals skip RS filter — weak stocks are ideal short candidates.")
    with c2:
        st.session_state.regime_filter = st.checkbox("Block strategies not suited for current regime",
            value=st.session_state.regime_filter, key="hub_regime",
            help=tip("regime_fit"))
    
    rg = st.session_state.regime
    cols = st.columns(4)
    selected = None
    
    for i, (k, p) in enumerate(STRATEGY_PROFILES.items()):
        with cols[i%4]:
            needs_breeze = p.get("requires_intraday", False)
            # Regime fit
            fit_class = ""
            fit_badge = ""
            if rg:
                if k in rg.get("allowed_strategies",[]): fit_class = "ideal"; fit_badge = '<span class="bg bg-ideal">IDEAL</span>'
                elif k in rg.get("blocked_strategies",[]): fit_class = "blocked"; fit_badge = '<span class="bg bg-blocked">BLOCKED</span>'
                elif k in rg.get("caution_strategies",[]): fit_class = "caution"; fit_badge = '<span class="bg bg-caution">CAUTION</span>'
            
            bc = {"Swing":"bg-s","Intraday":"bg-i","Positional":"bg-p","Overnight":"bg-o"}.get(p["type"],"bg-s")
            
            if needs_breeze and not st.session_state.breeze_connected:
                data_tag = '<small style="color:#ff4757">🔌 Needs Breeze</small>'
            elif needs_breeze:
                data_tag = '<small style="color:#00d26a">🔴 LIVE</small>'
            else:
                data_tag = '<small style="color:#5dade2">📊 Daily</small>'
            
            st.markdown(f'<div class="sc {fit_class}"><strong>{p["icon"]} {p["name"]}</strong><br>'
                f'<span class="bg {bc}">{p["type"]}</span> {fit_badge}<br>'
                f'<span style="color:#00d26a;font-size:0.8em">Win {p["win_rate"]}%</span> · '
                f'<span style="color:#FF6B35;font-size:0.8em">+{p["expectancy"]}%</span><br>{data_tag}</div>',
                unsafe_allow_html=True)
            
            # Disable button if blocked or needs breeze
            disabled = (fit_class == "blocked" and st.session_state.regime_filter) or \
                       (needs_breeze and not st.session_state.breeze_connected)
            if st.button("Scan" if not disabled else "🚫", key=f"s_{k}",
                         use_container_width=True, disabled=disabled,
                         help=tip(k)):
                selected = k
    
    st.markdown("---")
    if st.button("🚀 All Allowed Scanners", type="primary", use_container_width=True):
        selected = "ALL"
    
    if selected:
        n = st.session_state.nifty_data
        if selected == "ALL":
            with st.spinner("Scanning (regime + RS filtered)..."):
                st.session_state.scan_results = run_all_scanners(
                    st.session_state.stock_data, n, True,
                    regime=st.session_state.regime if st.session_state.regime_filter else None,
                    has_intraday=st.session_state.breeze_connected,
                    sector_rankings=st.session_state.sector_rankings,
                    min_rs=st.session_state.rs_filter)
        else:
            with st.spinner(f"Running {STRATEGY_PROFILES[selected]['name']}..."):
                st.session_state.scan_results[selected] = run_scanner(
                    selected, st.session_state.stock_data, n,
                    regime=st.session_state.regime if st.session_state.regime_filter else None,
                    has_intraday=st.session_state.breeze_connected,
                    sector_rankings=st.session_state.sector_rankings,
                    min_rs=st.session_state.rs_filter)
        st.session_state.last_scan_time = now_ist()
        # Auto-save signals to log
        all_sigs = [r for sigs in st.session_state.scan_results.values() for r in sigs]
        saved = save_signals_today(all_sigs, st.session_state.regime)
        send_scan_alerts(st.session_state.scan_results)
        st.rerun()
    
    if not st.session_state.scan_results:
        st.info("Select a strategy.")
        return
    
    # Confluence section
    confluence = detect_confluence(st.session_state.scan_results)
    if confluence:
        st.markdown("#### 🔥 Confluence — Multiple Strategy Confirmation")
        for sym, strats in confluence.items():
            strat_names = [STRATEGY_PROFILES.get(s, {}).get("icon", "") + " " +
                          STRATEGY_PROFILES.get(s, {}).get("name", s) for s in strats]
            fno = get_fno_tag(sym)
            st.success(f"**{sym}** [{fno}] — {len(strats)} strategies: {', '.join(strat_names)}")
    
    # v5.2: Basket Export
    all_signals = [r for sigs in st.session_state.scan_results.values() for r in sigs]
    if all_signals:
        rg = st.session_state.regime
        regime_name = rg.get("regime", "UNKNOWN") if rg else "UNKNOWN"
        pos_mult = rg.get("position_multiplier", 1.0) if rg else 1.0
        c1, c2, c3 = st.columns(3)
        with c1:
            basket_z = generate_zerodha_basket(all_signals, st.session_state.capital,
                                                RiskManager.get_regime_risk_per_trade(regime_name), pos_mult)
            if basket_z:
                st.download_button("📥 Zerodha Basket", basket_z,
                                   f"zerodha_basket_{date.today().isoformat()}.csv", "text/csv",
                                   key="hub_basket_z")
        with c2:
            basket_g = generate_generic_basket(all_signals, st.session_state.capital,
                                                RiskManager.get_regime_risk_per_trade(regime_name),
                                                pos_mult, regime_name)
            if basket_g:
                st.download_button("📥 Full Trade Plan CSV", basket_g,
                                   f"trade_plan_{date.today().isoformat()}.csv", "text/csv",
                                   key="hub_basket_g")
        with c3:
            st.caption(f"{len(all_signals)} signals | Capital ₹{st.session_state.capital:,.0f} | Risk/trade: {RiskManager.get_regime_risk_per_trade(regime_name)}%")
    
    for strategy, results in st.session_state.scan_results.items():
        if not results: continue
        p = STRATEGY_PROFILES.get(strategy, {})
        st.markdown(f"#### {p.get('icon','')} {p.get('name',strategy)} — {len(results)}")
        st.dataframe(results_df(results), use_container_width=True, hide_index=True)
        
        for r in results:
            sqi_val = getattr(r, 'sqi', None)
            sqi_icon = getattr(r, 'sqi_icon', '')
            sqi_grade = getattr(r, 'sqi_grade', '')
            sqi_breakdown = getattr(r, 'sqi_breakdown', '')
            
            # SQI tooltip explanation
            SQI_TOOLTIP = TIPS.get("sqi", "SQI: composite signal quality score 0-100")
            
            # Build header: SQI if available, else confidence
            direction_label = "🟢 BUY" if r.signal == "BUY" else "🔴 SHORT"
            if sqi_val is not None:
                header = f"📋 {r.symbol} — {direction_label} | CMP {fp(r.cmp)} | {sqi_icon} SQI {sqi_val:.0f} ({sqi_grade}) | RS {r.rs_rating:.0f}"
            else:
                header = f"📋 {r.symbol} — {direction_label} | CMP {fp(r.cmp)} | Conf {r.confidence}% | RS {r.rs_rating:.0f}"
            
            with st.expander(header):
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                with c1: pc("CMP", fp(r.cmp))
                with c2: pc("Entry", fp(r.entry))
                with c3: pc("SL", fp(r.stop_loss), "r")
                with c4: pc("T1", fp(r.target_1), "g")
                with c5: pc("T2", fp(r.target_2), "g")
                with c6: pc("Regime", r.regime_fit, "g" if r.regime_fit=="IDEAL" else ("y" if r.regime_fit=="CAUTION" else ""))
                
                # v5.2: SQI breakdown with info
                if sqi_val is not None:
                    sqi_col1, sqi_col2 = st.columns([10, 1])
                    with sqi_col1:
                        grade_color = {"ELITE":"gold","STRONG":"#5dade2","MODERATE":"#ffd700","WEAK":"#ff4757"}.get(sqi_grade,"white")
                        st.markdown(
                                    f'**{sqi_icon} Signal Quality: <span style="color:{grade_color}">' +
                                    f'{sqi_val:.0f}/100 — {sqi_grade}</span>**' +
                                    tip_md("sqi") + " " + tip_md("sqi_grade", "grade ℹ️"),
                                    unsafe_allow_html=True)
                    st.caption(f"Breakdown: {sqi_breakdown}")
                    st.caption("SQI = Backtest Edge 30% + RS Strength 25% + Regime Fit 20% + Vol Contraction 15% + Volume 10%")
                
                # v5.2: Fundamental Gate (if enabled)
                if st.session_state.fundamental_filter:
                    if r.symbol not in st.session_state.fundamental_cache:
                        st.session_state.fundamental_cache[r.symbol] = check_fundamental_quality(r.symbol)
                    fgate = st.session_state.fundamental_cache[r.symbol]
                    fund_color = "g" if fgate.grade in ("A", "B+") else ("y" if fgate.grade == "B" else ("o" if fgate.grade == "C" else "r"))
                    st.markdown(f"**{fgate.grade_icon} Fundamental: {fgate.grade}** ({fgate.score}/{fgate.max_score})")
                    if fgate.warnings:
                        for w in fgate.warnings:
                            st.caption(f"⚠️ {w}")
                    with st.expander("📊 Fundamental Details", expanded=False):
                        for d in fgate.details:
                            st.caption(d)
                
                # Multi-timeframe
                if r.symbol in (st.session_state.enriched_data or {}):
                    mtf = check_weekly_alignment(st.session_state.enriched_data[r.symbol])
                    if mtf["aligned"]:
                        st.success(f"✅ Weekly confirms ({mtf['score']}/4)")
                    else:
                        st.warning(f"⚠️ Weekly not aligned ({mtf['score']}/4)")
                
                # Chart
                if r.symbol in (st.session_state.enriched_data or {}):
                    fig = plot_candlestick(st.session_state.enriched_data[r.symbol], r.symbol,
                        entry=r.entry, stop_loss=r.stop_loss, target1=r.target_1, target2=r.target_2, signal=r.signal)
                    st.plotly_chart(fig, use_container_width=True)
                
                # "Why This Stock" — explainability
                st.markdown("**Why this stock qualified:**")
                for reason in r.reasons: st.markdown(f"  • {reason}")
                
                c1,c2,c3 = st.columns(3)
                with c1:
                    if st.button("⭐ Watch", key=f"a_{strategy}_{r.symbol}"):
                        ent = {"symbol":r.symbol,"strategy":strategy,"cmp":r.cmp,"entry":r.entry,
                               "stop":r.stop_loss,"target1":r.target_1,"target2":r.target_2,
                               "confidence":r.confidence,"date":r.timestamp,"entry_type":r.entry_type,"regime":r.regime_fit}
                        if not any(w["symbol"]==r.symbol and w["strategy"]==strategy for w in st.session_state.watchlist):
                            st.session_state.watchlist.append(ent)
                            st.success("Added!")
                with c2:
                    if st.button("📱 TG", key=f"t_{strategy}_{r.symbol}"):
                        if send_tg(fmt_alert(r)): st.success("Sent!")
                        else: st.warning("Setup Telegram first")
                with c3:
                    if PAPER_TRADING_AVAILABLE:
                        if st.button("🎮 Paper Trade", key=f"pt_{strategy}_{r.symbol}", help="Deploy ₹50,000 virtual capital on this signal"):
                            # Build a signal dict compatible with enter_paper_trade
                            sig_dict = {
                                "id": None,
                                "symbol": r.symbol, "strategy": strategy, "signal": r.signal,
                                "entry": r.entry, "cmp": r.cmp, "sl": r.stop_loss,
                                "t1": r.target_1, "t2": r.target_2, "rr": r.risk_reward,
                                "sqi": getattr(r,"sqi",50), "sqi_grade": getattr(r,"sqi_grade",""),
                                "sector": getattr(r,"sector",""), "regime": getattr(r,"regime_fit",""),
                            }
                            result = enter_paper_trade(sig_dict)
                            if result:
                                st.success(f"🎮 ₹50K virtual deployed on {r.symbol}!")
                            else:
                                st.warning("Could not open paper trade. Check Virtual Game page.")


# ============================================================================
# CHARTS & RS
# ============================================================================
def page_charts_rs():
    st.markdown("# 📈 Charts & Relative Strength")
    if not st.session_state.data_loaded:
        st.warning("Load data first.")
        return
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart", "💪 RS Rankings", "🗺️ Sectors", "📊 Volume Profile"])
    with tab1:
        enriched = st.session_state.enriched_data or st.session_state.stock_data
        sel = st.selectbox("Stock", sorted(enriched.keys()))
        days = st.slider("Days", 30, 250, 90)
        if sel in enriched:
            fig = plot_candlestick(enriched[sel], sel, days=days)
            st.plotly_chart(fig, use_container_width=True)
            lat = enriched[sel].iloc[-1]
            c1,c2,c3,c4 = st.columns(4)
            with c1: pc("CMP", fp(lat["close"]))
            with c2: pc("RSI", f"{lat.get('rsi_14',0):.0f}")
            with c3: pc("52W High", fp(lat.get("high_52w",0)))
            with c4:
                mtf = check_weekly_alignment(enriched[sel])
                pc("Weekly", f"{'✅' if mtf['aligned'] else '❌'} {mtf['score']}/4")
    with tab2:
        rs_df = compute_rs_rankings(st.session_state.enriched_data or st.session_state.stock_data,
                                     st.session_state.nifty_data, get_sector)
        if not rs_df.empty:
            fig = plot_rs_scatter(rs_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### ⭐ Top 20 RS Leaders")
            st.dataframe(rs_df.head(20)[["Symbol","Sector","CMP","1M %","3M %","RS Score","RS Rank"]], use_container_width=True, hide_index=True)
            st.markdown("#### 🔴 Bottom 20")
            st.dataframe(rs_df.tail(20)[["Symbol","Sector","CMP","1M %","3M %","RS Score","RS Rank"]], use_container_width=True, hide_index=True)
    with tab3:
        sector_df = compute_sector_performance(st.session_state.enriched_data or st.session_state.stock_data, get_sector)
        if not sector_df.empty:
            fig = plot_sector_heatmap(sector_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sector_df.reset_index().rename(columns={"index":"Sector","stocks":"#","avg_1w":"1W%","avg_1m":"1M%","avg_3m":"3M%"}),
                         use_container_width=True, hide_index=True)
    
    with tab4:
        st.markdown("### 📊 Volume Profile")
        st.caption("Point of Control (POC), Value Area High/Low, and High/Low Volume Nodes from Breeze intraday data.")
        
        if not st.session_state.breeze_connected:
            st.warning("🔌 Volume Profile requires Breeze API connection. Connect Breeze to use this feature.")
        else:
            enriched = st.session_state.enriched_data or st.session_state.stock_data
            vp_sym = st.selectbox("Stock", sorted(enriched.keys()), key="vp_sym")
            c1, c2 = st.columns(2)
            with c1:
                vp_days = st.slider("Lookback Days", 5, 60, 20, key="vp_days")
            with c2:
                vp_bins = st.slider("Price Bins", 10, 40, 20, key="vp_bins")
            
            if st.button("📊 Compute Volume Profile", key="vp_run"):
                be = st.session_state.breeze_engine
                if be:
                    with st.spinner(f"Fetching {vp_days}d intraday data for {vp_sym}..."):
                        vp = be.fetch_volume_profile(vp_sym, days_back=vp_days, num_bins=vp_bins)
                    if vp:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: pc("POC (Point of Control)", fp(vp["poc"]), "g")
                        with c2: pc("VAH (Value Area High)", fp(vp["vah"]), "g")
                        with c3: pc("VAL (Value Area Low)", fp(vp["val"]), "r")
                        with c4: pc("Data Bars", str(vp["total_bars"]))
                        
                        if vp["hvn"]:
                            st.markdown(f"**🟢 HVN (High Vol Nodes — Support/Resistance):** {', '.join([fp(p) for p in vp['hvn']])}")
                        if vp["lvn"]:
                            st.markdown(f"**🔵 LVN (Low Vol Nodes — Breakout Zones):** {', '.join([fp(p) for p in vp['lvn']])}")
                        
                        # Volume Profile chart
                        try:
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=vp["volumes"], y=vp["price_levels"],
                                orientation='h', name="Volume",
                                marker_color=["#FF6B35" if p == vp["poc"] else "#5dade2" for p in vp["price_levels"]]
                            ))
                            fig.add_hline(y=vp["poc"], line_color="#FF6B35", annotation_text="POC")
                            fig.add_hline(y=vp["vah"], line_color="#00d26a", line_dash="dash", annotation_text="VAH")
                            fig.add_hline(y=vp["val"], line_color="#ff4757", line_dash="dash", annotation_text="VAL")
                            fig.update_layout(template="plotly_dark", title=f"{vp_sym} Volume Profile ({vp_days}d)",
                                              height=500, margin=dict(t=40, b=30))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                        
                        st.caption("POC = most traded price level. HVN = areas of high acceptance (support/resistance). "
                                   "LVN = areas of low acceptance (potential breakout/breakdown zones).")
                    else:
                        st.warning(f"Could not compute Volume Profile for {vp_sym}. Breeze may not have sufficient intraday data.")
                else:
                    st.error("Breeze engine not initialized.")


# ============================================================================
# TRADE PLANNER
# ============================================================================
def page_trade_planner():
    st.markdown("# 📐 Trade Planner")
    st.caption("Calculate exact position size, stop loss, and targets for any trade signal.")
    with st.expander("💡 How to use Trade Planner", expanded=False):
        st.markdown("""
**Workflow:**
1. Run a scan in Scanner Hub → a signal appears (e.g. VCP on ASHOKLEY, Entry ₹220, SL ₹207)
2. Come to Trade Planner → select that signal from the dropdown
3. The planner automatically fills Entry and SL from the scan
4. It shows: **how many shares to buy**, **exact ₹ risk**, **T1/T2/T3 targets**
5. It also shows if your total portfolio risk is within safe limits

**The formula:**  
Shares = (Capital × Risk%) ÷ (Entry - Stop Loss)  
Example: ₹5,00,000 × 1.5% = ₹7,500 risk ÷ ₹13 per share = 576 shares

**Regime adjustment:** In PANIC market, position size is automatically reduced to 15%.
        """)
    c1,c2 = st.columns(2)
    with c1:
        capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000, min_value=10000)
        st.session_state.capital = capital
        risk_pct = st.slider("Risk %", 0.5, 3.0, 2.0, 0.25)
        sigs = [(f"{r.symbol} ({s}) {fp(r.cmp)}", s, r) for s, res in st.session_state.scan_results.items() for r in res]
        mode = st.radio("Input", ["Scanner","Manual"], horizontal=True)
        if mode == "Scanner" and sigs:
            sel = st.selectbox("Signal", [s[0] for s in sigs])
            r = sigs[[s[0] for s in sigs].index(sel)][2]
            entry, sl, short = r.entry, r.stop_loss, r.signal == "SHORT"
            st.info(f"**{r.symbol}** {r.signal} | CMP {fp(r.cmp)} | "
                   f"{getattr(r, 'sqi_icon', '')} SQI {getattr(r, 'sqi', r.confidence):.0f} | Regime: {r.regime_fit}")
        else:
            entry = st.number_input("Entry ₹", value=100.0, step=1.0)
            sl = st.number_input("SL ₹", value=95.0, step=1.0)
            short = st.checkbox("Short")
    with c2:
        mult = st.session_state.regime.get("position_multiplier", 1.0) if st.session_state.regime else 1.0
        regime_name = st.session_state.regime.get("regime", "UNKNOWN") if st.session_state.regime else "UNKNOWN"
        heat_cap = RiskManager.get_regime_heat_cap(regime_name)
        regime_risk = RiskManager.get_regime_risk_per_trade(regime_name)
        if mult < 0.6: st.warning(f"⚠️ Regime: positions at {mult*100:.0f}%")
        st.caption(f"🔥 Regime: {regime_name} | Heat cap: {heat_cap}% | Max risk/trade: {regime_risk}%")
        if entry > 0 and sl > 0 and entry != sl:
            pos = RiskManager.calculate_position(capital, risk_pct, entry, sl, mult)
            tgt = RiskManager.calculate_targets(entry, sl, short)
            c1,c2 = st.columns(2)
            with c1: pc("Shares", f"{pos.shares:,}"); pc("Position", f"₹{pos.position_value:,.0f}")
            with c2: pc("Risk", f"₹{pos.risk_amount:,.0f}"); pc("% Portfolio", f"{pos.pct_of_portfolio:.1f}%")
            for w in pos.warnings: st.warning(w)
            st.markdown("### 🎯 Targets")
            c1,c2,c3 = st.columns(3)
            with c1: pc("T1 (1.5R)", fp(tgt.t1), "g")
            with c2: pc("T2 (2.5R)", fp(tgt.t2), "g")
            with c3: pc("T3 (4R)", fp(tgt.t3), "g")
            st.markdown(f"**Trail at** {fp(tgt.trailing_trigger)} → SL to breakeven")
            for lb, m in [("SL",-1),("T1",1.5),("T2",2.5),("T3",4)]:
                pnl = pos.shares * m * tgt.risk_per_share
                st.markdown(f"{'🟢' if pnl>0 else '🔴'} **{lb}:** ₹{pnl:+,.0f}")


# ============================================================================
# WATCHLIST
# ============================================================================
def page_watchlist():
    st.markdown("# ⭐ Watchlist")
    st.caption("Save stocks from Scanner Hub to track, plus see which setups are 'almost triggering'.")
    with st.expander("💡 How to use Watchlist", expanded=False):
        st.markdown("""
**Two sections:**

**⭐ Manual Watchlist:** Stocks you've saved from Scanner Hub by clicking ⭐ Watch. Shows entry, SL, target for each.

**👀 Approaching Setups:** Stocks that are 50–95% through a setup but haven't triggered yet. These are your "get ready" stocks. Example: a VCP setup that's 85% compressed — it may break out tomorrow. Set a price alert on these stocks so you're ready when they trigger.

**How to add stocks:** In Scanner Hub, expand any signal and click ⭐ Watch.
        """)
    
    tab1, tab2 = st.tabs(["⭐ Manual Watchlist", "👀 Approaching Setups"])
    
    with tab1:
        if not st.session_state.watchlist:
            st.info("Empty. Add from Scanner Hub.")
        else:
            rows = [{"#":i+1,"Symbol":w["symbol"],"Strategy":w["strategy"],"CMP":fp(w.get("cmp",w["entry"])),
                     "Entry":fp(w["entry"]),"SL":fp(w["stop"]),"T1":fp(w["target1"]),
                     "Conf":f"{w['confidence']}%","Regime":w.get("regime","")
            } for i, w in enumerate(st.session_state.watchlist)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            c1,c2 = st.columns(2)
            with c1:
                syms = [f"{w['symbol']} ({w['strategy']})" for w in st.session_state.watchlist]
                to_rm = st.selectbox("Remove", syms)
                if st.button("🗑️ Remove"):
                    st.session_state.watchlist.pop(syms.index(to_rm)); st.rerun()
            with c2:
                if st.button("🗑️ Clear All"):
                    st.session_state.watchlist = []; st.rerun()
    
    with tab2:
        st.caption("Stocks 50-95% through a setup — not yet triggering, but on the way. Auto-refreshes with data load.")
        if not st.session_state.data_loaded:
            st.info("Load data from Dashboard to see approaching setups.")
        elif st.button("🔄 Refresh Approaching Setups", key="wl_refresh_app"):
            with st.spinner("Scanning for approaching setups..."):
                approaching = collect_approaching_setups(
                    st.session_state.stock_data, st.session_state.nifty_data, min_progress=50)
                st.session_state.approaching_setups = approaching
                st.rerun()
        
        approaching = st.session_state.approaching_setups
        if approaching:
            app_rows = []
            for s in approaching[:50]:
                p = STRATEGY_PROFILES.get(s.strategy, {})
                app_rows.append({
                    "Symbol": s.symbol,
                    "Strategy": f"{p.get('icon','')} {p.get('name', s.strategy)}",
                    "Progress": f"{s.progress_pct:.0f}%",
                    "CMP": fp(s.cmp),
                    "Trigger": fp(s.trigger_level),
                    "RS": int(s.rs_rating),
                    "Description": s.description,
                })
            st.dataframe(pd.DataFrame(app_rows), use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(app_rows)} of {len(approaching)} approaching setups.")
        else:
            st.info("No approaching setups found. Run a data load first, or stocks may be far from triggers.")


# ============================================================================
# JOURNAL
# ============================================================================
def page_journal():
    st.markdown("# 📓 Trade Journal")
    st.caption("Record every trade you take, track your P&L, and see your equity curve over time.")
    with st.expander("💡 How to use Journal", expanded=False):
        st.markdown("""
**Why journal?** Most traders lose money because they don't know which of their trades are actually profitable. The journal tells you.

**Workflow:**
1. When you take a trade from Scanner Hub, click 📓 Journal button on that signal — it pre-fills entry/SL/target
2. When you exit the trade, come to Journal, find the open trade, enter the exit price → it calculates your P&L
3. Over time the **equity curve** shows if you're actually making money

**Key stats to watch:**
- **Win Rate > 50%** with R:R of 1:2 = you're profitable
- **Profit Factor > 1.3** = every ₹1 lost, you make ₹1.30+ in wins
- If your actual results are worse than backtest → you're overriding signals or entering late
        """)
    journal = st.session_state.journal
    analytics = compute_journal_analytics(journal)
    if analytics and analytics.get("closed_trades", 0) > 0:
        c1,c2,c3,c4 = st.columns(4)
        with c1: pc("Win Rate", f"{analytics['win_rate']}%", "g" if analytics["win_rate"]>55 else "r")
        with c2: pc("Total P&L", f"₹{analytics['total_pnl']:+,.0f}", "g" if analytics["total_pnl"]>0 else "r")
        with c3: pc("Profit Factor", str(analytics["profit_factor"]))
        with c4: pc("Expectancy", f"₹{analytics['expectancy']:,.0f}/trade")
        fig = plot_equity_curve(analytics)
        if fig: st.plotly_chart(fig, use_container_width=True)
        if analytics.get("strategy_stats"):
            rows = []
            for s, d in analytics["strategy_stats"].items():
                wr = d["wins"]/d["trades"]*100 if d["trades"] else 0
                rows.append({"Strategy":s,"Trades":d["trades"],"Win%":f"{wr:.0f}%","P&L":f"₹{d['pnl']:+,.0f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    with st.form("add"):
        c1,c2,c3 = st.columns(3)
        with c1:
            sym = st.text_input("Symbol"); strat = st.selectbox("Strategy", list(STRATEGY_PROFILES.keys()))
            sig = st.selectbox("Signal", ["BUY","SHORT"])
        with c2:
            ep = st.number_input("Entry ₹",min_value=0.0,step=1.0)
            sl_ = st.number_input("SL ₹",min_value=0.0,step=1.0)
            tg = st.number_input("Target ₹",min_value=0.0,step=1.0)
        with c3:
            qty = st.number_input("Qty",min_value=1,value=1)
            status = st.selectbox("Status",["open","closed"])
            ex = st.number_input("Exit ₹",min_value=0.0,step=1.0)
        notes = st.text_area("Notes")
        if st.form_submit_button("Add"):
            pnl = (ex-ep)*qty if status=="closed" and ex>0 else 0
            if sig=="SHORT" and status=="closed" and ex>0: pnl=(ep-ex)*qty
            add_journal_entry({"symbol":sym.upper(),"strategy":strat,"signal":sig,"entry":ep,"stop":sl_,
                "target1":tg,"qty":qty,"status":status,"exit":ex if ex>0 else None,"pnl":pnl,
                "notes":notes,"entry_date":str(now_ist().date()),
                "exit_date":str(now_ist().date()) if status=="closed" else None})
            st.session_state.journal = load_journal()
            st.success(f"Added! P&L: ₹{pnl:+,.0f}" if pnl else "Added!"); st.rerun()
    
    if journal:
        rows = [{"#":e.get("id",""),"Symbol":e.get("symbol",""),"Strategy":e.get("strategy",""),
                 "Status":e.get("status",""),"P&L":f"₹{e.get('pnl',0):+,.0f}" if e.get("status")=="closed" else "—",
                 "Notes":e.get("notes","")[:30]} for e in reversed(journal)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        open_t = [e for e in journal if e.get("status")=="open"]
        if open_t:
            st.markdown("### Close Trade")
            labels = [f"#{e['id']} {e['symbol']}" for e in open_t]
            sel = st.selectbox("Trade", labels)
            trade = open_t[labels.index(sel)]
            ex = st.number_input("Exit ₹", min_value=0.0, step=1.0, key="cx")
            if st.button("Close"):
                if ex > 0:
                    pnl = (ex-trade["entry"])*trade.get("qty",1)
                    if trade.get("signal")=="SHORT": pnl=(trade["entry"]-ex)*trade.get("qty",1)
                    for e in journal:
                        if e.get("id")==trade["id"]:
                            e["status"]="closed"; e["exit"]=ex; e["pnl"]=pnl; e["exit_date"]=str(now_ist().date())
                    save_journal(journal); st.session_state.journal=journal
                    st.success(f"P&L: ₹{pnl:+,.0f}"); st.rerun()



# ============================================================================
# BACKTEST ENGINE
# ============================================================================
def page_backtest():
    st.markdown("# 🧪 Backtest Engine")
    st.caption("Tests how a strategy would have performed on historical data. Use to validate a strategy before trading real money.")

    with st.expander("💡 How to use Backtest", expanded=False):
        st.markdown("""
**What it does:** Simulates a strategy on past NSE data and shows win rate, profit factor, and equity curve.

**Typical workflow:**
1. Load data from Dashboard (Nifty 200 or 500)
2. Select a strategy (e.g. VCP) and a stock
3. Run backtest → check **Net Profit Factor** (PF > 1.3 = has edge)
4. Run **Portfolio Backtest** to see strategy performance across 200+ stocks
5. Use results to decide which strategies to focus on in Scanner Hub

**Key numbers to look at:**
- **Net PF > 1.3** = strategy has real edge after costs
- **Win Rate > 45%** = acceptable (high R:R strategies win less often but still profit)
- **Max Drawdown < 20%** = survivable worst case
- **Avg Hold < 25 days** = signals resolve quickly
        """)

    if not st.session_state.data_loaded:
        st.warning("Load data from Dashboard first.")
        return
    
    with st.expander("ℹ️ How does backtesting work?", expanded=False):
        st.markdown(f"""
**What it does:** Simulates running a strategy on historical data to see how it would have performed.

**Period:** Uses ~1 year of daily data from Yahoo Finance (approximately 248 trading days). This is the maximum available without a premium data feed.

**How it works (walk-forward, no cheating):**
1. For each historical day, the engine checks if the scanner would have triggered a signal using ONLY data available up to that day (no future peeking)
2. If a signal fires, it simulates the trade with the exact Entry, Stop Loss, and Target levels
3. Trade exits when: Stop Loss hit → loss, Target 1 hit → win, or Max Hold days reached → exit at close
4. All trades are logged and statistics computed

**Key metrics explained:**
- **{tip('win_rate')}**
- **{tip('profit_factor')}**
- **{tip('expectancy')}**
- **{tip('max_drawdown')}**

**Limitations:** Only ~1 year of data. Strategies like VCP need bull markets to shine — if the test period is sideways, VCP may underperform. Always consider the market regime during the test period.
""")
    
    tab1, tab2 = st.tabs(["📊 Single Stock", "📈 Multi-Stock Portfolio"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            stock_data = st.session_state.stock_data or {}
            sym = st.selectbox("Stock", sorted(stock_data.keys()), key="bt_sym") if stock_data else None
        with c2:
            strat = st.selectbox("Strategy", list(DAILY_SCANNERS.keys()),
                format_func=lambda x: f"{STRATEGY_PROFILES.get(x,{}).get('icon','')} {STRATEGY_PROFILES.get(x,{}).get('name',x)}",
                key="bt_strat", help="Only daily strategies can be backtested (intraday needs live Breeze data).")
        with c3:
            max_hold = st.number_input("Max Hold (days)", 5, 60, 20, 5, key="bt_hold",
                help="Force exit after N days if neither SL nor T1 is hit. Shorter = more trades, less risk per trade.")
        
        if sym and st.button("🧪 Run Backtest", type="primary", key="bt_run1"):
            if sym not in stock_data:
                st.error(f"{sym} data not available."); return
            
            run_wf = st.session_state.get("bt_wf_toggle", False)
            with st.spinner(f"Backtesting {STRATEGY_PROFILES[strat]['name']} on {sym} (~1 year data)..."):
                if run_wf:
                    result = backtest_walk_forward(stock_data[sym], sym, strat, max_hold=max_hold)
                else:
                    result = backtest_strategy(stock_data[sym], sym, strat,
                                              lookback_days=500, max_hold=max_hold)
            
            if result and result.total_trades > 0:
                _render_backtest_result(result)
            else:
                st.info(f"No {STRATEGY_PROFILES[strat]['name']} signals found for {sym} in ~1 year of data. "
                        "This strategy may need different market conditions to trigger.")
        
        st.session_state["bt_wf_toggle"] = st.checkbox(
            "🔬 Walk-Forward Validation (70% train / 30% test split)",
            value=st.session_state.get("bt_wf_toggle", False), key="bt_wf_cb",
            help="Splits data into train (70%) and test (30%). Detects overfitting if train win rate exceeds test by >15%. "
                 "Requires at least 400 days of data per stock. Off by default — most strategies use established methodologies."
        )
    
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            strat2 = st.selectbox("Strategy", list(DAILY_SCANNERS.keys()),
                format_func=lambda x: f"{STRATEGY_PROFILES.get(x,{}).get('icon','')} {STRATEGY_PROFILES.get(x,{}).get('name',x)}",
                key="bt_strat2")
        with c2:
            max_hold2 = st.number_input("Max Hold (days)", 5, 60, 20, 5, key="bt_hold2")
        with c3:
            stock_count = len(st.session_state.stock_data) if st.session_state.stock_data else 0
            st.metric("Stocks in Universe", stock_count)
        
        if st.button("🧪 Run Portfolio Backtest", type="primary", key="bt_run2"):
            stock_data = st.session_state.stock_data or {}
            with st.spinner(f"Backtesting {STRATEGY_PROFILES[strat2]['name']} across {len(stock_data)} stocks (~1 year each)..."):
                result2 = backtest_multi_stock(
                    list(stock_data.keys()),
                    stock_data,
                    strat2, lookback=500, max_hold=max_hold2)
            
            if result2 and result2.total_trades > 0:
                _render_backtest_result(result2)
            else:
                st.info(f"No {STRATEGY_PROFILES[strat2]['name']} signals found across the portfolio.")
    
    # Strategy comparison
    st.markdown("---")
    st.markdown("### 📊 Compare All Strategies")
    st.caption("Runs each daily strategy across your entire loaded universe (~1 year) and compares performance side by side.")
    if st.button("🔬 Run All Strategy Backtests", key="bt_compare"):
        stock_data = st.session_state.stock_data or {}
        if not stock_data:
            st.warning("No data loaded."); return
        comparison = []
        progress = st.progress(0)
        strats = list(DAILY_SCANNERS.keys())
        for idx, s in enumerate(strats):
            progress.progress((idx + 1) / len(strats), f"Testing {STRATEGY_PROFILES[s]['name']}...")
            r = backtest_multi_stock(list(stock_data.keys()),
                                     stock_data, s, lookback=500, max_hold=20)
            if r and r.total_trades > 0:
                comparison.append({
                    "Strategy": f"{STRATEGY_PROFILES[s]['icon']} {STRATEGY_PROFILES[s]['name']}",
                    "Trades": r.total_trades,
                    "Win Rate": f"{r.win_rate}%",
                    "Total P&L": f"{r.total_pnl_pct:+.1f}%",
                    "Avg Win": f"+{r.avg_win_pct:.1f}%",
                    "Avg Loss": f"-{r.avg_loss_pct:.1f}%",
                    "Profit Factor": r.profit_factor,
                    "Max DD": f"{r.max_drawdown_pct:.1f}%",
                    "Expectancy": f"{r.expectancy_pct:+.2f}%",
                    "Avg Hold": f"{r.avg_holding_days:.0f}d",
                })
            else:
                comparison.append({
                    "Strategy": f"{STRATEGY_PROFILES[s]['icon']} {STRATEGY_PROFILES[s]['name']}",
                    "Trades": 0, "Win Rate": "-", "Total P&L": "-", "Avg Win": "-",
                    "Avg Loss": "-", "Profit Factor": "-", "Max DD": "-", 
                    "Expectancy": "-", "Avg Hold": "-",
                })
        progress.empty()
        if comparison:
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
            st.caption("📌 Data period: ~1 year per stock from Yahoo Finance. "
                       "Strategies with 0 trades had no matching signals in this period.")


def _render_backtest_result(result):
    """Render backtest results with metrics, equity curve, and trade log."""
    st.markdown(f"### Results: {result.strategy} on {result.symbol}")
    st.caption(f"Period: {result.period} | Net P&L after all realistic Indian market costs")
    
    # Walk-forward info
    if getattr(result, "is_walk_forward", False):
        col1, col2 = st.columns(2)
        with col1: st.info(f"**Train Win Rate:** {result.train_win_rate:.1f}%")
        with col2: st.info(f"**Test Win Rate:** {result.test_win_rate:.1f}%")
        if getattr(result, "overfit_warning", False):
            st.error(f"⚠️ **Overfit Warning:** Train ({result.train_win_rate:.1f}%) exceeds Test "
                     f"({result.test_win_rate:.1f}%) by more than 15%. Strategy may be curve-fitted.")
        else:
            st.success("✅ Walk-Forward: Train and Test performance are consistent (no overfit detected).")
    
    # Key metrics row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: pc("Trades", str(result.total_trades))
    with c2: pc("Win Rate", f"{result.win_rate}%", "g" if result.win_rate > 50 else "r")
    with c3: pc("Gross P&L", f"{result.total_pnl_pct:+.1f}%", "g" if result.total_pnl_pct > 0 else "r")
    net_pnl = getattr(result, "net_pnl_pct", result.total_pnl_pct)
    with c4: pc("Net P&L", f"{net_pnl:+.1f}%", "g" if net_pnl > 0 else "r")
    with c5: pc("Net PF", str(getattr(result, "net_profit_factor", result.profit_factor)),
                "g" if getattr(result, "net_profit_factor", result.profit_factor) > 1.5 else "r")
    with c6: pc("Max Drawdown", f"{result.max_drawdown_pct:.1f}%", "r")
    
    # Costs summary
    total_costs = getattr(result, "total_costs_pct", 0)
    if total_costs > 0:
        st.caption(f"💰 Total trading costs: {total_costs:.2f}% | Avg per trade: {total_costs/result.total_trades:.2f}% "
                   f"(includes slippage 0.20%, STT 0.025%, brokerage ₹40, exchange charges, GST)")
    
    # Interpretation
    net_pf = getattr(result, "net_profit_factor", result.profit_factor)
    if net_pf > 1.5 and result.win_rate > 45:
        st.success(f"✅ **Strong edge detected.** Net PF {net_pf} with {result.win_rate}% win rate.")
    elif net_pf > 1:
        st.info(f"➖ **Marginal edge.** Net PF {net_pf} — consider with regime filter for better results.")
    else:
        st.warning(f"⚠️ **No edge in this period.** Net PF {net_pf} — strategy may need different market conditions.")
    
    c1,c2 = st.columns(2)
    with c1: pc("Avg Win", f"+{result.avg_win_pct:.1f}%", "g"); pc("Best Trade", f"{result.best_trade_pct:+.1f}%", "g")
    with c2: pc("Avg Loss", f"-{result.avg_loss_pct:.1f}%", "r"); pc("Worst Trade", f"{result.worst_trade_pct:+.1f}%", "r")
    
    # Equity curve (Net P&L)
    if result.equity_curve:
        import plotly.graph_objects as go
        eq_df = pd.DataFrame(result.equity_curve)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_df["date"], y=eq_df["equity"], mode="lines+markers",
                                  name="Net P&L %", line=dict(color="#FF6B35", width=2),
                                  marker=dict(size=5)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(template="plotly_dark", title="Equity Curve (Net P&L % after costs)",
                          xaxis_title="Date", yaxis_title="Cumulative Net P&L %",
                          height=350, margin=dict(t=40, b=30, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade log
    with st.expander(f"📋 Trade Log ({result.total_trades} trades)", expanded=False):
        rows = []
        for t in result.trades:
            net_pnl_t = getattr(t, "net_pnl_pct", t.pnl_pct)
            pnl_color = "🟢" if net_pnl_t > 0 else "🔴"
            rows.append({
                "": pnl_color,
                "Symbol": t.symbol,
                "Entry Date": t.entry_date,
                "Entry ₹": fp(t.entry_price),
                "Exit Date": t.exit_date,
                "Exit ₹": fp(t.exit_price),
                "Gross P&L": f"{t.pnl_pct:+.1f}%",
                "Net P&L": f"{net_pnl_t:+.1f}%",
                "Cost": f"{getattr(t, 'cost_pct', 0):.2f}%",
                "Hold": f"{t.holding_days}d",
                "Exit Reason": t.exit_reason,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================================
# SIGNAL LOG — Daily signal recording
# ============================================================================
def page_signal_log():
    st.markdown("# 📋 Signal Log")
    st.caption("Every scan auto-records signals here. Deduped by Strategy + Symbol per day.")
    
    # Date selector
    dates = get_signal_dates()
    if not dates:
        st.info("No signals recorded yet. Run a scan from Dashboard or Scanner Hub — signals auto-save here.")
        st.markdown("""
        **How it works:**
        1. Every time you run a scan (Dashboard or Scanner Hub), signals are automatically saved with timestamp
        2. GitHub Actions runs EOD scans daily at 4:30 PM and 7:00 PM IST (if configured)
        3. Duplicate signals (same strategy + stock on same day) are ignored
        """)
        return
    
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_date = st.selectbox("Date", dates, key="sl_date",
            format_func=lambda d: f"{d} {'(today)' if d == date.today().isoformat() else ''}")
    with c2:
        if st.button("🔄 Update Tracker", key="sl_update", help="Check if any OPEN signals hit SL or T1"):
            if st.session_state.stock_data:
                updated = update_open_signals_live(st.session_state.stock_data)
                st.success(f"Updated {updated} signals" if updated else "No changes — all signals unchanged")
                st.rerun()
            else:
                st.warning("Load data from Dashboard first to update live prices.")
    
    df = load_signals(selected_date)
    if df is None or df.empty:
        st.info(f"No signals for {selected_date}.")
        return
    
    # Summary
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: pc("Total Signals", str(len(df)))
    with c2: pc("Strategies", str(df["Strategy"].nunique()))
    with c3: pc("Stocks", str(df["Symbol"].nunique()))
    open_c = len(df[df["Status"]=="OPEN"])
    with c4: pc("Open", str(open_c), "o" if open_c > 0 else "")
    regime_val = df["Regime"].iloc[0] if "Regime" in df.columns and len(df) > 0 else "-"
    with c5: pc("Regime", str(regime_val))
    
    # Grouped by strategy
    for strat in df["Strategy"].unique():
        sdf = df[df["Strategy"] == strat]
        p = STRATEGY_PROFILES.get(strat, {})
        name = p.get("name", strat)
        icon = p.get("icon", "")
        
        with st.expander(f"{icon} {name} — {len(sdf)} signals", expanded=True):
            display_cols = ["Time","Symbol","Signal","CMP","Entry","SL","T1","RR",
                          "Confidence","RS","Sector","Regime_Fit","Status"]
            cols_present = [c for c in display_cols if c in sdf.columns]
            st.dataframe(sdf[cols_present], use_container_width=True, hide_index=True)
    
    # Download
    st.markdown("---")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 Download {selected_date} signals (CSV)", csv_data,
                       f"signals_{selected_date}.csv", "text/csv", key="sl_dl")


# ============================================================================
# SIGNAL TRACKER — Forward testing performance
# ============================================================================
def page_tracker():
    st.markdown("# 📊 Signal Tracker")
    st.caption("Forward-test: did scanner signals actually hit their targets?")
    
    tracker_df = load_tracker()
    if tracker_df is None or tracker_df.empty:
        st.info("No tracked signals yet. Run scans to start recording, then check back to see outcomes.")
        st.markdown("""
        **How forward testing works:**
        1. Every scan auto-records signals with Entry, SL, and T1 prices
        2. Each day, the system checks if price hit SL (loss) or T1 (win)
        3. After 30 days, unresolved signals expire at current price
        4. Over time, this builds a real track record for each strategy
        """)
        return
    
    stats = compute_tracker_stats(tracker_df)
    if not stats:
        st.info("Insufficient data for statistics.")
        return
    
    # Header metrics
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: pc("Total Signals", str(stats["total"]))
    with c2: pc("🟢 Targets Hit", str(stats["targets"]), "g")
    with c3: pc("🔴 Stopped Out", str(stats["stopped"]), "r")
    with c4: pc("⏳ Open", str(stats["open"]), "o")
    with c5: pc("Win Rate", f"{stats['win_rate']}%", "g" if stats["win_rate"] > 50 else "r")
    with c6: pc("Total P&L", f"{stats['total_pnl']:+.1f}%", "g" if stats["total_pnl"] > 0 else "r")
    
    # Update button
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Refresh Tracker (check SL/T1 hits)", key="tr_refresh", type="primary"):
            if st.session_state.stock_data:
                updated = update_open_signals_live(st.session_state.stock_data)
                st.success(f"Updated {updated} signals") if updated else st.info("No changes")
                st.rerun()
            else:
                st.warning("Load data from Dashboard first.")
    with c2:
        st.caption(f"Period: {stats.get('first_date','')} to {stats.get('last_date','')}")
    with c3:
        st.caption(f"Trading days: {stats.get('trading_days',0)}")
    
    # Strategy performance table
    st.markdown("### 📊 Strategy Performance (Forward Test)")
    strat_rows = []
    for strat, s in stats.get("strategy_stats", {}).items():
        p = STRATEGY_PROFILES.get(strat, {})
        strat_rows.append({
            "Strategy": f"{p.get('icon','')} {p.get('name',strat)}",
            "Total": s["total"],
            "Open": s["open"],
            "🟢 Target": s["targets"],
            "🔴 Stopped": s["stopped"],
            "Win Rate": f"{s['win_rate']}%" if s["targets"] + s["stopped"] > 0 else "-",
        })
    if strat_rows:
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)
    
    # Detailed signal table with filters
    st.markdown("### 📋 All Tracked Signals")
    c1, c2, c3 = st.columns(3)
    with c1:
        status_filter = st.multiselect("Status", ["OPEN","TARGET","STOPPED","EXPIRED"],
                                        default=["OPEN","TARGET","STOPPED"], key="tr_status")
    with c2:
        strat_options = list(tracker_df["Strategy"].unique())
        strat_filter = st.multiselect("Strategy", strat_options, default=strat_options, key="tr_strat")
    with c3:
        signal_filter = st.multiselect("Signal Type", ["BUY","SHORT"],
                                        default=["BUY","SHORT"], key="tr_signal")
    
    filtered = tracker_df[
        (tracker_df["Status"].isin(status_filter)) &
        (tracker_df["Strategy"].isin(strat_filter)) &
        (tracker_df["Signal"].isin(signal_filter))
    ].copy()
    
    # Color code status
    display_cols = ["Date","Time","Strategy_Name","Symbol","Signal","Entry","SL","T1",
                   "Confidence","RS","Status","Exit_Date","Exit_Reason","PnL_Pct"]
    cols_present = [c for c in display_cols if c in filtered.columns]
    
    st.dataframe(filtered[cols_present].sort_values("Date", ascending=False),
                 use_container_width=True, hide_index=True)
    
    # Download full tracker
    st.markdown("---")
    csv_bytes = generate_csv_download()
    if csv_bytes:
        st.download_button("📥 Download Full Signal History (CSV)", csv_bytes,
                           f"signal_tracker_{date.today().isoformat()}.csv", "text/csv",
                           key="tr_dl")


# ============================================================================
# SETTINGS
# ============================================================================
def page_settings():
    st.markdown("# ⚙️ Settings")
    
    # Auth status
    from auth_manager import get_current_user, get_current_profile, is_admin
    _su = get_current_user()
    _sp = get_current_profile() or {}
    if _su:
        st.success(f"✅ Signed in as **{_su.get('name',_su.get('email',''))}** "
                   f"· Plan: **{_sp.get('plan','free').upper()}**")
        st.caption(f"User ID: {_su.get('id','?')[:8]}...")
    st.divider()
    
    st.session_state.capital = st.number_input("Capital ₹", value=st.session_state.capital, step=50000)

    st.markdown("### 🔌 Breeze API")
    if st.session_state.breeze_connected:
        st.success("✅ Breeze Connected! Intraday scanners + Option Chain + Volume Profile are LIVE.")
    else:
        if st.session_state.breeze_msg:
            msg_lower = st.session_state.breeze_msg.lower()
            if "session" in msg_lower or "token" in msg_lower or "expired" in msg_lower:
                st.error("🔑 Breeze session token expired — update it below (no Streamlit secrets needed).")
            else:
                st.error(st.session_state.breeze_msg)
        st.markdown("**Without Breeze:** ORB, VWAP, Lunch Low, Option Chain, Volume Profile are **disabled**. "
                     "VCP, EMA21, 52WH, Short, ATH work fine with daily data.")

    # ── Daily token update — THE ONLY THING YOU DO EACH MORNING ──────────
    st.markdown("#### 🔑 Update Session Token (do this each morning)")
    st.caption(
        "Generate a new token from ICICIDirect.com → My Profile → API → Generate Token. "
        "Paste it here — no Streamlit secrets update needed, no GitHub secrets needed."
    )
    current_token = _get_breeze_token_from_supabase()
    with st.form("token_update_form"):
        new_token = st.text_input(
            "New Session Token",
            value="",
            placeholder="Paste today's token here (e.g. 55163439)",
            type="password",
            help="Get from ICICIDirect → My Profile → Generate Session Token"
        )
        submitted = st.form_submit_button("✅ Update Token & Reconnect", type="primary", use_container_width=True)
        if submitted:
            if new_token.strip():
                ok = update_breeze_token_in_supabase(new_token.strip())
                if ok:
                    st.success("✅ Token saved. Reconnecting...")
                    # Reset state so try_breeze() runs fresh
                    st.session_state.breeze_connected = False
                    st.session_state.breeze_msg = ""
                    st.session_state.breeze_engine = None
                    try_breeze()
                    if st.session_state.breeze_connected:
                        st.success("✅ Breeze connected successfully!")
                    else:
                        st.error(f"Connection failed: {st.session_state.breeze_msg}")
                        st.info("The token was saved. Try clicking 'Retry Breeze' in the sidebar, or check the token is today's value.")
                    st.rerun()
                else:
                    # Supabase not connected yet — fall back to direct connect
                    e = BreezeEngine()
                    api_key = ""
                    api_secret = ""
                    try:
                        api_key    = st.secrets.get("BREEZE_API_KEY", "")
                        api_secret = st.secrets.get("BREEZE_API_SECRET", "")
                    except Exception:
                        pass
                    if api_key and api_secret:
                        ok2, msg = e.connect(api_key, api_secret, new_token.strip())
                        if ok2:
                            st.success("✅ Breeze connected! (Supabase not set up yet — token only saved locally for this session)")
                            st.session_state.breeze_connected = True
                            st.session_state.breeze_engine = e
                            st.rerun()
                        else:
                            st.error(f"Token rejected: {msg}")
                    else:
                        st.error("BREEZE_API_KEY and BREEZE_API_SECRET must be in Streamlit secrets.")
            else:
                st.warning("Please paste a token before submitting.")

    if current_token:
        st.caption(f"Current token in Supabase: {current_token[:4]}••••{current_token[-4:] if len(current_token) > 4 else ''}")
    else:
        st.warning("⚠️ No token found in Supabase. Paste one above and click Update.")

    # ── Diagnostic test (collapsible) ──────────────────────────────────────
    with st.expander("🔍 Diagnose connection (tap if Breeze not connecting)"):
        if st.button("Run diagnostic", key="breeze_diag"):
            st.markdown("**Step 1** — Reading token from Supabase...")
            diag_token = _get_breeze_token_from_supabase()
            if diag_token:
                st.success(f"✅ Supabase returned token: {diag_token[:4]}••••{diag_token[-4:]}")
            else:
                st.error("❌ Supabase returned empty token. Check SUPABASE_URL and SUPABASE_SERVICE_KEY are in Streamlit secrets.")
            
            st.markdown("**Step 2** — Reading API key/secret from Streamlit secrets...")
            try:
                _ak = st.secrets.get("BREEZE_API_KEY", "")
                _as = st.secrets.get("BREEZE_API_SECRET", "")
                if _ak and _as:
                    st.success(f"✅ API Key: {_ak[:6]}•••• | Secret: {_as[:6]}••••")
                else:
                    st.error("❌ BREEZE_API_KEY or BREEZE_API_SECRET missing from Streamlit secrets")
                    _ak, _as = "", ""
            except Exception as ex:
                st.error(f"❌ Cannot read secrets: {ex}")
                _ak, _as = "", ""
            
            if diag_token and _ak and _as:
                st.markdown("**Step 3** — Attempting Breeze connection...")
                try:
                    e_test = BreezeEngine()
                    ok_test, msg_test = e_test.connect(_ak, _as, diag_token)
                    if ok_test:
                        st.success(f"✅ Breeze connected! Setting session state...")
                        st.session_state.breeze_connected = True
                        st.session_state.breeze_engine = e_test
                        st.session_state.breeze_msg = "Connected via diagnostic"
                        st.rerun()
                    else:
                        st.error(f"❌ Connection failed: {msg_test}")
                        if "session" in msg_test.lower() or "invalid" in msg_test.lower():
                            st.info("💡 This means the token itself is wrong or expired. Get a fresh token from ICICIDirect → My Profile → API → Generate Token")
                        elif "key" in msg_test.lower():
                            st.info("💡 API key or secret seems wrong. Double-check them in Streamlit secrets.")
                except Exception as ex:
                    st.error(f"❌ Exception during connect: {type(ex).__name__}: {ex}")

    with st.expander("🔧 Advanced — Manual Connect with custom credentials"):
        with st.form("bf"):
            ak = st.text_input("Key", type="password")
            asc = st.text_input("Secret", type="password")
            st_ = st.text_input("Token", type="password")
            if st.form_submit_button("Connect"):
                if ak and asc and st_:
                    e = BreezeEngine(); ok, msg = e.connect(ak, asc, st_)
                    if ok: st.success(msg); st.session_state.breeze_connected=True; st.session_state.breeze_engine=e
                    else: st.error(msg)

    st.session_state.universe_size = st.selectbox("Universe", ["nifty50","nifty200","nifty500"],
        index=["nifty50","nifty200","nifty500"].index(st.session_state.universe_size))

    st.markdown("### 🔐 Access Control")
    st.markdown("Add this to Streamlit secrets to require a password:")
    st.code('APP_PASSWORD = "your_chosen_password"', language="toml")
    st.caption("Leave blank or remove to disable.")

    st.markdown("### 🤖 GitHub Actions (Auto-Scanner)")
    st.markdown("""
    Auto-scans run daily via GitHub Actions (**free** on public repos):
    - **4:30 PM IST** — Quick Nifty 200 scan
    - **7:00 PM IST** — Full NSE 500 EOD scan + signal tracking
    - Results auto-committed to `signals/` folder in your repo
    """)

    st.markdown("### 🧠 Regime Behavior (Adaptive Heat Caps)")
    st.markdown("""
    | Regime | Position Size | Heat Cap | Risk/Trade | Ideal For |
    |--------|---:|---:|---:|---|
    | 🟢 EXPANSION | 100% | 6% | 2.0% | VCP, 52WH, ORB, ATH — buy breakouts |
    | 🟡 ACCUMULATION | 70% | 4% | 1.5% | VCP, EMA21, VWAP — be selective |
    | 🟠 DISTRIBUTION | 40% | 2% | 1.0% | Shorts, mean-reversion — defensive |
    | 🔴 PANIC | 15% | 1% | 0.5% | Shorts only — protect capital |
    """)

    st.markdown("### 🗄️ Supabase (Data Persistence)")
    sb = _get_supabase()
    if sb:
        st.success("✅ Supabase Connected — signals, portfolio, and auto-learning are active")
    else:
        st.warning("⚠️ Supabase not configured. Add to Streamlit Secrets:")
        st.code('SUPABASE_URL = "https://aiebaqvclyzxajigvkfd.supabase.co"\nSUPABASE_SERVICE_KEY = "your_service_role_key"', language="toml")
        st.markdown("""
**Your project is already created:**
- URL: `https://aiebaqvclyzxajigvkfd.supabase.co`
- Get the service_role key from: supabase.com → nse-scanner-pro → Project Settings → API
- Paste both in Streamlit Cloud → App Settings → Secrets
        """)

    st.markdown("### 💹 Telegram Setup")
    st.caption("Alerts are sent via Telegram when scans run. Set in Streamlit Secrets:")
    st.code('TELEGRAM_BOT_TOKEN = "your_bot_token"\nTELEGRAM_CHAT_ID = "your_chat_id"', language="toml")

    # Per-user alert preferences
    st.divider()
    render_alert_preferences()


# ============================================================================
# STOCK LOOKUP — Deep-dive per stock
# ============================================================================
def page_stock_lookup():
    st.markdown("# 🔎 Stock Lookup")
    st.caption("Deep-dive into any stock — indicators, scanner verdict, signal history, and RRG sector status.")

    enriched = st.session_state.get("enriched_data") or st.session_state.get("stock_data") or {}
    all_symbols = sorted(enriched.keys()) if enriched else []

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("🔍 Search stock (e.g. ASHOKLEY, RELIANCE, TATAMOTORS)",
                              key="sl_query", placeholder="Type stock symbol...").strip().upper()
    with c2:
        if all_symbols:
            selected = st.selectbox("Or select from loaded", [""] + all_symbols, key="sl_select")
            if selected: query = selected

    if not query:
        st.info("Enter a stock symbol above. Load data from Dashboard first.")
        if not st.session_state.get("data_loaded"):
            st.warning("⚠️ Load data from Dashboard first to enable lookup.")
        return

    sym = query.replace(".NS", "")
    if sym not in enriched:
        matches = [s for s in all_symbols if sym in s]
        if matches:
            if len(matches) == 1:
                sym = matches[0]
            else:
                sym = st.selectbox(f"Multiple matches for '{sym}':", matches, key="sl_fuzzy")
        else:
            st.error(f"**{sym}** not found in loaded universe ({len(all_symbols)} stocks). "
                     "Try loading a larger universe (Nifty 500) from Dashboard.")
            return

    df = enriched[sym]
    if df.empty or len(df) < 10:
        st.warning(f"Insufficient data for {sym}.")
        return

    lat = df.iloc[-1]
    sector = get_sector(sym)
    fno_tag = get_fno_tag(sym)
    cmp = lat["close"]

    st.markdown(f"## {sym} {fno_tag}")
    st.caption(f"Sector: **{sector}** | Data: {len(df)} bars | Last: {df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])}")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: pc("CMP", f"₹{cmp:,.1f}")
    with c2:
        chg_1d = ((cmp / df.iloc[-2]["close"]) - 1) * 100 if len(df) > 1 else 0
        pc("1D Change", f"{chg_1d:+.1f}%", "g" if chg_1d > 0 else "r")
    with c3: pc("RSI (14)", f"{lat.get('rsi_14', 0):.0f}", "r" if lat.get('rsi_14', 50) > 70 else ("g" if lat.get('rsi_14', 50) < 30 else ""))
    with c4:
        vol_ratio = lat.get("volume", 0) / lat.get("vol_sma_20", 1) if lat.get("vol_sma_20", 0) > 0 else 0
        pc("Vol Ratio", f"{vol_ratio:.1f}x", "g" if vol_ratio > 1.5 else "")
    with c5: pc("52W High", f"₹{lat.get('high_52w', 0):,.1f}")
    with c6:
        pct_52 = lat.get("pct_from_52w_high", 0)
        pc("From 52WH", f"{pct_52:+.1f}%", "g" if pct_52 > -5 else ("r" if pct_52 < -25 else ""))

    with st.expander("📊 Technical Indicators", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown("**Moving Averages**")
            st.markdown(f"SMA 20: ₹{lat.get('sma_20', 0):,.1f}")
            st.markdown(f"SMA 50: ₹{lat.get('sma_50', 0):,.1f}")
            st.markdown(f"SMA 200: ₹{lat.get('sma_200', 0):,.1f}")
            st.markdown(f"EMA 21: ₹{lat.get('ema_21', 0):,.1f}")
            above_20 = cmp > lat.get("sma_20", 0)
            above_50 = cmp > lat.get("sma_50", 0)
            above_200 = cmp > lat.get("sma_200", 0)
            alignment = sum([above_20, above_50, above_200])
            st.markdown(f"**MA Alignment: {alignment}/3** {'🟢' if alignment == 3 else '🟡' if alignment >= 2 else '🔴'}")
        with c2:
            st.markdown("**Momentum**")
            st.markdown(f"RSI 14: {lat.get('rsi_14', 0):.1f}")
            st.markdown(f"MACD: {lat.get('macd', 0):.2f}")
            st.markdown(f"MACD Signal: {lat.get('macd_signal', 0):.2f}")
            macd_bull = lat.get("macd", 0) > lat.get("macd_signal", 0)
            st.markdown(f"MACD Cross: {'🟢 Bullish' if macd_bull else '🔴 Bearish'}")
        with c3:
            st.markdown("**Volatility**")
            st.markdown(f"ATR 14: ₹{lat.get('atr_14', 0):.1f}")
            st.markdown(f"ADX: {lat.get('adx_14', 0):.1f}")
            st.markdown(f"BB Upper: ₹{lat.get('bb_upper', 0):,.1f}")
            st.markdown(f"BB Lower: ₹{lat.get('bb_lower', 0):,.1f}")
        with c4:
            st.markdown("**Volume**")
            st.markdown(f"Volume: {lat.get('volume', 0):,.0f}")
            st.markdown(f"Avg 20d: {lat.get('vol_sma_20', 0):,.0f}")
            st.markdown(f"Vol Ratio: {vol_ratio:.2f}x")
            st.markdown(f"52W Low: ₹{lat.get('low_52w', 0):,.1f}")

    # Weekly alignment
    mtf = check_weekly_alignment(df)
    if mtf["aligned"]:
        st.success(f"✅ Weekly timeframe confirms ({mtf['score']}/4)")
    else:
        st.warning(f"⚠️ Weekly not aligned ({mtf['score']}/4)")

    # RRG sector status
    rrg = st.session_state.get("rrg_data", {})
    if rrg and sector in rrg:
        rrg_s = rrg[sector]
        q = rrg_s.get("quadrant", "N/A")
        q_icon = {"LEADING":"🟢","WEAKENING":"🟡","IMPROVING":"🔵","LAGGING":"🔴"}.get(q, "⚪")
        st.info(f"Sector **{sector}** is {q_icon} **{q}** (RRG Score: {rrg_s.get('score', 0)})")

    # Chart
    fig = plot_candlestick(df, sym, days=90)
    st.plotly_chart(fig, use_container_width=True)

    # Scanner verdict
    st.markdown("### 🎯 Scanner Verdict")
    st.caption("Running all daily strategies on this stock right now.")
    nifty_df = st.session_state.get("nifty_data")
    hits = []
    for scanner_name, scanner_func in DAILY_SCANNERS.items():
        try:
            result = scanner_func(df, sym, nifty_df)
            if result:
                hits.append(result)
        except Exception:
            pass

    if hits:
        st.success(f"**{sym} qualifies in {len(hits)} strateg{'y' if len(hits)==1 else 'ies'}!**")
        for r in hits:
            p = STRATEGY_PROFILES.get(r.strategy, {})
            st.markdown(f"**{p.get('icon','')} {p.get('name', r.strategy)}** — {r.signal} | "
                        f"Entry ₹{r.entry:,.1f} | SL ₹{r.stop_loss:,.1f} | T1 ₹{r.target_1:,.1f}")
            for reason in r.reasons[:3]:
                st.markdown(f"  • {reason}")
    else:
        st.info(f"**{sym}** doesn't qualify in any daily strategy right now.")

    # ── Gemini News Analysis (on-demand, cached per symbol per day) ──────────────
    if PERPLEXITY_AVAILABLE:
        import os as _os
        _gemini_key = _os.environ.get("GEMINI_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
        if _gemini_key:
            with st.expander("📰 Why is this stock moving? (AI News Analysis)", expanded=True):
                # Use the signal direction from latest signal if available, else neutral
                _sig_dir = "BUY"
                _strategy = "Technical"
                if 'hits' in dir() and hits:
                    _sig_dir = hits[0].signal
                    _strategy = hits[0].strategy
                render_signal_context_card(sym, _strategy, _sig_dir)
        else:
            with st.expander("📰 AI News Analysis (not configured)"):
                st.caption("Add `GEMINI_API_KEY` in Streamlit Secrets to enable live news analysis for any stock.")
                st.code('GEMINI_API_KEY = "AIza..."', language="toml")

    # Signal history for this stock
    st.markdown("### 📜 Signal History")
    all_signals = load_signals()
    if all_signals is not None and not all_signals.empty:
        stock_hist = all_signals[all_signals["Symbol"] == sym].copy()
        if not stock_hist.empty:
            stock_hist = stock_hist.sort_values("Date", ascending=False)
            first = stock_hist.iloc[-1]
            first_entry = first.get("Entry", 0)
            pnl_since = ((cmp / first_entry) - 1) * 100 if first_entry > 0 else 0
            c1, c2, c3, c4 = st.columns(4)
            with c1: pc("First Flagged", str(first.get("Date", "?")))
            with c2: pc("Entry Then", f"₹{first_entry:,.1f}")
            with c3: pc("CMP Now", f"₹{cmp:,.1f}")
            with c4: pc("Since Signal", f"{pnl_since:+.1f}%", "g" if pnl_since > 0 else "r")
            st.dataframe(stock_hist[["Date","Strategy","Signal","CMP","Entry","SL","T1","Confidence","RS","Status"]].head(20),
                         use_container_width=True, hide_index=True)
        else:
            st.info(f"No historical signals recorded for {sym}.")
    else:
        st.info("No signal history yet.")


# ============================================================================
# SIGNAL HISTORY — Cross-stock signal tracking & drift detection
# ============================================================================
def _fmt_ist(ts_str: str) -> str:
    """Convert any timestamp string to IST display format."""
    if not ts_str:
        return "—"
    try:
        import pytz
        from datetime import datetime as _dt
        IST = pytz.timezone("Asia/Kolkata")
        if isinstance(ts_str, str):
            ts = _dt.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            ts = ts_str
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts)
        return ts.astimezone(IST).strftime("%d %b %H:%M IST")
    except Exception:
        return str(ts_str)[:16]


def page_signal_history():
    st.markdown("# 📜 Signal History")
    st.caption("Every signal ever generated — when first detected, how many times re-detected, and entry drift. All times in IST.")

    sb = _get_supabase()

    # ── Load from Supabase directly (shows IST first_seen + scan_count) ──────
    supa_rows = []
    if sb:
        try:
            res = (sb.table("signals")
                   .select("date,strategy,symbol,signal,entry,sl,t1,cmp,sqi_grade,sector,status,first_seen_ist,last_seen_ist,scan_count,pnl_pct")
                   .order("first_seen_ist", desc=True)
                   .limit(500)
                   .execute())
            supa_rows = res.data or []
        except Exception as e:
            st.warning(f"Supabase load failed: {e}")

    if not supa_rows:
        all_signals = load_signals()
        if all_signals is None or all_signals.empty:
            st.info("No signals recorded yet. Run scans to start building history.")
            return
        # Fallback: show CSV-based history
        st.caption("⚠️ Showing CSV cache — Supabase data unavailable.")
        st.dataframe(all_signals, use_container_width=True, hide_index=True)
        return

    # ── Summary stats ─────────────────────────────────────────────────────────
    unique_stocks = len(set(r["symbol"] for r in supa_rows))
    total_signals = len(supa_rows)
    re_detected   = sum(1 for r in supa_rows if (r.get("scan_count") or 1) > 1)
    c1, c2, c3, c4 = st.columns(4)
    with c1: pc("Unique Stocks", str(unique_stocks))
    with c2: pc("Total Signals", str(total_signals))
    with c3: pc("Re-detected", str(re_detected))
    with c4: pc("Open", str(sum(1 for r in supa_rows if r.get("status") == "OPEN")))

    # ── Filter bar ────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_status = st.selectbox("Status", ["ALL", "OPEN", "TARGET", "STOPPED"], index=0)
    with col_f2:
        filter_redet = st.checkbox("Show re-detected only (scan_count > 1)", value=False)

    filtered = supa_rows
    if filter_status != "ALL":
        filtered = [r for r in filtered if r.get("status") == filter_status]
    if filter_redet:
        filtered = [r for r in filtered if (r.get("scan_count") or 1) > 1]

    # ── Main table ────────────────────────────────────────────────────────────
    st.markdown(f"### 📋 Signal Log ({len(filtered)} signals)")
    rows = []
    for r in filtered:
        cnt   = r.get("scan_count") or 1
        grade = r.get("sqi_grade", "—")
        pnl   = r.get("pnl_pct")
        rows.append({
            "Symbol":       r["symbol"],
            "Strategy":     r.get("strategy",""),
            "Signal":       r.get("signal",""),
            "First Seen (IST)": _fmt_ist(r.get("first_seen_ist", r.get("date",""))),
            "Last Seen (IST)":  _fmt_ist(r.get("last_seen_ist", r.get("date",""))),
            "Seen×":        cnt,
            "Entry ₹":      r.get("entry",""),
            "SL ₹":         r.get("sl",""),
            "T1 ₹":         r.get("t1",""),
            "Grade":        grade,
            "Sector":       r.get("sector",""),
            "Status":       r.get("status",""),
            "P&L%":         f"{float(pnl):+.1f}%" if pnl is not None else "—",
        })

    if rows:
        result_df = pd.DataFrame(rows)
        # Colour-code re-detected rows
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    # ── Entry drift detection ─────────────────────────────────────────────────
    st.markdown("### ⚠️ Entry Drift — Same Stock, Price Changed")
    st.caption("When the same stock+strategy appears on different days with a changed entry price (>5% drift).")
    by_stock_strat: dict = {}
    for r in supa_rows:
        key = (r["symbol"], r.get("strategy",""))
        if key not in by_stock_strat:
            by_stock_strat[key] = []
        by_stock_strat[key].append(r)

    drift_found = False
    for (sym, strat), recs in by_stock_strat.items():
        entries = [float(r["entry"]) for r in recs if r.get("entry")]
        if len(entries) >= 2:
            first_e, last_e = entries[0], entries[-1]
            if abs(first_e - last_e) / (first_e + 1) > 0.05:
                drift = ((last_e / first_e) - 1) * 100
                first_ts = _fmt_ist(recs[0].get("first_seen_ist", ""))
                last_ts  = _fmt_ist(recs[-1].get("last_seen_ist", ""))
                st.warning(
                    f"**{sym} — {strat}:** ₹{first_e:,.1f} ({first_ts}) → ₹{last_e:,.1f} ({last_ts}) "
                    f"| Drift: {drift:+.1f}% | Seen {len(recs)}×"
                )
                drift_found = True

    if not drift_found:
        st.success("No significant entry drift detected across your signals.")


# ============================================================================
# OPTION CHAIN PAGE — 7-Factor Composite Scoring
# ============================================================================
def page_option_chain():
    st.markdown("# 🔗 Option Chain Analysis")
    st.caption("7-factor composite scoring for F&O stocks via Breeze API.")

    if not OPTION_CHAIN_AVAILABLE:
        st.error("option_chain.py module not found.")
        return

    if not st.session_state.breeze_connected:
        st.warning("🔌 Option Chain requires Breeze API. Connect Breeze in Settings → Update Session Token.")
        return

    be = st.session_state.breeze_engine
    if not be:
        st.error("Breeze engine not initialized.")
        return

    # Market hours note — Breeze returns last available data even after hours
    _ist_now = now_ist()
    from datetime import time as _dtime
    _is_market = _dtime(9,15) <= _ist_now.time() <= _dtime(15,30) and _ist_now.weekday() < 5
    if not _is_market:
        st.info(
            f"Market closed ({_ist_now.strftime('%I:%M %p IST')}). "
            f"Showing last available option chain data. OI and PCR figures reflect closing positions — "
            f"useful for planning next session's trades."
        )

    # Allow using any F&O symbol by typing — don't require data load first
    enriched = st.session_state.get("enriched_data") or st.session_state.get("stock_data") or {}
    fno_stocks_loaded = sorted([s for s in enriched.keys() if is_fno(s)])

    # Build the complete F&O list (don't require data load)
    from fno_list import FNO_STOCKS
    all_fno = sorted(FNO_STOCKS)
    fno_stocks = fno_stocks_loaded if fno_stocks_loaded else all_fno

    if not fno_stocks:
        st.info("No F&O stocks available. Load data from Dashboard or the full list will appear here.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        oc_mode = st.radio("Mode", ["Single Stock", "Batch Scan"], horizontal=True)
    with c2:
        if oc_mode == "Single Stock":
            oc_sym = st.selectbox("F&O Stock", fno_stocks, key="oc_sym")
    with c3:
        if oc_mode == "Batch Scan":
            batch_limit = st.number_input("Max stocks to scan", 5, 50, 20)

    if oc_mode == "Single Stock":
        if st.button("🔗 Analyze Option Chain", type="primary", key="oc_run_single"):
            with st.spinner(f"Fetching option chain for {oc_sym}..."):
                try:
                    oc_data = be.fetch_option_chain(oc_sym)
                    if oc_data:
                        calls_df = oc_data.get("calls")
                        puts_df  = oc_data.get("puts")
                        if (calls_df is not None and not calls_df.empty) or \
                           (puts_df  is not None and not puts_df.empty):
                            result = analyze_option_chain(oc_sym, oc_data, enriched.get(oc_sym))
                            _render_oc_result(result)
                            st.session_state.oc_results[oc_sym] = result
                        else:
                            st.warning(f"Option chain fetched but no strike data for {oc_sym}. "
                                       "The expiry may have no open positions, or the session token needs renewal.")
                            with st.expander("🔍 Debug: raw fetch result"):
                                st.json({k: str(v) for k, v in oc_data.items() if k not in ("calls","puts")})
                    else:
                        from breeze_symbol_map import to_breeze_code
                        breeze_code = to_breeze_code(oc_sym)
                        st.warning(f"No option chain data returned for **{oc_sym}** (Breeze code: `{breeze_code}`).")
                        
                        # Real diagnostic: test NIFTY option chain (always works if token valid)
                        with st.expander("🔍 Diagnose connection", expanded=True):
                            st.caption(f"Testing with NIFTY (most reliable F&O instrument)...")
                            try:
                                from datetime import date, timedelta
                                today = date.today()
                                # Get next Thursday
                                days_to_thu = (3 - today.weekday()) % 7 or 7
                                next_thu = today + timedelta(days=days_to_thu)
                                test_expiry = next_thu.strftime("%Y-%m-%dT06:00:00.000Z")
                                test_resp = be.breeze.get_option_chain_quotes(
                                    stock_code="NIFTY", exchange_code="NFO",
                                    product_type="options", expiry_date=test_expiry,
                                    right="call", strike_price="0"
                                )
                                if test_resp and test_resp.get("Status") == 200 and test_resp.get("Success"):
                                    n_rows = len(test_resp["Success"])
                                    st.success(f"✅ Breeze token is valid — NIFTY returned {n_rows} strikes.")
                                    st.error(f"Problem is specific to **{oc_sym}** (Breeze code: `{breeze_code}`). "
                                             f"This stock may not have weekly options or the Breeze code mapping may be wrong.")
                                    st.info(f"Known working F&O stocks: RELIANCE (→RELIND), HDFCBANK (→HDFWA2), INFY (→INFY), TCS (→TCS). "
                                            f"Try one of these to confirm.")
                                else:
                                    err = test_resp.get("Error", "unknown") if test_resp else "no response"
                                    st.error(f"❌ Breeze token appears expired or invalid. NIFTY test failed: {err}")
                                    st.warning("**Fix:** Go to Settings → Update Session Token → paste today's token from ICICIDirect portal.")
                            except Exception as te:
                                st.error(f"❌ Breeze API error: {te}")
                                st.warning("Breeze session is not working. Go to Settings to update the token.")
                except Exception as e:
                    st.error(f"Option chain error: {e}")
                    import traceback
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc())

    else:
        if st.button("🔗 Batch Option Chain Scan", type="primary", key="oc_run_batch"):
            results = []
            progress = st.progress(0)
            scan_stocks = fno_stocks[:int(batch_limit)]
            for i, sym in enumerate(scan_stocks):
                progress.progress((i+1)/len(scan_stocks), f"Scanning {sym}...")
                try:
                    oc_data = be.fetch_option_chain(sym)
                    if oc_data:
                        result = analyze_option_chain(sym, oc_data, enriched.get(sym))
                        results.append(result)
                        st.session_state.oc_results[sym] = result
                except Exception:
                    pass
            progress.empty()

            if results:
                strong_buys = [r for r in results if r.score >= 75]
                buys = [r for r in results if 60 <= r.score < 75]
                st.success(f"Scan complete: {len(strong_buys)} Strong Buy, {len(buys)} Buy signals from {len(results)} stocks")

                if strong_buys:
                    st.markdown("### 🟢 Strong Buy (Score ≥ 75)")
                    rows = [_oc_summary_row(r) for r in sorted(strong_buys, key=lambda x: -x.score)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if buys:
                    st.markdown("### 🟡 Buy (Score 60-75)")
                    rows = [_oc_summary_row(r) for r in sorted(buys, key=lambda x: -x.score)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                sells = [r for r in results if r.score < 45]
                if sells:
                    with st.expander(f"🔴 Sell/Strong Sell Signals ({len(sells)})"):
                        rows = [_oc_summary_row(r) for r in sorted(sells, key=lambda x: x.score)]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No option chain data retrieved. Check Breeze connection.")

    # Show cached results
    if st.session_state.oc_results:
        with st.expander(f"📋 Session Results ({len(st.session_state.oc_results)} stocks analyzed)"):
            summary_rows = [_oc_summary_row(r) for r in st.session_state.oc_results.values()]
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _oc_summary_row(r):
    """Build a summary row dict for option chain result."""
    signal_emoji = {"STRONG BUY":"🟢","BUY":"🟡","NEUTRAL":"⚪","SELL":"🟠","STRONG SELL":"🔴"}.get(r.signal, "⚪")
    return {
        "Symbol": r.symbol,
        "Signal": f"{signal_emoji} {r.signal}",
        "Score": f"{r.score:.0f}/100",
        "Confidence": f"{r.confidence:.0f}%",
        "IV %ile": f"{getattr(r, 'ivp', 0):.0f}%",
        "PCR": f"{getattr(r, 'pcr', 0):.2f}",
        "F&O Ban": "⚠️ YES" if getattr(r, 'fno_ban', False) else "No",
        "DTE": str(getattr(r, 'dte', "?")),
    }


def _render_oc_result(r):
    """Render detailed option chain result for a single stock."""
    signal_map = {
        "STRONG BUY": ("🟢", "success"),
        "BUY": ("🟡", "info"),
        "NEUTRAL": ("⚪", "info"),
        "SELL": ("🟠", "warning"),
        "STRONG SELL": ("🔴", "error"),
    }
    emoji, level = signal_map.get(r.signal, ("⚪", "info"))
    getattr(st, level)(f"{emoji} **{r.symbol}** — {r.signal} | Score: {r.score:.0f}/100 | Confidence: {r.confidence:.0f}%")

    if getattr(r, 'fno_ban', False):
        st.error("⚠️ F&O BAN: This stock is at >95% MWPL. No new positions allowed.")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: pc("Composite Score", f"{r.score:.0f}")
    with c2: pc("Confidence", f"{r.confidence:.0f}%")
    with c3: pc("IV Percentile", f"{getattr(r, 'ivp', 0):.0f}%")
    with c4: pc("PCR", f"{getattr(r, 'pcr', 0):.2f}")
    with c5: pc("Max Pain", fp(getattr(r, 'max_pain', 0)))
    with c6: pc("DTE", str(getattr(r, 'dte', "?")))

    if hasattr(r, "components") and r.components:
        with st.expander("📊 Factor Breakdown"):
            comp_rows = []
            for factor, score in r.components.items():
                comp_rows.append({"Factor": factor, "Score": f"{score:.1f}/100"})
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    if hasattr(r, "reasons") and r.reasons:
        st.markdown("**Why this signal:**")
        for reason in r.reasons:
            st.markdown(f"  • {reason}")


# ============================================================================
# IPO SCANNER PAGE — IPO Base Detection
# ============================================================================
def page_ipo_scanner():
    st.markdown("# 🚀 IPO Scanner")
    st.caption("Detects IPO base formations and breakout signals. "
               "O'Neil study: 57% hit rate, +2.79% alpha on high-volume breakouts (≥150% of 50-day avg). "
               "Tracks lock-up expiry alerts: Day 30/90/180/540.")

    if not IPO_SCANNER_AVAILABLE:
        st.error("ipo_scanner.py module not found.")
        return

    tab1, tab2, tab3 = st.tabs(["🔍 Scan IPO Universe", "⚠️ Lock-up Alerts", "📖 IPO Strategy Guide"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Scans Mainboard + SME IPOs for base formations and breakout signals.")
            include_months = st.slider("Months of history", 3, 36, 24)
        with c2:
            min_score = st.slider("Min Quality Score", 0, 80, 0)
        with c3:
            seg_filter = st.selectbox("Segment", ["All", "Mainboard", "SME"])

        if st.button("🔍 Scan IPO Universe", type="primary", key="ipo_scan"):
            with st.spinner("Analyzing IPOs (uses yfinance — no NSE scraping needed)..."):
                try:
                    results = scan_ipo_universe(
                        max_listing_months=include_months,
                        min_score=min_score,
                        segment_filter=seg_filter,
                        breeze_engine=st.session_state.breeze_engine
                    )
                    st.session_state.ipo_results = results
                except Exception as e:
                    st.error(f"IPO scan error: {e}")
                    st.session_state.ipo_results = []

            if st.session_state.ipo_results:
                st.success(f"Found {len(st.session_state.ipo_results)} IPO setups")

        results = st.session_state.get("ipo_results", [])
        if results:
            strong_buys = [r for r in results if r.score >= 80]
            buys = [r for r in results if 60 <= r.score < 80]
            watches = [r for r in results if 40 <= r.score < 60]

            for label, subset, color in [
                ("🟢 Strong Buy (Score ≥ 80)", strong_buys, "success"),
                ("🟡 Buy (Score 60-79)", buys, "info"),
                ("👀 Watch (Score 40-59)", watches, None),
            ]:
                if not subset: continue
                st.markdown(f"### {label}")
                rows = []
                for r in sorted(subset, key=lambda x: -x.score):
                    rows.append({
                        "Symbol": r.symbol,
                        "Company": r.company_name[:20] if hasattr(r,'company_name') else "",
                        "Segment": getattr(r, "segment", ""),
                        "Score": f"{r.score:.0f}/100",
                        "Signal": r.signal,
                        "CMP": fp(getattr(r, "cmp", 0)),
                        "Issue Price": fp(getattr(r, "issue_price", 0)),
                        "Gain%": f"{getattr(r,'gain_since_listing',0):+.1f}%",
                        "Listing Date": str(getattr(r, "listing_date", "")),
                        "Base": f"{getattr(r,'base_depth_pct',0):.0f}%d/{getattr(r,'base_days',0)}d",
                        "RS": str(getattr(r, "rs_rating", 0)),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            if not st.session_state.get("ipo_scan_run"):
                st.info("Click 'Scan IPO Universe' to find IPO base formations.")

    with tab2:
        st.markdown("### ⚠️ Upcoming Lock-up Expiry Alerts")
        st.caption("Lock-up expirations create significant supply events. Day 30 (anchor): 76% of stocks decline avg 2.6%.")
        if st.button("🔍 Check Lock-up Events", key="lockup_scan"):
            with st.spinner("Checking IPO lock-up calendar..."):
                try:
                    alerts = get_upcoming_lock_up_alerts(days_ahead=30)
                    if alerts:
                        st.warning(f"⚠️ {len(alerts)} lock-up events in next 30 days")
                        rows = []
                        for a in alerts:
                            rows.append({
                                "Symbol": a.get("symbol",""),
                                "Event": a.get("event",""),
                                "Date": str(a.get("date","")),
                                "Days Until": str(a.get("days_until","")),
                                "Action": a.get("action",""),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.success("No major lock-up events in next 30 days.")
                except Exception as e:
                    st.error(f"Lock-up scan error: {e}")

    with tab3:
        st.markdown("""
### 📖 IPO Base Strategy — Research Summary

**What is an IPO Base?**
After listing, IPOs typically consolidate 15-30% below their first-week high for 10-14 days minimum.
This is the "IPO base" — a healthy pause before the next leg up.

**Entry Rules (O'Neil methodology):**
- Close > left-side high of the IPO base (the pivot)
- Volume ≥ 150% of 50-day average on breakout day
- RS Rating ≥ 80 (stock must outperform Nifty since listing)
- Enter within 5% of the pivot (not extended)
- Stop loss: 7-8% below entry

**8-Week Hold Rule:**
If the stock gains ≥20% within 3 weeks of breakout, hold for 8 full weeks minimum.
This rule captures the biggest winners.

**Indian Market Lock-up Calendar:**
| Event | Day | Average Impact |
|-------|-----|---------------|
| Anchor investor unlock | 30 | -2.6% avg (76% of stocks decline) |
| Public shareholder unlock | 90 | Supply increases 50% |
| PE/VC investor unlock | 180 | -5 to -6% avg drag |
| Promoter unlock | 540 | Largest supply event |

**Risk Management:**
- Max 0.75% portfolio risk per IPO trade
- Max 5% position size per IPO
- Max 25% total IPO allocation (3-5 positions max)

**Win Rate:** 57% | **Average Winner:** +19.9% | **Average Loser:** -14.5% (favorable R:R)
        """)


# ============================================================================
# ROUTER
# ============================================================================
page_map = {
    "📊 Dashboard": page_dashboard,
    "🔍 Scanner Hub": page_scanner_hub,
    "📈 Charts & RS": page_charts_rs,
    "🔗 Option Chain": page_option_chain,
    "🚀 IPO Scanner": page_ipo_scanner,
    "🔎 Stock Lookup": page_stock_lookup,
    "📜 Signal History": page_signal_history,
    "🧪 Backtest": page_backtest,
    "📊 Performance": page_performance,
    "🎮 Virtual Game": (render_paper_trading_page if PAPER_TRADING_AVAILABLE else lambda: st.warning("paper_trading.py not found in repo")),
    "🧠 AI Deep Dive": (page_ai_deep_dive if AI_DEEP_DIVE_AVAILABLE else lambda: st.error(f"AI Deep Dive load failed: {_AI_DEEP_DIVE_ERROR}")),
    "⚙️ Settings": page_settings,
    "👑 Admin": render_admin_dashboard,
}
page_func = page_map.get(page, page_dashboard)
if page not in _NAV_DIVIDERS:
    page_func()