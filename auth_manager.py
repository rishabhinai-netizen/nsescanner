"""
auth_manager.py v2 — NSE Scanner Pro
=====================================
FIXES v2:
- Google OAuth: graceful fallback when provider not enabled
- Test endpoint to check if Google is enabled before showing button
- Better error messages
- Complete admin privilege system
- Cleaner UI with proper plan badges
"""

import streamlit as st
import requests as _req
from typing import Optional, Dict, Any
import json, time


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def _get_supabase_url() -> str:
    try: return st.secrets.get("SUPABASE_URL", "")
    except: return ""

def _get_anon_key() -> str:
    try: return st.secrets.get("SUPABASE_ANON_KEY", "")
    except: return ""

def _get_service_key() -> str:
    try: return st.secrets.get("SUPABASE_SERVICE_KEY", "")
    except: return ""

def _is_auth_enabled() -> bool:
    try: return st.secrets.get("AUTH_ENABLED", "false").lower() == "true"
    except: return False


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE CLIENTS
# ══════════════════════════════════════════════════════════════════════════════

def _client_user():
    """Client using current user's session."""
    try:
        from supabase import create_client
        url, key = _get_supabase_url(), _get_anon_key()
        if not url or not key: return None
        client = create_client(url, key)
        sess = st.session_state.get("_auth_session")
        if sess and sess.get("access_token"):
            try:
                client.auth.set_session(
                    access_token=sess["access_token"],
                    refresh_token=sess.get("refresh_token", "")
                )
            except Exception: pass
        return client
    except Exception as e:
        print(f"[Auth] user client error: {e}")
        return None

def _client_admin():
    """Client with service role — bypasses RLS. Use only for admin ops."""
    try:
        from supabase import create_client
        url, key = _get_supabase_url(), _get_service_key()
        if not url or not key: return None
        return create_client(url, key)
    except Exception as e:
        print(f"[Auth] admin client error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init():
    for k, v in {
        "_auth_session": None,
        "_auth_user": None,
        "_auth_profile": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_current_user() -> Optional[Dict]:
    _init()
    return st.session_state.get("_auth_user")

def get_current_profile() -> Optional[Dict]:
    _init()
    return st.session_state.get("_auth_profile")

def get_user_id() -> Optional[str]:
    u = get_current_user()
    return u["id"] if u else None

def is_admin() -> bool:
    p = get_current_profile()
    return bool(p and p.get("plan") == "admin")

def is_pro() -> bool:
    p = get_current_profile()
    return bool(p and p.get("plan") in ("pro", "admin"))

def is_authenticated() -> bool:
    return get_current_user() is not None

def _load_profile(uid: str) -> Optional[Dict]:
    try:
        sb = _client_user()
        if not sb: return None
        res = sb.table("user_profiles").select("*").eq("id", uid).single().execute()
        return res.data
    except Exception:
        return None

def _save_session(session_data: Dict):
    st.session_state["_auth_session"] = session_data
    user = session_data.get("user") or {}
    st.session_state["_auth_user"] = {
        "id":     user.get("id"),
        "email":  user.get("email"),
        "name":   (user.get("user_metadata") or {}).get("full_name",
                   user.get("email", "").split("@")[0]),
        "avatar": (user.get("user_metadata") or {}).get("avatar_url", ""),
    }
    uid = user.get("id")
    if uid:
        st.session_state["_auth_profile"] = _load_profile(uid)

def logout():
    try:
        sb = _client_user()
        if sb: sb.auth.sign_out()
    except Exception: pass
    for k in ["_auth_session", "_auth_user", "_auth_profile"]:
        st.session_state[k] = None
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL AUTH
# ══════════════════════════════════════════════════════════════════════════════

def sign_up_email(email: str, password: str, name: str = "") -> Dict:
    try:
        sb = _client_user()
        if not sb: return {"error": "Supabase not configured. Check SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit Secrets."}
        res = sb.auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"full_name": name or email.split("@")[0]}}
        })
        if res.user:
            return {"success": True, "message": "✅ Account created! Check your email to verify."}
        return {"error": "Sign-up failed. Try a different email."}
    except Exception as e:
        err = str(e)
        if "already registered" in err.lower() or "unique" in err.lower():
            return {"error": "Email already registered. Sign in instead."}
        return {"error": f"Sign-up error: {err[:120]}"}

def sign_in_email(email: str, password: str) -> Dict:
    try:
        sb = _client_user()
        if not sb: return {"error": "Supabase not configured."}
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        if res.session:
            _save_session({
                "access_token":  res.session.access_token,
                "refresh_token": res.session.refresh_token,
                "user": {
                    "id":            res.user.id,
                    "email":         res.user.email,
                    "user_metadata": res.user.user_metadata or {},
                }
            })
            return {"success": True}
        return {"error": "Invalid email or password."}
    except Exception as e:
        err = str(e)
        if "invalid" in err.lower() or "credentials" in err.lower() or "wrong" in err.lower():
            return {"error": "Wrong email or password. Use 'Forgot password?' to reset."}
        if "not confirmed" in err.lower() or "email" in err.lower() and "confirm" in err.lower():
            return {"error": "Email not verified. Check your inbox and click the verification link."}
        return {"error": f"Login error: {err[:120]}"}

def reset_password(email: str) -> Dict:
    try:
        sb = _client_user()
        if not sb: return {"error": "Supabase not configured."}
        redirect = "https://nsescannerbyrishabh.streamlit.app"
        try: redirect = st.secrets.get("APP_URL", redirect)
        except: pass
        sb.auth.reset_password_email(email, options={"redirect_to": redirect})
        return {"success": True, "message": f"✅ Password reset link sent to {email}. Check your inbox."}
    except Exception as e:
        return {"error": f"Reset failed: {str(e)[:100]}"}


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE OAUTH — with provider check
# ══════════════════════════════════════════════════════════════════════════════

def _is_google_enabled() -> bool:
    """Check if Google OAuth is enabled in Supabase by probing the endpoint."""
    try:
        url = _get_supabase_url()
        if not url: return False
        # Quick probe: GET the auth settings endpoint
        r = _req.get(f"{url}/auth/v1/settings", timeout=5,
                     headers={"apikey": _get_anon_key()})
        if r.status_code == 200:
            data = r.json()
            providers = data.get("external", {})
            return providers.get("google", {}).get("enabled", False)
        return False
    except Exception:
        return False

def get_google_oauth_url() -> Optional[str]:
    """Returns Google OAuth URL only if the provider is actually enabled."""
    try:
        if not _is_google_enabled():
            return None
        url = _get_supabase_url()
        if not url: return None
        redirect = "https://nsescannerbyrishabh.streamlit.app"
        try: redirect = st.secrets.get("APP_URL", redirect)
        except: pass
        return (
            f"{url}/auth/v1/authorize"
            f"?provider=google"
            f"&redirect_to={redirect}"
        )
    except Exception:
        return None

def handle_oauth_callback():
    """Process OAuth token from URL params on page load."""
    try:
        params = st.query_params
        access_token  = params.get("access_token")
        refresh_token = params.get("refresh_token", "")
        # Security: immediately clear token from URL to prevent exposure in history/logs
        if access_token:
            try: st.query_params.clear()
            except Exception: pass
        if access_token and not is_authenticated():
            sb = _client_user()
            if sb:
                res = sb.auth.set_session(access_token, refresh_token)
                if res.user:
                    _save_session({
                        "access_token":  access_token,
                        "refresh_token": refresh_token,
                        "user": {
                            "id":            res.user.id,
                            "email":         res.user.email,
                            "user_metadata": res.user.user_metadata or {},
                        }
                    })
                    st.query_params.clear()
                    st.rerun()
    except Exception as e:
        print(f"[Auth] OAuth callback error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def update_profile(updates: Dict) -> bool:
    uid = get_user_id()
    if not uid: return False
    try:
        sb = _client_user()
        if not sb: return False
        sb.table("user_profiles").update(updates).eq("id", uid).execute()
        st.session_state["_auth_profile"] = _load_profile(uid)
        return True
    except Exception as e:
        print(f"[Auth] profile update error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# ALERT DELIVERY
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram_alert(user_id: str, message: str, signal_id=None,
                        symbol="", strategy="", alert_type="SIGNAL") -> bool:
    try:
        sb = _client_admin()
        if not sb: return False
        res = sb.table("user_profiles").select(
            "telegram_chat_id,telegram_bot_token,telegram_alerts,alert_min_sqi,alert_strategies"
        ).eq("id", user_id).single().execute()
        if not res.data: return False
        p = res.data
        if not p.get("telegram_alerts"): return False
        chat_id   = p.get("telegram_chat_id")
        bot_token = p.get("telegram_bot_token")
        if not bot_token:
            try: bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
            except: bot_token = ""
        if not chat_id or not bot_token: return False
        resp = _req.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        success = resp.status_code == 200
        sb.table("user_alerts").insert({
            "user_id": user_id, "signal_id": signal_id,
            "symbol": symbol, "strategy": strategy,
            "alert_type": alert_type, "channel": "telegram",
            "message": message[:500], "success": success,
            "error_msg": resp.text[:200] if not success else None,
        }).execute()
        return success
    except Exception as e:
        print(f"[Auth] Telegram error: {e}")
        return False

def broadcast_signal_alert(signal: Dict, sqi: float = 0) -> int:
    """Send signal to all subscribed users. Called from auto_scanner."""
    try:
        sb = _client_admin()
        if not sb: return 0
        strategy = signal.get("strategy", "")
        symbol   = signal.get("symbol", "")
        res = sb.table("user_profiles").select(
            "id,telegram_alerts,email_alerts,alert_strategies,alert_min_sqi"
        ).or_("telegram_alerts.eq.true,email_alerts.eq.true").execute()
        if not res.data: return 0
        count = 0
        for user in res.data:
            uid      = user["id"]
            min_sqi  = user.get("alert_min_sqi") or 50
            if sqi < min_sqi: continue
            strats = user.get("alert_strategies") or []
            if strats and strategy not in strats: continue
            msg = _format_alert(signal)
            if user.get("telegram_alerts"):
                if send_telegram_alert(uid, msg, symbol=symbol,
                                       strategy=strategy, alert_type="SIGNAL"):
                    count += 1
        return count
    except Exception as e:
        print(f"[Auth] broadcast error: {e}")
        return 0

def _format_alert(s: Dict) -> str:
    sig_icon  = "🟢 BUY" if s.get("signal") == "BUY" else "🔴 SHORT"
    sqi_icon  = {"ELITE":"🏆","STRONG":"⭐","MODERATE":"✅","WEAK":"⚠️"}.get(s.get("sqi_grade",""),"")
    return (
        f"<b>🎯 NSE Scanner Pro</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<b>{sig_icon}: {s.get('symbol','')}</b>\n"
        f"Strategy : {s.get('strategy','')}\n"
        f"CMP      : ₹{float(s.get('cmp',0)):,.2f}\n"
        f"Entry    : ₹{float(s.get('entry',0)):,.2f}\n"
        f"SL       : ₹{float(s.get('sl',0)):,.2f}\n"
        f"T1       : ₹{float(s.get('t1',0)):,.2f}\n"
        f"T2       : ₹{float(s.get('t2',0)):,.2f}\n"
        f"R:R      : 1:{s.get('rr',0)}\n"
        f"{sqi_icon} SQI: {s.get('sqi',0)}/100 ({s.get('sqi_grade','')})\n"
        f"Regime: {s.get('regime','')} · Sector: {s.get('sector','')}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<i>NSE Scanner Pro</i>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE UI
# ══════════════════════════════════════════════════════════════════════════════

_AUTH_CSS = """
<style>
.auth-wrap{max-width:400px;margin:0 auto;padding:0 16px}
.auth-logo{text-align:center;font-size:2rem;font-weight:800;margin:24px 0 20px;
  background:linear-gradient(120deg,#FF6B35,#00d4ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.auth-card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:28px 28px 20px}
.auth-tab{display:flex;gap:0;margin-bottom:20px;border:1px solid #333;border-radius:8px;overflow:hidden}
.auth-tab-btn{flex:1;padding:8px;text-align:center;font-size:.85rem;cursor:pointer;
  background:#1a1d23;color:#888;border:none}
.auth-tab-btn.active{background:#FF6B35;color:#fff;font-weight:600}
.g-btn{display:block;text-align:center;padding:11px;background:#fff;
  border:1px solid #dadce0;border-radius:8px;color:#3c4043;font-size:.9rem;
  text-decoration:none;font-weight:500;margin-bottom:14px;transition:box-shadow .2s}
.g-btn:hover{box-shadow:0 2px 8px rgba(0,0,0,.3)}
.divider{text-align:center;color:#555;margin:14px 0;font-size:.78rem;
  display:flex;align-items:center;gap:8px}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:#2d2d2d}
.plan-chip{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.68rem;font-weight:600}
.chip-free{background:#1a2332;color:#5dade2;border:1px solid #1e3a5f}
.chip-pro{background:#1a2d1a;color:#00d26a;border:1px solid #1b5e20}
.chip-admin{background:#3d2200;color:#ff9800;border:1px solid #7d4800}
</style>
"""

def render_auth_page():
    handle_oauth_callback()
    if is_authenticated(): return

    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-logo">🎯 NSE Scanner Pro</div>', unsafe_allow_html=True)

    mode = st.session_state.get("_auth_mode", "login")
    if mode == "login":
        _render_login()
    elif mode == "signup":
        _render_signup()
    elif mode == "forgot":
        _render_forgot()


def _google_button(label: str):
    """Show Google OAuth button only if provider is enabled in Supabase."""
    oauth_url = get_google_oauth_url()
    if oauth_url:
        st.markdown(
            f'<a href="{oauth_url}" target="_self" class="g-btn">'
            f'<img src="https://www.google.com/favicon.ico" width="16" '
            f'style="vertical-align:middle;margin-right:8px">{label}</a>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="divider">or continue with email</div>', unsafe_allow_html=True)
        return True
    return False


def _render_login():
    st.markdown("#### 👋 Welcome back")
    _google_button("Sign in with Google")

    with st.form("login_form", clear_on_submit=False):
        email    = st.text_input("Email address", placeholder="you@example.com", key="li_email")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="li_pass")
        submit   = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

    if submit:
        if not email or not password:
            st.error("Enter your email and password.")
        else:
            with st.spinner("Signing in…"):
                result = sign_in_email(email.strip().lower(), password)
            if result.get("success"):
                st.rerun()
            else:
                st.error(result.get("error", "Sign in failed."))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Create account →", use_container_width=True):
            st.session_state["_auth_mode"] = "signup"; st.rerun()
    with c2:
        if st.button("Forgot password?", use_container_width=True):
            st.session_state["_auth_mode"] = "forgot"; st.rerun()

    st.markdown(
        '<div style="text-align:center;margin-top:18px;font-size:.75rem;color:#555">'
        '<span class="plan-chip chip-free">FREE</span> '
        'Full access during beta · No credit card needed</div>',
        unsafe_allow_html=True
    )


def _render_signup():
    st.markdown("#### 🚀 Create your account")
    _google_button("Sign up with Google")

    with st.form("signup_form", clear_on_submit=False):
        name     = st.text_input("Your name", placeholder="Rishabh", key="su_name")
        email    = st.text_input("Email address", placeholder="you@example.com", key="su_email")
        password = st.text_input("Password", type="password",
                                 placeholder="At least 8 characters", key="su_pass")
        confirm  = st.text_input("Confirm password", type="password", key="su_confirm")
        st.caption("🔒 Per-user data isolation — your signals and trades are private.")
        submit   = st.form_submit_button("Create Account →", use_container_width=True, type="primary")

    if submit:
        if not email or not password:
            st.error("Email and password are required.")
        elif len(password) < 8:
            st.error("Password must be at least 8 characters.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            with st.spinner("Creating your account…"):
                result = sign_up_email(email.strip().lower(), password, name.strip())
            if result.get("success"):
                st.success(result["message"])
                st.info("👉 After clicking the verification link in your email, come back and sign in.")
            else:
                st.error(result.get("error"))

    if st.button("← Already have an account? Sign in", use_container_width=True):
        st.session_state["_auth_mode"] = "login"; st.rerun()


def _render_forgot():
    st.markdown("#### 🔑 Reset your password")
    st.caption("Enter your email and we'll send a reset link.")

    with st.form("forgot_form", clear_on_submit=False):
        email  = st.text_input("Email address", placeholder="you@example.com", key="fp_email")
        submit = st.form_submit_button("Send Reset Link →", use_container_width=True, type="primary")

    if submit:
        if not email:
            st.error("Please enter your email address.")
        else:
            with st.spinner("Sending reset link…"):
                result = reset_password(email.strip().lower())
            if result.get("success"):
                st.success(result["message"])
            else:
                st.error(result.get("error"))

    if st.button("← Back to sign in", use_container_width=True):
        st.session_state["_auth_mode"] = "login"; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ALERT PREFERENCES (shown in Settings page)
# ══════════════════════════════════════════════════════════════════════════════

def render_alert_preferences():
    user    = get_current_user()
    profile = get_current_profile() or {}
    if not user:
        st.warning("Sign in to configure personal alerts.")
        return

    st.markdown("### 🔔 Alert Preferences")
    st.caption("Alerts are sent per-user — each account gets alerts only for their chosen strategies at their chosen SQI threshold.")

    # Telegram
    st.markdown("#### 📱 Telegram")
    with st.expander("How to set up your personal Telegram bot", expanded=False):
        st.markdown("""
1. Open Telegram → search **@BotFather** → send `/newbot`
2. Choose a name → it gives you a **bot token** (`1234567890:AAF...`)
3. Open your new bot → send it any message (e.g. "hi")
4. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Find `"chat"` → `"id"` in the JSON — that's your **Chat ID**
6. Paste both below and save
        """)
    c1, c2 = st.columns(2)
    with c1:
        tg_token = st.text_input("Bot Token", value=profile.get("telegram_bot_token",""),
                                  type="password", placeholder="110201543:AAHdqTcvCH...", key="tg_tok")
    with c2:
        tg_chat = st.text_input("Chat ID", value=profile.get("telegram_chat_id",""),
                                 placeholder="123456789", key="tg_chat")
    tg_on = st.toggle("Enable Telegram alerts", value=bool(profile.get("telegram_alerts")), key="tg_on")
    if tg_token and tg_chat:
        if st.button("📤 Test Telegram", key="tg_test"):
            r = _req.post(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                          json={"chat_id": tg_chat,
                                "text": "✅ NSE Scanner Pro — Telegram connected! You'll receive alerts here."},
                          timeout=10)
            if r.status_code == 200: st.success("✅ Test message sent!")
            else: st.error(f"Failed: {r.json().get('description','Unknown error')}")

    # Email
    st.markdown("#### 📧 Email")
    email_on = st.toggle(f"Email alerts to **{user.get('email','')}**",
                          value=bool(profile.get("email_alerts")), key="email_on")

    # Filters
    st.markdown("#### 🎯 Alert Filters")
    try:
        from scanners import STRATEGY_PROFILES
        all_strats = list(STRATEGY_PROFILES.keys())
        sel_strats = st.multiselect(
            "Strategies to alert on (empty = all)",
            options=all_strats,
            default=profile.get("alert_strategies") or [],
            format_func=lambda x: f"{STRATEGY_PROFILES[x]['icon']} {STRATEGY_PROFILES[x]['name']}",
            key="alert_strats"
        )
    except Exception:
        sel_strats = []
        st.info("Load scanner to configure strategy filters.")

    min_sqi = st.slider("Minimum SQI to alert", 0, 90,
                         int(profile.get("alert_min_sqi") or 50), 5, key="alert_sqi",
                         help="Only send alerts when SQI ≥ this. 0 = all signals. 65+ = STRONG/ELITE only.")

    if st.button("💾 Save Alert Settings", type="primary", key="save_alert_prefs"):
        ok = update_profile({
            "telegram_bot_token": tg_token,
            "telegram_chat_id":   tg_chat,
            "telegram_alerts":    tg_on,
            "email_alerts":       email_on,
            "alert_strategies":   sel_strats,
            "alert_min_sqi":      min_sqi,
        })
        if ok: st.success("✅ Saved!")
        else:  st.error("Save failed — check Supabase connection.")

    # Alert history
    st.markdown("#### 📜 Alert History (last 20)")
    try:
        sb = _client_user()
        if sb:
            res = sb.table("user_alerts").select(
                "sent_at,symbol,strategy,alert_type,channel,success"
            ).eq("user_id", user["id"]).order("sent_at", desc=True).limit(20).execute()
            if res.data:
                import pandas as pd
                df = pd.DataFrame(res.data)
                df["sent_at"] = pd.to_datetime(df["sent_at"]).dt.strftime("%d %b %H:%M")
                df["success"] = df["success"].map({True:"✅", False:"❌"})
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No alerts sent yet.")
    except Exception: st.caption("Alert history unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_admin_dashboard():
    """
    Admin-only page. Admins can:
    1. See all registered users + their usage
    2. Promote/demote user plans
    3. See platform-wide signal stats
    4. Broadcast announcements
    5. View alert delivery stats
    """
    if not is_admin():
        st.error("🔒 Admin access required.")
        return

    st.markdown("# 👑 Admin Dashboard")
    st.caption("Platform management — visible only to admin accounts.")

    sb = _client_admin()
    if not sb:
        st.error("Admin client not available — check SUPABASE_SERVICE_KEY in Streamlit Secrets.")
        return

    # ── Overview metrics ────────────────────────────────────────────────────
    try:
        users_res   = sb.table("user_profiles").select("id,plan,telegram_alerts,email_alerts,created_at,last_seen_at").execute()
        signals_res = sb.table("signals").select("id,status,strategy,created_at").execute()
        alerts_res  = sb.table("user_alerts").select("id,success,channel").execute()
        users   = users_res.data   or []
        signals = signals_res.data or []
        alerts  = alerts_res.data  or []

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("👥 Total Users",   len(users))
        c2.metric("📡 Total Signals", len(signals))
        c3.metric("🔔 Alerts Sent",   len([a for a in alerts if a.get("success")]))
        c4.metric("📱 TG Subscribers",len([u for u in users if u.get("telegram_alerts")]))
        c5.metric("📧 Email Subs",    len([u for u in users if u.get("email_alerts")]))
    except Exception as e:
        st.error(f"Stats error: {e}")

    st.divider()

    # ── User management ──────────────────────────────────────────────────────
    st.markdown("### 👥 Users")
    try:
        res = sb.table("user_profiles").select(
            "id,email,display_name,plan,telegram_alerts,email_alerts,alert_min_sqi,created_at,last_seen_at"
        ).order("created_at", desc=True).execute()

        if res.data:
            import pandas as pd
            df = pd.DataFrame(res.data)
            df["created_at"]  = pd.to_datetime(df["created_at"]).dt.strftime("%d %b %Y")
            df["last_seen_at"] = pd.to_datetime(df["last_seen_at"]).dt.strftime("%d %b %H:%M IST") if "last_seen_at" in df else "-"
            df["telegram_alerts"] = df["telegram_alerts"].map({True:"✅",False:"—"})
            df["email_alerts"]    = df["email_alerts"].map({True:"✅",False:"—"})
            st.dataframe(df[["email","display_name","plan","telegram_alerts","email_alerts","alert_min_sqi","created_at","last_seen_at"]],
                         use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Users table error: {e}")

    # ── Plan management ──────────────────────────────────────────────────────
    st.markdown("### 🔧 Manage User Plan")
    st.caption("Upgrade/downgrade any user's plan. Admin = full access. Pro = premium features. Free = standard.")
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        target_email = st.text_input("User email", key="admin_email")
    with cm2:
        new_plan = st.selectbox("New plan", ["free", "pro", "admin"], key="admin_plan")
    with cm3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Update Plan", key="admin_update", type="primary"):
            if target_email:
                try:
                    sb.table("user_profiles").update({"plan": new_plan}).eq("email", target_email).execute()
                    st.success(f"✅ {target_email} → {new_plan}")
                except Exception as e:
                    st.error(f"Update failed: {e}")

    st.divider()

    # ── Signal stats ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Signal Stats (All Users)")
    try:
        res2 = sb.table("signals").select(
            "strategy,signal,status,sqi_grade,regime,created_at"
        ).order("created_at", desc=True).limit(500).execute()
        if res2.data:
            import pandas as pd
            df2 = pd.DataFrame(res2.data)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("**By Strategy**")
                st.dataframe(df2.groupby("strategy").size().sort_values(ascending=False).rename("Count"),
                             use_container_width=True)
            with col_s2:
                st.markdown("**By Status**")
                st.dataframe(df2.groupby("status").size().rename("Count"),
                             use_container_width=True)
    except Exception as e:
        st.error(f"Signal stats error: {e}")

    st.divider()

    # ── Broadcast announcement ────────────────────────────────────────────────
    st.markdown("### 📣 Broadcast Telegram Message")
    st.caption("Send a message to ALL users who have Telegram alerts enabled.")
    bcast_msg = st.text_area("Message (supports HTML tags: <b>, <i>)",
                              placeholder="<b>NSE Scanner Pro Update:</b>\nNew feature released: AI Deep Dive v4 is live!",
                              key="bcast_msg", height=100)
    if st.button("📣 Send to All Telegram Subscribers", key="bcast_send", type="primary"):
        if not bcast_msg.strip():
            st.error("Enter a message first.")
        else:
            with st.spinner("Broadcasting…"):
                try:
                    users_res2 = sb.table("user_profiles").select(
                        "id,telegram_bot_token,telegram_chat_id,telegram_alerts"
                    ).eq("telegram_alerts", True).execute()
                    sent = 0
                    for u in (users_res2.data or []):
                        tok  = u.get("telegram_bot_token","") or ""
                        chat = u.get("telegram_chat_id","") or ""
                        if not tok:
                            try: tok = st.secrets.get("TELEGRAM_BOT_TOKEN","")
                            except: pass
                        if tok and chat:
                            r = _req.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                                          json={"chat_id": chat, "text": bcast_msg, "parse_mode": "HTML"},
                                          timeout=10)
                            if r.status_code == 200: sent += 1
                    st.success(f"✅ Broadcast sent to {sent} user(s).")
                except Exception as e:
                    st.error(f"Broadcast error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PLAN PRIVILEGES (used throughout app)
# ══════════════════════════════════════════════════════════════════════════════

def check_plan_feature(feature: str) -> bool:
    """
    Returns True if current user's plan allows the feature.
    
    Features:
      "ai_deep_dive"        — AI Deep Dive page          → free+
      "paper_trading"       — Virtual Game                → free+
      "option_chain"        — Option Chain analysis       → free+
      "ipo_scanner"         — IPO Scanner                 → free+
      "backtest"            — Strategy backtesting        → free+
      "unlimited_signals"   — No scan limit               → pro+
      "alert_history"       — Full alert history          → pro+
      "admin_dashboard"     — Admin management UI         → admin only
      "broadcast"           — Broadcast to all users      → admin only
      "plan_management"     — Change user plans           → admin only
    """
    plan = (get_current_profile() or {}).get("plan", "free")
    
    FREE_FEATURES = {
        "ai_deep_dive", "paper_trading", "option_chain",
        "ipo_scanner", "backtest", "basic_alerts",
    }
    PRO_FEATURES  = FREE_FEATURES | {
        "unlimited_signals", "alert_history", "advanced_charts",
        "export_signals", "sector_rrg",
    }
    ADMIN_FEATURES = PRO_FEATURES | {
        "admin_dashboard", "broadcast", "plan_management",
        "view_all_users", "platform_stats",
    }
    
    if plan == "admin":  return feature in ADMIN_FEATURES
    if plan == "pro":    return feature in PRO_FEATURES
    return feature in FREE_FEATURES


def plan_gate(feature: str, show_upgrade: bool = True) -> bool:
    """
    Returns True if allowed. If not, shows upgrade message.
    Usage:
        if not plan_gate("unlimited_signals"): return
    """
    if check_plan_feature(feature): return True
    if show_upgrade:
        plan = (get_current_profile() or {}).get("plan", "free")
        if feature in ("admin_dashboard", "broadcast", "plan_management"):
            st.error("🔒 Admin access required.")
        else:
            st.warning(
                f"⭐ **Pro feature** — '{feature}' requires a Pro plan. "
                f"Your current plan: **{plan.upper()}**. "
                f"Contact the admin to upgrade."
            )
    return False


# ══════════════════════════════════════════════════════════════════════════════
# AUTH GATE
# ══════════════════════════════════════════════════════════════════════════════

def require_auth() -> bool:
    """
    Call at top of app. Returns True = proceed. False = show auth page.
    Skips auth if AUTH_ENABLED != "true" in secrets (dev mode).
    """
    _init()
    handle_oauth_callback()

    if not _is_auth_enabled():
        return True  # Dev mode

    if not _get_supabase_url() or not _get_anon_key():
        return True  # Supabase not configured

    if is_authenticated():
        # Update last_seen
        try:
            uid = get_user_id()
            if uid:
                sb = _client_user()
                if sb:
                    sb.table("user_profiles").update(
                        {"last_seen_at": "now()"}
                    ).eq("id", uid).execute()
        except Exception: pass
        return True

    render_auth_page()
    return False
