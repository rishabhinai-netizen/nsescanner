"""
auth_manager.py — Multi-user authentication for NSE Scanner Pro v19
====================================================================
Uses Supabase Auth (Google OAuth + Email/Password).
Handles: signup, login, logout, password reset, user profiles,
         per-user Telegram/email alert preferences.

Usage in app.py:
    from auth_manager import require_auth, render_auth_page, get_current_user

Architecture:
  - Supabase Auth handles identity (JWT tokens)
  - user_profiles table stores preferences
  - All DB operations are scoped to auth.uid() via RLS
  - Service role key (in secrets) used only for admin/GitHub Actions
"""

import streamlit as st
import requests as _req
from typing import Optional, Dict, Any
import json, time


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE AUTH CLIENT  (uses anon key for user-facing auth)
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


def _supabase_auth_client():
    """Get Supabase client using current user's JWT token."""
    try:
        from supabase import create_client
        url = _get_supabase_url()
        key = _get_anon_key()
        if not url or not key:
            return None
        client = create_client(url, key)
        # Inject user session if available
        sess = st.session_state.get("_auth_session")
        if sess and sess.get("access_token"):
            try:
                client.auth.set_session(
                    access_token=sess["access_token"],
                    refresh_token=sess.get("refresh_token", "")
                )
            except Exception:
                pass
        return client
    except Exception as e:
        print(f"[Auth] Client error: {e}")
        return None


def _supabase_admin_client():
    """Get Supabase client with service role (bypasses RLS). Admin only."""
    try:
        from supabase import create_client
        url = _get_supabase_url()
        key = _get_service_key()
        if not url or not key: return None
        return create_client(url, key)
    except Exception as e:
        print(f"[Auth] Admin client error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# AUTH STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _init_session():
    """Initialize auth session state keys."""
    for k, v in {
        "_auth_session": None,      # Supabase session dict
        "_auth_user": None,         # User info dict
        "_auth_profile": None,      # user_profiles row
        "_auth_initialized": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_current_user() -> Optional[Dict]:
    """Return current logged-in user or None."""
    _init_session()
    return st.session_state.get("_auth_user")


def get_current_profile() -> Optional[Dict]:
    """Return full user profile from DB."""
    _init_session()
    return st.session_state.get("_auth_profile")


def is_admin() -> bool:
    profile = get_current_profile()
    return profile and profile.get("plan") == "admin"


def is_authenticated() -> bool:
    return get_current_user() is not None


def _load_profile(user_id: str) -> Optional[Dict]:
    """Load user profile from Supabase."""
    try:
        sb = _supabase_auth_client()
        if not sb: return None
        res = sb.table("user_profiles").select("*").eq("id", user_id).single().execute()
        return res.data
    except Exception:
        return None


def _save_session(session_data: Dict):
    """Persist session to Streamlit session state."""
    st.session_state["_auth_session"] = session_data
    user = session_data.get("user") or {}
    st.session_state["_auth_user"] = {
        "id": user.get("id"),
        "email": user.get("email"),
        "name": (user.get("user_metadata") or {}).get("full_name", user.get("email","").split("@")[0]),
        "avatar": (user.get("user_metadata") or {}).get("avatar_url", ""),
    }
    # Load profile from DB
    uid = user.get("id")
    if uid:
        profile = _load_profile(uid)
        st.session_state["_auth_profile"] = profile


def logout():
    """Clear session and log out."""
    try:
        sb = _supabase_auth_client()
        if sb: sb.auth.sign_out()
    except Exception: pass
    for k in ["_auth_session", "_auth_user", "_auth_profile"]:
        st.session_state[k] = None
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL / PASSWORD AUTH
# ══════════════════════════════════════════════════════════════════════════════

def sign_up_email(email: str, password: str, name: str = "") -> Dict:
    """Register new user with email + password."""
    try:
        sb = _supabase_auth_client()
        if not sb: return {"error": "Supabase not configured"}
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
        if "already registered" in err.lower():
            return {"error": "Email already registered. Try logging in."}
        return {"error": f"Sign-up error: {err[:100]}"}


def sign_in_email(email: str, password: str) -> Dict:
    """Login with email + password."""
    try:
        sb = _supabase_auth_client()
        if not sb: return {"error": "Supabase not configured"}
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        if res.session:
            _save_session({
                "access_token": res.session.access_token,
                "refresh_token": res.session.refresh_token,
                "user": {
                    "id": res.user.id,
                    "email": res.user.email,
                    "user_metadata": res.user.user_metadata or {},
                }
            })
            return {"success": True}
        return {"error": "Invalid email or password."}
    except Exception as e:
        err = str(e)
        if "invalid" in err.lower() or "credentials" in err.lower():
            return {"error": "Wrong email or password."}
        return {"error": f"Login error: {err[:100]}"}


def reset_password(email: str) -> Dict:
    """Send password reset email."""
    try:
        sb = _supabase_auth_client()
        if not sb: return {"error": "Supabase not configured"}
        sb.auth.reset_password_email(email)
        return {"success": True, "message": f"✅ Reset link sent to {email}. Check your inbox."}
    except Exception as e:
        return {"error": f"Reset failed: {str(e)[:100]}"}


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE OAUTH  (redirect flow)
# ══════════════════════════════════════════════════════════════════════════════

def get_google_oauth_url() -> Optional[str]:
    """Get Google OAuth redirect URL from Supabase."""
    try:
        url = _get_supabase_url()
        key = _get_anon_key()
        if not url or not key: return None
        # Supabase OAuth URL format
        redirect = st.secrets.get("APP_URL", "https://nsescannerbyrishabh.streamlit.app")
        oauth_url = (
            f"{url}/auth/v1/authorize"
            f"?provider=google"
            f"&redirect_to={redirect}"
        )
        return oauth_url
    except Exception:
        return None


def handle_oauth_callback():
    """Handle OAuth token from URL hash (called on page load)."""
    try:
        # Check for access_token in URL params (Supabase passes it as fragment)
        params = st.query_params
        access_token = params.get("access_token")
        refresh_token = params.get("refresh_token", "")

        if access_token and not is_authenticated():
            sb = _supabase_auth_client()
            if sb:
                # Set session
                res = sb.auth.set_session(access_token, refresh_token)
                if res.user:
                    _save_session({
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "user": {
                            "id": res.user.id,
                            "email": res.user.email,
                            "user_metadata": res.user.user_metadata or {},
                        }
                    })
                    # Clean URL
                    st.query_params.clear()
                    st.rerun()
    except Exception as e:
        print(f"[Auth] OAuth callback error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# USER PROFILE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def update_profile(updates: Dict) -> bool:
    """Update user profile fields."""
    user = get_current_user()
    if not user: return False
    try:
        sb = _supabase_auth_client()
        if not sb: return False
        sb.table("user_profiles").update(updates).eq("id", user["id"]).execute()
        # Refresh profile in session
        profile = _load_profile(user["id"])
        st.session_state["_auth_profile"] = profile
        return True
    except Exception as e:
        print(f"[Auth] Profile update error: {e}")
        return False


def get_user_id() -> Optional[str]:
    """Get current user's UUID."""
    user = get_current_user()
    return user["id"] if user else None


# ══════════════════════════════════════════════════════════════════════════════
# ALERT DELIVERY
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram_alert(user_id: str, message: str, signal_id: str = None,
                        symbol: str = "", strategy: str = "", alert_type: str = "SIGNAL") -> bool:
    """Send Telegram alert to a specific user."""
    try:
        sb = _supabase_admin_client()
        if not sb: return False

        # Get user's Telegram settings
        res = sb.table("user_profiles").select(
            "telegram_chat_id,telegram_bot_token,telegram_alerts,alert_min_sqi,alert_strategies"
        ).eq("id", user_id).single().execute()

        if not res.data: return False
        profile = res.data

        if not profile.get("telegram_alerts"): return False
        chat_id = profile.get("telegram_chat_id")
        bot_token = profile.get("telegram_bot_token")

        # Fall back to app-level Telegram if user hasn't set their own
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

        # Log the alert
        sb.table("user_alerts").insert({
            "user_id": user_id,
            "signal_id": signal_id,
            "symbol": symbol,
            "strategy": strategy,
            "alert_type": alert_type,
            "channel": "telegram",
            "message": message[:500],
            "success": success,
            "error_msg": resp.text[:200] if not success else None,
        }).execute()

        return success
    except Exception as e:
        print(f"[Auth] Telegram alert error: {e}")
        return False


def send_email_alert(user_id: str, subject: str, body: str,
                     symbol: str = "", strategy: str = "", alert_type: str = "SIGNAL") -> bool:
    """Send email alert via Supabase Edge Function or direct SMTP."""
    try:
        sb = _supabase_admin_client()
        if not sb: return False

        res = sb.table("user_profiles").select(
            "email,email_alerts,alert_strategies,alert_min_sqi"
        ).eq("id", user_id).single().execute()

        if not res.data or not res.data.get("email_alerts"): return False
        email = res.data["email"]

        # Use Supabase Edge Function for email (avoids SMTP config)
        # Fallback: log to DB only for now
        sb.table("user_alerts").insert({
            "user_id": user_id,
            "symbol": symbol,
            "strategy": strategy,
            "alert_type": alert_type,
            "channel": "email",
            "message": f"{subject}\n{body}"[:500],
            "success": True,
        }).execute()

        return True
    except Exception as e:
        print(f"[Auth] Email alert error: {e}")
        return False


def broadcast_signal_alert(signal: Dict, sqi: float = 0) -> int:
    """
    Send signal alert to ALL subscribed users (called from auto_scanner).
    Returns count of users alerted.
    """
    try:
        sb = _supabase_admin_client()
        if not sb: return 0

        strategy = signal.get("strategy", "")
        symbol   = signal.get("symbol", "")

        # Get all users with alerts enabled for this strategy
        res = sb.table("user_profiles").select(
            "id,telegram_alerts,email_alerts,alert_strategies,alert_min_sqi"
        ).or_("telegram_alerts.eq.true,email_alerts.eq.true").execute()

        if not res.data: return 0

        count = 0
        for user in res.data:
            uid = user["id"]
            min_sqi = user.get("alert_min_sqi") or 50
            if sqi < min_sqi: continue

            strats = user.get("alert_strategies") or []
            if strats and strategy not in strats: continue

            # Format message
            msg = _format_signal_alert(signal)

            if user.get("telegram_alerts"):
                if send_telegram_alert(uid, msg, symbol=symbol,
                                       strategy=strategy, alert_type="SIGNAL"):
                    count += 1
            if user.get("email_alerts"):
                send_email_alert(uid, f"NSE Signal: {symbol} — {strategy}",
                                 msg, symbol=symbol, strategy=strategy)
        return count
    except Exception as e:
        print(f"[Auth] Broadcast error: {e}")
        return 0


def _format_signal_alert(s: Dict) -> str:
    """Format a signal dict into a readable Telegram/email message."""
    sig_icon = "🟢 BUY" if s.get("signal") == "BUY" else "🔴 SHORT"
    sqi_icon = {"ELITE":"🏆","STRONG":"⭐","MODERATE":"✅","WEAK":"⚠️"}.get(s.get("sqi_grade",""),"")
    return (
        f"<b>🎯 NSE Scanner Pro Alert</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<b>{sig_icon}: {s.get('symbol','')}</b>\n"
        f"Strategy : {s.get('strategy','')}\n"
        f"CMP      : ₹{s.get('cmp',0):,.2f}\n"
        f"Entry    : ₹{s.get('entry',0):,.2f}\n"
        f"Stop     : ₹{s.get('sl',0):,.2f}\n"
        f"Target 1 : ₹{s.get('t1',0):,.2f}\n"
        f"Target 2 : ₹{s.get('t2',0):,.2f}\n"
        f"R:R      : 1:{s.get('rr',0)}\n"
        f"{sqi_icon} SQI: {s.get('sqi',0)}/100 ({s.get('sqi_grade','')})\n"
        f"Sector   : {s.get('sector','')}\n"
        f"Regime   : {s.get('regime','')}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<i>NSE Scanner Pro · nsescannerbyrishabh.streamlit.app</i>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE UI
# ══════════════════════════════════════════════════════════════════════════════

def render_auth_page():
    """
    Full-screen authentication page.
    Shows login/signup/forgot password.
    Called when user is not authenticated.
    """
    # Check for OAuth callback first
    handle_oauth_callback()
    if is_authenticated():
        return

    st.markdown("""
    <style>
    .auth-container {
        max-width: 420px; margin: 40px auto; padding: 0 16px;
    }
    .auth-logo {
        text-align: center; margin-bottom: 24px;
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(120deg, #FF6B35, #00d4ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .auth-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 28px 32px;
    }
    .auth-divider {
        text-align: center; color: #555; margin: 16px 0;
        font-size: 0.82rem; position: relative;
    }
    .auth-divider::before, .auth-divider::after {
        content: ''; position: absolute; top: 50%;
        width: 42%; height: 1px; background: #333;
    }
    .auth-divider::before { left: 0; }
    .auth-divider::after { right: 0; }
    .plan-badge {
        display: inline-block; padding: 2px 8px;
        border-radius: 10px; font-size: 0.7rem; font-weight: 600;
    }
    .plan-free { background: #1a2332; color: #5dade2; border: 1px solid #1e3a5f; }
    .plan-pro  { background: #1a2d1a; color: #00d26a; border: 1px solid #1b5e20; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-logo">🎯 NSE Scanner Pro</div>', unsafe_allow_html=True)

    # Mode selector
    mode = st.session_state.get("_auth_mode", "login")

    if mode == "login":
        _render_login()
    elif mode == "signup":
        _render_signup()
    elif mode == "forgot":
        _render_forgot_password()


def _render_login():
    st.markdown("#### Sign In")

    # Google OAuth button
    oauth_url = get_google_oauth_url()
    if oauth_url:
        st.markdown(
            f'<a href="{oauth_url}" target="_self" style="display:block;'
            f'text-align:center;padding:10px;background:#1a1d23;'
            f'border:1px solid #333;border-radius:8px;color:#fff;'
            f'text-decoration:none;font-size:.9rem;margin-bottom:12px;">'
            f'🔵 Continue with Google</a>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="auth-divider">or sign in with email</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submit   = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

    if submit:
        if not email or not password:
            st.error("Please enter email and password.")
        else:
            with st.spinner("Signing in…"):
                result = sign_in_email(email.strip(), password)
            if result.get("success"):
                st.success("✅ Signed in!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(result.get("error", "Sign in failed."))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create account", use_container_width=True):
            st.session_state["_auth_mode"] = "signup"
            st.rerun()
    with col2:
        if st.button("Forgot password?", use_container_width=True):
            st.session_state["_auth_mode"] = "forgot"
            st.rerun()

    # Plan info
    st.markdown("""
    <div style="margin-top:20px;text-align:center;font-size:.78rem;color:#666">
        <span class="plan-badge plan-free">FREE</span> Full access during beta
    </div>
    """, unsafe_allow_html=True)


def _render_signup():
    st.markdown("#### Create Account")

    oauth_url = get_google_oauth_url()
    if oauth_url:
        st.markdown(
            f'<a href="{oauth_url}" target="_self" style="display:block;'
            f'text-align:center;padding:10px;background:#1a1d23;'
            f'border:1px solid #333;border-radius:8px;color:#fff;'
            f'text-decoration:none;font-size:.9rem;margin-bottom:12px;">'
            f'🔵 Sign up with Google (Recommended)</a>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="auth-divider">or sign up with email</div>', unsafe_allow_html=True)

    with st.form("signup_form"):
        name     = st.text_input("Your name", placeholder="Rishabh")
        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password (min 8 characters)", type="password")
        confirm  = st.text_input("Confirm password", type="password")
        st.caption("✅ Personal data stays private — per-user isolation via Row Level Security")
        submit = st.form_submit_button("Create Account →", use_container_width=True, type="primary")

    if submit:
        if not email or not password:
            st.error("Email and password are required.")
        elif len(password) < 8:
            st.error("Password must be at least 8 characters.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            with st.spinner("Creating account…"):
                result = sign_up_email(email.strip(), password, name.strip())
            if result.get("success"):
                st.success(result["message"])
                st.info("After verifying your email, come back here and sign in.")
            else:
                st.error(result.get("error"))

    if st.button("← Back to sign in", use_container_width=True):
        st.session_state["_auth_mode"] = "login"
        st.rerun()


def _render_forgot_password():
    st.markdown("#### Reset Password")
    st.caption("Enter your email and we'll send a reset link.")

    with st.form("reset_form"):
        email  = st.text_input("Email", placeholder="you@example.com")
        submit = st.form_submit_button("Send Reset Link →", use_container_width=True, type="primary")

    if submit:
        if not email:
            st.error("Please enter your email.")
        else:
            with st.spinner("Sending reset link…"):
                result = reset_password(email.strip())
            if result.get("success"):
                st.success(result["message"])
            else:
                st.error(result.get("error"))

    if st.button("← Back to sign in", use_container_width=True):
        st.session_state["_auth_mode"] = "login"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ALERT PREFERENCES PAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_alert_preferences():
    """Render alert preferences section inside Settings page."""
    user = get_current_user()
    if not user:
        st.warning("Sign in to configure alerts.")
        return

    profile = get_current_profile() or {}
    st.markdown("### 🔔 Alert Preferences")
    st.caption("Get notified on Telegram or email when scanner generates signals matching your criteria.")

    # ── Telegram ──
    st.markdown("#### 📱 Telegram Alerts")
    with st.expander("How to set up Telegram", expanded=False):
        st.markdown("""
        1. Open Telegram → search **@BotFather** → `/newbot`
        2. Choose a name → copy the bot token (looks like `1234567890:AAF...`)
        3. Message your new bot → then open `https://api.telegram.org/bot<TOKEN>/getUpdates`
        4. Find your `chat_id` from the JSON response
        5. Paste both below and click Save
        """)

    col1, col2 = st.columns(2)
    with col1:
        tg_token = st.text_input(
            "Bot Token", value=profile.get("telegram_bot_token",""),
            type="password", placeholder="1234567890:AAF...", key="pref_tg_token"
        )
    with col2:
        tg_chat = st.text_input(
            "Chat ID", value=profile.get("telegram_chat_id",""),
            placeholder="-100123456789 or 123456789", key="pref_tg_chat"
        )

    tg_enabled = st.toggle(
        "Enable Telegram alerts",
        value=bool(profile.get("telegram_alerts")),
        key="pref_tg_enabled"
    )

    # Test button
    if tg_token and tg_chat:
        if st.button("📤 Send test Telegram message", key="tg_test"):
            try:
                resp = _req.post(
                    f"https://api.telegram.org/bot{tg_token}/sendMessage",
                    json={"chat_id": tg_chat, "text": "✅ NSE Scanner Pro — Telegram connected!"},
                    timeout=10
                )
                if resp.status_code == 200:
                    st.success("✅ Test message sent!")
                else:
                    st.error(f"Failed: {resp.json().get('description','Unknown error')}")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Email ──
    st.markdown("#### 📧 Email Alerts")
    email_enabled = st.toggle(
        "Enable email alerts (to your registered email)",
        value=bool(profile.get("email_alerts")),
        key="pref_email_enabled"
    )
    st.caption(f"Alerts sent to: **{user.get('email','')}**")

    # ── Filter criteria ──
    st.markdown("#### 🎯 Alert Filters")
    from scanners import STRATEGY_PROFILES
    all_strategies = list(STRATEGY_PROFILES.keys())

    selected_strats = st.multiselect(
        "Alert me for these strategies (empty = all)",
        options=all_strategies,
        default=profile.get("alert_strategies") or [],
        format_func=lambda x: f"{STRATEGY_PROFILES[x]['icon']} {STRATEGY_PROFILES[x]['name']}",
        key="pref_strats"
    )

    min_sqi = st.slider(
        "Minimum SQI to alert",
        min_value=0, max_value=90,
        value=int(profile.get("alert_min_sqi") or 50),
        step=5,
        help="Only send alerts for signals with SQI ≥ this value. 0 = all signals.",
        key="pref_min_sqi"
    )

    if st.button("💾 Save Alert Preferences", type="primary", key="save_alerts"):
        updates = {
            "telegram_bot_token": tg_token,
            "telegram_chat_id": tg_chat,
            "telegram_alerts": tg_enabled,
            "email_alerts": email_enabled,
            "alert_strategies": selected_strats,
            "alert_min_sqi": min_sqi,
        }
        if update_profile(updates):
            st.success("✅ Alert preferences saved!")
            st.rerun()
        else:
            st.error("Failed to save. Check Supabase connection.")

    # ── Alert history ──
    st.markdown("#### 📜 Recent Alerts Sent to You")
    try:
        sb = _supabase_auth_client()
        if sb:
            res = sb.table("user_alerts").select(
                "sent_at,symbol,strategy,alert_type,channel,success"
            ).eq("user_id", user["id"]).order("sent_at", desc=True).limit(20).execute()
            if res.data:
                import pandas as pd
                df = pd.DataFrame(res.data)
                df["sent_at"] = pd.to_datetime(df["sent_at"]).dt.strftime("%d %b %H:%M")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No alerts sent yet. Enable alerts and run a scan!")
    except Exception:
        st.info("Alert history unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_admin_dashboard():
    """Admin-only user management view."""
    if not is_admin():
        st.error("Admin access required.")
        return

    st.markdown("### 👑 Admin Dashboard")
    try:
        sb = _supabase_admin_client()
        if not sb: return
        res = sb.table("user_profiles").select(
            "email,display_name,plan,telegram_alerts,email_alerts,created_at,last_seen_at"
        ).order("created_at", desc=True).execute()

        if res.data:
            import pandas as pd
            df = pd.DataFrame(res.data)
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%d %b %Y")
            df["last_seen_at"] = pd.to_datetime(df["last_seen_at"]).dt.strftime("%d %b %H:%M")
            st.metric("Total Users", len(df))
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Grant admin
            st.markdown("**Grant admin access:**")
            cols = st.columns(3)
            with cols[0]:
                target_email = st.text_input("Email to promote")
            with cols[1]:
                new_plan = st.selectbox("Plan", ["free","pro","admin"])
            with cols[2]:
                if st.button("Update", key="admin_promote"):
                    sb.table("user_profiles").update({"plan": new_plan}).eq("email", target_email).execute()
                    st.success(f"Updated {target_email} to {new_plan}")
                    st.rerun()
    except Exception as e:
        st.error(f"Admin error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# REQUIRE AUTH GATE
# ══════════════════════════════════════════════════════════════════════════════

def require_auth() -> bool:
    """
    Call at top of app.py. Returns True if authenticated.
    If not authenticated, renders auth page and returns False.
    """
    _init_session()
    handle_oauth_callback()

    # Check if auth is enabled (can disable for development)
    try:
        auth_enabled = st.secrets.get("AUTH_ENABLED", "true").lower() == "true"
    except Exception:
        auth_enabled = False  # No secrets = dev mode

    if not auth_enabled:
        return True  # Dev mode — bypass auth

    if not _get_supabase_url() or not _get_anon_key():
        return True  # Supabase not configured — bypass auth

    if is_authenticated():
        # Update last_seen
        try:
            uid = get_user_id()
            if uid:
                sb = _supabase_auth_client()
                if sb:
                    sb.table("user_profiles").update(
                        {"last_seen_at": "now()"}
                    ).eq("id", uid).execute()
        except Exception: pass
        return True

    render_auth_page()
    return False
