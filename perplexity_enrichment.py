"""
perplexity_enrichment.py — News & Fundamental Context for NSE Scanner Pro
=========================================================================
Uses Google Gemini API for stock analysis. Two modes:
  - With SEARCH GROUNDING: Live news from Google Search (Gemini 2.0 Flash)
  - FALLBACK: Training-data based fundamental analysis (always works)

Get free key: https://aistudio.google.com/apikey
Add GEMINI_API_KEY to Streamlit Secrets + GitHub Secrets.
"""

import os, json, requests, streamlit as st
from datetime import date

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

CATALYST_TYPES = [
    "Q results / earnings surprise", "Management guidance",
    "FII/DII activity", "Block deal / bulk deal",
    "Promoter activity", "Order win / contract",
    "SEBI / regulatory", "Sectoral theme",
    "Global macro / crude / USD", "Analyst upgrade/downgrade",
    "M&A / demerger", "Technical breakout", "No specific catalyst",
]


def _call_gemini(prompt: str, use_search: bool = True) -> dict:
    """
    Call Gemini API. Tries search grounding first, falls back to base model.
    Returns raw JSON dict from API or raises exception.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")

    # Gemini 2.0 Flash with Google Search grounding
    if use_search:
        url     = f"{BASE_URL}/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools":    [{"google_search": {}}],
            "generationConfig": {"temperature": 0.05, "maxOutputTokens": 400},
        }
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 400:
            # Search grounding not available — fall back to base model
            return _call_gemini(prompt, use_search=False)
        r.raise_for_status()
        return r.json()

    # Fallback: Gemini 1.5 Flash, no search tool (uses training data)
    url     = f"{BASE_URL}/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 400},
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def get_signal_context(symbol: str, strategy: str, signal_direction: str) -> dict:
    """
    Get AI analysis for why a stock is in play.
    Returns {catalyst, catalyst_type, factors, sentiment, confidence, sources, is_live}
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set. Add it in Streamlit Secrets."}

    today = date.today().strftime("%d %B %Y")
    prompt = f"""Today is {today}. You are a precise NSE India market analyst.

The stock {symbol} (NSE) triggered a {signal_direction} signal using the {strategy} pattern.

Search for and report the most likely reason this stock is in focus. Look for:
1. Recent Q results / earnings (PAT, revenue vs estimates)
2. Management guidance or concall highlights
3. Large FII/DII buy or sell activity
4. Promoter activity (pledge/sale/buy)
5. Order wins or contract news
6. SEBI/regulatory news
7. Analyst rating changes with target price
8. Sectoral news affecting this company
9. Global events (US tariffs, crude, USD/INR)

RULES: Be specific with numbers. If nothing found, say so clearly. Do NOT hallucinate.

Respond ONLY with valid JSON (no markdown fences):
{{"catalyst": "specific one-sentence reason with data", "catalyst_type": "one of: {' / '.join(CATALYST_TYPES[:8])}", "factors": ["factor 1 with data", "factor 2 with data"], "sentiment": "BULLISH or BEARISH or NEUTRAL", "confidence": "HIGH or MEDIUM or LOW"}}"""

    try:
        data     = _call_gemini(prompt, use_search=True)
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        raw_text = raw_text.strip().replace("```json", "").replace("```", "").strip()

        # Extract grounding sources if search was used
        sources   = []
        grounding = data["candidates"][0].get("groundingMetadata", {})
        for chunk in grounding.get("groundingChunks", [])[:3]:
            uri = chunk.get("web", {}).get("uri", "")
            if uri:
                sources.append(uri)

        parsed              = json.loads(raw_text)
        parsed["sources"]   = sources
        parsed["is_live"]   = len(sources) > 0   # True = search grounded, False = training data
        return parsed

    except json.JSONDecodeError:
        return {
            "catalyst":      raw_text[:300] if "raw_text" in dir() else "Parse error",
            "catalyst_type": "other", "factors": [],
            "sentiment":     "NEUTRAL", "confidence": "LOW",
            "sources": [], "is_live": False,
        }
    except Exception as e:
        return {"error": str(e)[:300]}


def get_sector_context(sector: str) -> str:
    """One-line sector theme. Used in Dashboard sector cards."""
    if not GEMINI_API_KEY:
        return ""
    today  = date.today().strftime("%d %B %Y")
    prompt = (
        f"Today is {today}. In ONE sentence, what is the main specific news driving "
        f"the {sector} sector in India's NSE today? Name actual companies or events. "
        f"If nothing found, say 'No sector-specific news today'."
    )
    try:
        data = _call_gemini(prompt, use_search=True)
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


def enrich_telegram_alert(base_message: str, symbol: str, strategy: str, signal_dir: str) -> str:
    """Appends AI news context to an existing Telegram alert. Never breaks alerts on failure."""
    ctx = get_signal_context(symbol, strategy, signal_dir)
    if "error" in ctx or not ctx.get("catalyst"):
        return base_message
    if ctx.get("confidence") == "LOW" and not ctx.get("sources"):
        return base_message

    sent_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(ctx.get("sentiment",""), "⚪")
    live_tag  = "🔍 Live" if ctx.get("is_live") else "📚 Training"
    ctype     = ctx.get("catalyst_type", "")

    msg = f"\n\n📰 <b>[{ctype}]</b> {ctx.get('catalyst','—')}\n"
    msg += f"{sent_icon} {ctx.get('sentiment','—')} · {ctx.get('confidence','—')} · {live_tag}\n"
    for f in ctx.get("factors", []):
        msg += f"• {f}\n"
    sources = ctx.get("sources", [])
    if sources:
        msg += f'🔗 <a href="{sources[0]}">Source</a>'
    return base_message + msg


@st.cache_data(ttl=21600, show_spinner=False)   # Cache 6 hours — prevents 429 on re-renders
def _cached_gemini_context(symbol: str, strategy: str, signal_dir: str, _date_key: str) -> dict:
    """Cache wrapper — _date_key ensures cache resets each day automatically."""
    return get_signal_context(symbol, strategy, signal_dir)


def render_signal_context_card(symbol: str, strategy: str, signal_dir: str):
    """
    Streamlit card showing AI news analysis for a stock.
    Uses st.cache_data (6hr TTL) — safe against Streamlit re-renders causing repeated API calls.
    Drop into any page.
    """
    if not GEMINI_API_KEY:
        st.caption("💡 Add `GEMINI_API_KEY` in Streamlit Secrets to enable live news analysis.")
        st.code('GEMINI_API_KEY = "AIza..."', language="toml")
        return

    with st.spinner(f"Searching news for {symbol}..."):
        ctx = _cached_gemini_context(symbol, strategy, signal_dir, str(date.today()))

    if "error" in ctx:
        st.warning(f"⚠️ Analysis failed: {ctx['error']}")
        st.caption("This is cached — click Retry only once. Repeated calls waste free API quota.")
        if st.button("🔄 Retry once", key=f"retry_{symbol}_{date.today()}"):
            _cached_gemini_context.clear()
            st.rerun()
        return

    catalyst = ctx.get("catalyst", "")
    if "No specific catalyst" in catalyst and ctx.get("confidence") == "LOW":
        st.caption(f"📰 No specific news catalyst for {symbol} today — pure technical setup.")
        return

    conf      = ctx.get("confidence", "LOW")
    ctype     = ctx.get("catalyst_type", "")
    sent      = ctx.get("sentiment", "NEUTRAL")
    is_live   = ctx.get("is_live", False)
    colour    = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "orange"}.get(sent, "gray")
    live_badge = "🔍 Live news" if is_live else "📚 AI knowledge"

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{ctype}** &nbsp; :{colour}[**{sent}**] &nbsp; `{conf} confidence`")
    with col2:
        st.caption(live_badge)

    st.write(catalyst)
    for f in ctx.get("factors", []):
        st.write(f"• {f}")
    srcs = ctx.get("sources", [])
    if srcs:
        st.caption(f"[Source]({srcs[0]})")
