"""
perplexity_enrichment.py — News Context Layer for NSE Scanner Pro
==================================================================
Uses Google Gemini Flash (FREE) with Google Search grounding.
Prompt engineered for genuine NSE price catalysts, not generic summaries.

FREE TIER: gemini-2.0-flash — 15 RPM, 1,000 req/day, 500 search grounding/day
Get key at: https://aistudio.google.com/apikey
Add GEMINI_API_KEY to Streamlit Secrets + GitHub Secrets.
"""

import os, json, requests, streamlit as st
from datetime import date

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# What actually moves NSE stocks — in priority order
CATALYST_TYPES = [
    "Q results / earnings surprise",
    "Management guidance / concall",
    "FII/DII buying or selling",
    "Block deal / bulk deal",
    "Promoter buy / pledge / sale",
    "Order win / contract loss",
    "SEBI / regulatory action",
    "Sectoral tailwind / headwind",
    "Global macro / crude / USD",
    "Analyst upgrade / downgrade",
    "Merger / acquisition / demerger",
    "52-week high / technical breakout",
    "No specific news found",
]


def get_signal_context(symbol: str, strategy: str, signal_direction: str) -> dict:
    """
    Ask Gemini with live Google Search: what is ACTUALLY driving this NSE stock today?
    Returns {catalyst, catalyst_type, factors, sentiment, confidence, sources}
    Only reports what it finds — returns 'No specific news found' if nothing genuine exists.
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set. Add it in Streamlit Secrets."}

    today    = date.today().strftime("%d %B %Y")
    today_yr = date.today().strftime("%Y")
    prompt   = f"""You are a precise NSE India market analyst. Today is {today}.

The stock {symbol} (NSE India) has triggered a {signal_direction} signal via the {strategy} technical pattern.

TASK: Search Google right now for news about {symbol} NSE India {today_yr}. Find the REAL fundamental or news catalyst behind any recent price movement.

Look specifically for these in order of priority:
1. Quarterly results / earnings (PAT, revenue beat/miss vs estimates)
2. Management guidance changes or concall highlights
3. Large FII/DII buy or sell (from NSE bulk/block deal data)
4. Promoter activity (pledge increase/decrease, share sale)
5. Order wins, contract announcements, capacity expansions
6. SEBI order, regulatory approval/rejection
7. Analyst rating changes with target price
8. Sectoral news (RBI policy, government order, commodity prices) affecting this company
9. Global events (US tariffs, crude oil, USD/INR) affecting this sector

STRICT RULES:
- Only report what you ACTUALLY FIND in search results. Do NOT speculate or hallucinate.
- If no genuine news found, set catalyst to "No specific catalyst found — technical setup only" and confidence to "LOW"
- Numbers must be specific: "Q3 PAT up 23% YoY to ₹456Cr" not "strong results"
- Source must be a real URL you found

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "catalyst": "one specific sentence with actual numbers/facts",
  "catalyst_type": "one of: {' / '.join(CATALYST_TYPES[:8])} / other",
  "factors": [
    "specific factor 1 with data",
    "specific factor 2 with data"
  ],
  "sentiment": "BULLISH or BEARISH or NEUTRAL",
  "confidence": "HIGH if strong news found / MEDIUM if soft signal / LOW if no news"
}}"""

    payload = {
        "contents":        [{"parts": [{"text": prompt}]}],
        "tools":           [{"google_search": {}}],
        "generationConfig": {"temperature": 0.05, "maxOutputTokens": 400},
    }

    try:
        resp = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload, timeout=20)
        resp.raise_for_status()
        data     = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
        raw_text = raw_text.strip().replace("```json", "").replace("```", "").strip()

        grounding = data["candidates"][0].get("groundingMetadata", {})
        sources   = [
            chunk["web"]["uri"]
            for chunk in grounding.get("groundingChunks", [])[:3]
            if chunk.get("web", {}).get("uri")
        ]

        parsed           = json.loads(raw_text)
        parsed["sources"] = sources
        return parsed

    except json.JSONDecodeError:
        return {
            "catalyst":      raw_text[:250] if "raw_text" in dir() else "Parse error",
            "catalyst_type": "other",
            "factors":       [],
            "sentiment":     "NEUTRAL",
            "confidence":    "LOW",
            "sources":       [],
        }
    except requests.exceptions.Timeout:
        return {"error": "Gemini API timeout (>20s). NSE news fetch skipped."}
    except Exception as e:
        return {"error": str(e)[:100]}


def get_sector_context(sector: str) -> str:
    """One-line sector theme with actual news. Used in Dashboard sector cards."""
    if not GEMINI_API_KEY:
        return ""
    today = date.today().strftime("%d %B %Y")
    prompt = (
        f"Today is {today}. Search for actual news driving the {sector} sector "
        f"in India's NSE market today. In ONE sentence, state the specific catalyst "
        f"(e.g. a specific stock result, RBI action, crude price, FII data). "
        f"Use real data only. If nothing found, say 'No sector-specific news today'."
    )
    payload = {
        "contents":        [{"parts": [{"text": prompt}]}],
        "tools":           [{"google_search": {}}],
        "generationConfig": {"temperature": 0.05, "maxOutputTokens": 120},
    }
    try:
        resp = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload, timeout=12)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


def enrich_telegram_alert(base_message: str, symbol: str, strategy: str, signal_dir: str) -> str:
    """
    Appends Gemini news context to an existing Telegram alert message.
    Only called for ELITE/STRONG signals (controlled in alert_engine.py).
    Returns base_message unchanged if Gemini fails — never breaks alerts.
    """
    ctx = get_signal_context(symbol, strategy, signal_dir)
    if "error" in ctx or not ctx.get("catalyst"):
        return base_message

    # Skip adding context if confidence is LOW and no sources found
    if ctx.get("confidence") == "LOW" and not ctx.get("sources"):
        return base_message

    sent_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(ctx.get("sentiment", ""), "⚪")
    ctype     = ctx.get("catalyst_type", "")

    enriched  = f"\n\n📰 <b>[{ctype}]</b> {ctx.get('catalyst', '—')}\n"
    enriched += f"{sent_icon} {ctx.get('sentiment','—')} · {ctx.get('confidence','—')} confidence\n"

    for factor in ctx.get("factors", []):
        enriched += f"• {factor}\n"

    sources = ctx.get("sources", [])
    if sources:
        enriched += f'🔗 <a href="{sources[0]}">Source</a>'

    return base_message + enriched


def render_signal_context_card(symbol: str, strategy: str, signal_dir: str):
    """
    Drop into any Streamlit signal detail expander to show live news context.
    Caches per symbol per day — won't re-call Gemini on every render.
    """
    if not GEMINI_API_KEY:
        st.caption("💡 Add GEMINI_API_KEY in Streamlit Secrets to enable news context.")
        return

    cache_key = f"gemini_ctx_{symbol}_{date.today()}"
    if cache_key not in st.session_state:
        with st.spinner(f"🔍 Searching news for {symbol}..."):
            st.session_state[cache_key] = get_signal_context(symbol, strategy, signal_dir)

    ctx = st.session_state[cache_key]

    if "error" in ctx:
        st.caption(f"⚠️ News lookup failed: {ctx['error']}")
        return

    conf    = ctx.get("confidence", "LOW")
    ctype   = ctx.get("catalyst_type", "")
    sent    = ctx.get("sentiment", "NEUTRAL")
    colour  = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "orange"}.get(sent, "gray")

    # Don't render a card if genuinely nothing was found
    catalyst = ctx.get("catalyst", "")
    if "No specific catalyst" in catalyst and conf == "LOW":
        st.caption(f"📰 No news catalyst found for {symbol} today — pure technical setup.")
        return

    st.markdown(f"**📰 {ctype}** &nbsp; :{colour}[{sent}]  `{conf} confidence`")
    st.write(catalyst)
    for f in ctx.get("factors", []):
        st.write(f"• {f}")
    srcs = ctx.get("sources", [])
    if srcs:
        st.caption(f"[News source]({srcs[0]})")
