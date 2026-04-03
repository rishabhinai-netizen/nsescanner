"""
perplexity_enrichment.py — News Context Layer for NSE Scanner Pro
==================================================================
Uses Google Gemini Flash (FREE) with Google Search grounding to
answer "why is this stock moving?" and enrich Telegram alerts.

FREE TIER limits (Google AI Studio):
  - gemini-2.0-flash: 15 RPM, 1,000 RPD, 1M TPM/day
  - Google Search grounding: 500 free grounding calls/day
  - Cost: $0 forever at these volumes

Get your free API key at: https://aistudio.google.com/apikey
Then add GEMINI_API_KEY to Streamlit Secrets + GitHub Secrets.
"""

import os, json, requests, streamlit as st
from datetime import date


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"       # free tier
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def get_signal_context(symbol: str, strategy: str, signal_direction: str) -> dict:
    """
    Ask Gemini (with live Google Search) why this NSE stock is moving today.
    Returns dict: {catalyst, factors, sentiment, confidence, sources}
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}

    today = date.today().strftime("%d %B %Y")
    prompt = (
        f"Today is {today}. The NSE Indian stock {symbol} has triggered a "
        f"{signal_direction} trading signal ({strategy} pattern).\n\n"
        f"Search for today's news about {symbol} NSE India and tell me:\n"
        f"1. ONE sentence: main catalyst/reason this stock is in focus today\n"
        f"2. TWO brief bullet points: key factors affecting it\n"
        f"3. Sentiment: BULLISH, BEARISH, or NEUTRAL\n\n"
        f"Respond ONLY with valid JSON, no markdown:\n"
        f'{{ "catalyst": "...", "factors": ["...", "..."], "sentiment": "BULLISH", "confidence": "HIGH" }}'
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],       # enable live Google Search grounding
        "generationConfig": {
            "temperature":     0.1,
            "maxOutputTokens": 300,
        },
    }

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text content
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        raw = raw.strip().replace("```json", "").replace("```", "").strip()

        # Extract grounding sources (Google Search results used)
        sources = []
        grounding = data["candidates"][0].get("groundingMetadata", {})
        for chunk in grounding.get("groundingChunks", [])[:3]:
            web = chunk.get("web", {})
            if web.get("uri"):
                sources.append(web["uri"])

        parsed = json.loads(raw)
        parsed["sources"] = sources
        return parsed

    except json.JSONDecodeError:
        return {"catalyst": raw[:200] if "raw" in dir() else "—", "factors": [], "sentiment": "NEUTRAL", "sources": []}
    except Exception as e:
        return {"error": str(e)}


def get_sector_context(sector: str) -> str:
    """One-line sector theme for today."""
    if not GEMINI_API_KEY:
        return ""
    today = date.today().strftime("%d %B %Y")
    prompt = (
        f"Today is {today}. In ONE sentence, what is the main news or theme "
        f"driving the {sector} sector in India's NSE market today? Be specific."
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 100},
    }
    try:
        resp = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


def enrich_telegram_alert(base_message: str, symbol: str, strategy: str, signal_dir: str) -> str:
    """
    Appends Gemini news context to an existing Telegram alert.
    Safe — returns base_message unchanged if Gemini call fails.
    """
    ctx = get_signal_context(symbol, strategy, signal_dir)
    if "error" in ctx or not ctx.get("catalyst"):
        return base_message

    sentiment_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(ctx.get("sentiment", ""), "⚪")
    enriched = (
        f"\n\n📰 <b>Why in focus:</b> {ctx.get('catalyst', '—')}\n"
        f"{sentiment_icon} Sentiment: {ctx.get('sentiment', '—')} ({ctx.get('confidence', '—')})\n"
    )
    for f in ctx.get("factors", []):
        enriched += f"• {f}\n"

    sources = ctx.get("sources", [])
    if sources:
        enriched += f'🔗 <a href="{sources[0]}">Source</a>'

    return base_message + enriched


def render_signal_context_card(symbol: str, strategy: str, signal_dir: str):
    """
    Drop into any Streamlit signal detail view to show a news context card.
    Caches per symbol per day so repeated renders don't hit the API.
    """
    cache_key = f"gemini_ctx_{symbol}_{date.today()}"
    if cache_key not in st.session_state:
        with st.spinner(f"Searching news for {symbol}..."):
            st.session_state[cache_key] = get_signal_context(symbol, strategy, signal_dir)

    ctx = st.session_state[cache_key]
    if "error" in ctx:
        st.caption(f"News context unavailable: {ctx['error']}")
        return

    sentiment = ctx.get("sentiment", "NEUTRAL")
    colour = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "orange"}.get(sentiment, "gray")
    st.markdown(f"**📰 Why in focus** &nbsp; :{colour}[{sentiment} — {ctx.get('confidence','')}]")
    st.write(ctx.get("catalyst", "—"))
    for factor in ctx.get("factors", []):
        st.write(f"• {factor}")
    srcs = ctx.get("sources", [])
    if srcs:
        st.caption(f"[Source]({srcs[0]})")
