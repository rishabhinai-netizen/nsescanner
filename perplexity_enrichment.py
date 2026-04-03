"""
perplexity_enrichment.py — News Context Layer for NSE Scanner Pro
==================================================================
Uses Perplexity Sonar API to answer "why is this stock moving?"
and enrich Telegram alerts + app display with cited news context.

Cost estimate: ~$0.001 per call (Sonar model, ~300 tokens output).
With Pro subscription you get $5 API credit/month = ~5,000 free calls.
Additional calls billed at $1/M input + $1/M output.
"""

import os
import json
import requests
import streamlit as st
from datetime import date


PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
SONAR_MODEL = "sonar"      # cheapest, fast: $1/M in + $1/M out
SONAR_PRO   = "sonar-pro"  # deeper: $3/M in + $15/M out (use for weekly reports)

HEADERS = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json",
}


def get_signal_context(symbol: str, strategy: str, signal_direction: str) -> dict:
    """
    Call Perplexity Sonar to explain why a stock is moving.
    Returns dict: {summary, reasons, sentiment, sources}
    
    Example call: get_signal_context("RELIANCE", "VCP", "BUY")
    """
    if not PERPLEXITY_API_KEY:
        return {"error": "PERPLEXITY_API_KEY not set in secrets"}

    today = date.today().strftime("%d %B %Y")
    prompt = f"""Today is {today}. The NSE stock {symbol} has triggered a {signal_direction} signal ({strategy} pattern) on the Indian stock market.

Give me:
1. ONE sentence: why this stock is in focus today (news catalyst, earnings, sector move, FII/DII activity, etc.)
2. TWO bullet points: key near-term factors (bullish or bearish)
3. Overall sentiment: BULLISH / BEARISH / NEUTRAL

Format your response as JSON only, no markdown:
{{
  "catalyst": "one sentence here",
  "factors": ["factor 1", "factor 2"],
  "sentiment": "BULLISH",
  "confidence": "HIGH/MEDIUM/LOW"
}}"""

    payload = {
        "model": SONAR_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise Indian stock market analyst. Always respond with valid JSON only. No markdown, no preamble. Base all answers on real current news."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 300,
        "search_recency_filter": "week",   # only news from last 7 days
        "return_citations": True,
    }

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"]
        sources = data.get("citations", [])

        # Strip any accidental markdown fences
        content = content.strip().replace("```json", "").replace("```", "").strip()
        parsed  = json.loads(content)
        parsed["sources"] = sources[:3]   # max 3 source URLs
        return parsed

    except json.JSONDecodeError:
        return {"catalyst": content[:200], "factors": [], "sentiment": "NEUTRAL", "sources": []}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_sector_context(sector: str) -> str:
    """
    Get a one-line sector-level context for dashboard display.
    Example: "IT sector is under pressure due to US tariff fears and weak Q4 guidance."
    """
    if not PERPLEXITY_API_KEY:
        return ""

    today = date.today().strftime("%d %B %Y")
    prompt = f"Today is {today}. In ONE sentence, what is the key theme or news driving the {sector} sector in India's NSE market today? Be specific, cite actual news if any."

    payload = {
        "model": SONAR_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "search_recency_filter": "day",
    }

    try:
        resp    = requests.post("https://api.perplexity.ai/chat/completions", headers=HEADERS, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def enrich_telegram_alert(base_message: str, symbol: str, strategy: str, signal_dir: str) -> str:
    """
    Append Perplexity news context to an existing Telegram alert message.
    Safe — returns base_message unchanged if Perplexity call fails.
    """
    ctx = get_signal_context(symbol, strategy, signal_dir)
    if "error" in ctx or not ctx.get("catalyst"):
        return base_message

    sentiment_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(ctx.get("sentiment", ""), "⚪")

    enrichment = (
        f"\n\n📰 <b>Why in focus:</b> {ctx.get('catalyst', '—')}\n"
        f"{sentiment_icon} Sentiment: {ctx.get('sentiment', '—')} ({ctx.get('confidence', '—')})\n"
    )
    factors = ctx.get("factors", [])
    if factors:
        enrichment += "• " + "\n• ".join(factors) + "\n"

    sources = ctx.get("sources", [])
    if sources:
        enrichment += f"🔗 <a href='{sources[0]}'>Source</a>"

    return base_message + enrichment


# ── Streamlit display helper ───────────────────────────────────────────────────

def render_signal_context_card(symbol: str, strategy: str, signal_dir: str):
    """
    Drop this into any Streamlit signal detail view.
    Shows a contextual news card. Caches for 30 minutes.
    """
    cache_key = f"pplx_{symbol}_{date.today()}"
    if cache_key not in st.session_state:
        with st.spinner(f"Fetching news context for {symbol}..."):
            st.session_state[cache_key] = get_signal_context(symbol, strategy, signal_dir)

    ctx = st.session_state[cache_key]
    if "error" in ctx:
        st.caption(f"News context unavailable: {ctx['error']}")
        return

    sentiment = ctx.get("sentiment", "NEUTRAL")
    sentiment_color = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "orange"}.get(sentiment, "gray")

    st.markdown(f"**📰 Why in focus** &nbsp; :{sentiment_color}[{sentiment}]")
    st.write(ctx.get("catalyst", "—"))
    factors = ctx.get("factors", [])
    for f in factors:
        st.write(f"• {f}")
    sources = ctx.get("sources", [])
    if sources:
        st.caption(f"[Source]({sources[0]})")


# ── Usage in alert_engine.py ───────────────────────────────────────────────────
# 
# from perplexity_enrichment import enrich_telegram_alert
#
# # In your existing send_telegram_alert() function, before sending:
# message = build_base_alert(signal)
# message = enrich_telegram_alert(message, signal["symbol"], signal["strategy"], signal["signal"])
# send_to_telegram(message)
#
# Add PERPLEXITY_API_KEY to GitHub Secrets and Streamlit Secrets.
# Get your key at: https://www.perplexity.ai/api-platform
