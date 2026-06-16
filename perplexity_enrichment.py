"""
perplexity_enrichment.py — News & Fundamental Context (v2: Claude)
===================================================================
v2 change: Migrated from Google Gemini to Anthropic Claude.

For live news grounding we use Claude's web_search tool. Falls back to
training-data answers if web_search unavailable on the API tier.

Setup: Add ANTHROPIC_API_KEY to Streamlit Secrets + GitHub Secrets.
Get a key: https://console.anthropic.com/settings/keys
"""

import os, json, streamlit as st
from datetime import date

# ── Detect API key availability ──────────────────────────────────────────
def _get_anthropic_key() -> str:
    try:
        v = st.secrets.get("ANTHROPIC_API_KEY", "")
        if v: return v
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")

ANTHROPIC_API_KEY = _get_anthropic_key()

CATALYST_TYPES = [
    "Q results / earnings surprise", "Management guidance",
    "FII/DII activity", "Block deal / bulk deal",
    "Promoter activity", "Order win / contract",
    "SEBI / regulatory", "Sectoral theme",
    "Global macro / crude / USD", "Analyst upgrade/downgrade",
    "M&A / demerger", "Technical breakout", "No specific catalyst",
]


def _call_claude(prompt: str, use_search: bool = True) -> dict:
    """
    Call Claude (Haiku 4.5 default). Optional web_search tool for live news.
    Returns a dict {'text': str, 'sources': list, 'is_live': bool}.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")

    from anthropic import Anthropic
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    tools = []
    if use_search:
        tools = [{"type": "web_search_20250305", "name": "web_search"}]

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            temperature=0.1,
            tools=tools,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        # If web_search tool not enabled on this API tier, retry without it
        if "tool" in str(e).lower() or "web_search" in str(e).lower():
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            raise

    # Extract text + any web sources
    text_parts = []
    sources = []
    for block in resp.content:
        btype = getattr(block, "type", "")
        if btype == "text":
            text_parts.append(getattr(block, "text", ""))
        elif btype == "web_search_tool_result":
            # Source URLs from web search
            content = getattr(block, "content", None) or []
            for item in (content if isinstance(content, list) else []):
                u = (getattr(item, "url", "") or
                     (item.get("url", "") if isinstance(item, dict) else ""))
                if u:
                    sources.append(u)

    return {
        "text": "\n".join(text_parts).strip(),
        "sources": sources[:3],
        "is_live": len(sources) > 0,
    }


def get_signal_context(symbol: str, strategy: str, signal_direction: str) -> dict:
    """
    Get AI analysis for why a stock is in play.
    Returns {catalyst, catalyst_type, factors, sentiment, confidence, sources, is_live}
    """
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not set. Add it in Streamlit Secrets."}

    today = date.today().strftime("%d %B %Y")
    cat_types_str = " / ".join(CATALYST_TYPES[:8])
    prompt = f"""Today is {today}. You are a precise NSE India market analyst.

OUTPUT: Respond ONLY with this exact JSON — no prose, no markdown fences, no explanation:
{{"catalyst": "one-sentence specific reason with numbers", "catalyst_type": "one of: {cat_types_str}", "factors": ["specific factor with data point", "specific factor with data point"], "sentiment": "BULLISH or BEARISH or NEUTRAL", "confidence": 65}}

TASK: {symbol} (NSE) has a {signal_direction} signal via {strategy} pattern.
Search and identify the most likely catalyst. Look for (in priority order):
1. Q results / earnings — PAT, revenue vs estimates, margins
2. Management guidance or concall highlights
3. Large FII/DII activity (amount in crores)
4. Promoter activity — pledge change, buy/sell
5. Order wins or contracts (amount if available)
6. Analyst upgrades/downgrades with target price
7. Sectoral / macro theme
8. If nothing found: say so — set catalyst_type to "No specific catalyst"

RULES: Numbers required. No hallucination. confidence = integer 0-100."""

    try:
        data = _call_claude(prompt, use_search=True)
        raw_text = data["text"].strip()

        # Strip markdown fences
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        # Find JSON object — safe version that doesn't raise ValueError
        parsed = None
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw_text[start:end])
            except json.JSONDecodeError:
                pass
        if parsed is None:
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                pass

        if not parsed:
            return {
                "catalyst": "Could not parse AI response. Claude may have returned prose instead of JSON.",
                "catalyst_type": "No specific catalyst",
                "factors": [], "sentiment": "NEUTRAL",
                "confidence": 50, "sources": [], "is_live": False,
            }

        # Normalise confidence: convert "HIGH"/"MEDIUM"/"LOW" string to int
        conf = parsed.get("confidence", 50)
        if isinstance(conf, str):
            conf = {"HIGH": 80, "MEDIUM": 55, "LOW": 30}.get(conf.upper(), 50)
        parsed["confidence"] = conf

        parsed["sources"] = data["sources"]
        parsed["is_live"] = data["is_live"]
        return parsed

    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def get_sector_context(sector: str) -> str:
    """One-line sector theme. Used in Dashboard sector cards."""
    if not ANTHROPIC_API_KEY:
        return ""
    today = date.today().strftime("%d %B %Y")
    prompt = (
        f"Today is {today}. In ONE sentence, what is the main specific news driving "
        f"the {sector} sector in India's NSE today? Name actual companies or events. "
        f"If nothing found, say 'No sector-specific news today'."
    )
    try:
        data = _call_claude(prompt, use_search=True)
        return data["text"]
    except Exception:
        return ""


def enrich_telegram_alert(base_message: str, symbol: str, strategy: str, signal_dir: str) -> str:
    """Appends AI news context to an existing Telegram alert. Never breaks alerts on failure."""
    try:
        ctx = get_signal_context(symbol, strategy, signal_dir)
    except Exception:
        return base_message
    if "error" in ctx or not ctx.get("catalyst"):
        return base_message
    if ctx.get("confidence") == "LOW" and not ctx.get("sources"):
        return base_message

    sent_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(ctx.get("sentiment", ""), "⚪")
    live_tag  = "🔍 Live" if ctx.get("is_live") else "📚 Training"
    ctype     = ctx.get("catalyst_type", "")

    msg = f"\n\n📰 <b>[{ctype}]</b> {ctx.get('catalyst', '—')}\n"
    msg += f"{sent_icon} {ctx.get('sentiment','—')} · {ctx.get('confidence','—')} · {live_tag}\n"
    for f in ctx.get("factors", []):
        msg += f"• {f}\n"
    sources = ctx.get("sources", [])
    if sources:
        msg += f'🔗 <a href="{sources[0]}">Source</a>'
    return base_message + msg


@st.cache_data(ttl=21600, show_spinner=False)
def _cached_claude_context(symbol: str, strategy: str, signal_dir: str, _date_key: str) -> dict:
    """Cache wrapper — _date_key ensures cache resets each day automatically."""
    return get_signal_context(symbol, strategy, signal_dir)


def render_signal_context_card(symbol: str, strategy: str, signal_dir: str):
    """Streamlit card showing AI news analysis for a stock."""
    if not ANTHROPIC_API_KEY:
        st.caption("💡 Add `ANTHROPIC_API_KEY` in Streamlit Secrets to enable live news analysis.")
        st.code('ANTHROPIC_API_KEY = "sk-ant-..."', language="toml")
        return

    with st.spinner(f"Searching news for {symbol}..."):
        ctx = _cached_claude_context(symbol, strategy, signal_dir, str(date.today()))

    if "error" in ctx:
        st.warning(f"⚠️ Analysis failed: {ctx['error']}")
        st.caption("This is cached — click Retry only once.")
        if st.button("🔄 Retry once", key=f"retry_{symbol}_{date.today()}"):
            _cached_claude_context.clear()
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
