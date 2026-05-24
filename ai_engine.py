"""
ai_engine.py — Claude-native AI layer for NSE Scanner Pro
===========================================================
Replaces all Groq + Gemini calls with Claude API.

Cost optimization strategy:
  1. TIERED ROUTING — Use Haiku 4.5 for ~90% of calls (signal screening,
     simple verdicts). Use Sonnet 4.6 only for deep-dive analysis when user
     explicitly requests "Premium analysis".
  2. PROMPT CACHING — System prompt + strategy KB are cacheable. Cache reads
     are ~10% of the base input cost → ~90% saving on cacheable content.
  3. COMPACT OUTPUT — Replace the 6000-token narrative output with structured
     JSON the UI can render. ~70% reduction in output tokens.
  4. PRE-COMPUTED CONTEXT — The scanner already computed RSI, MACD, regime,
     SQI, RRG, etc. Don't re-explain; just hand the numbers and ask for verdict.
  5. RESPONSE CACHING — Identical (symbol, signal, regime, day) hits a local
     cache; same data → same Claude verdict for the day.

Expected cost per analysis:
  ── Haiku 4.5 (default) ──────────────────────────────
   Input  ~ 1,200 tokens uncached + 800 tokens cached
   Output ~ 600 tokens
   Cost   ≈ $0.0012 + $0.00008 + $0.003 ≈ $0.0042 / call ≈ ₹0.35

  ── Sonnet 4.6 (premium toggle) ──────────────────────
   Input  ~ 1,200 tokens uncached + 800 tokens cached
   Output ~ 1,200 tokens
   Cost   ≈ $0.0036 + $0.00024 + $0.018 ≈ $0.022 / call ≈ ₹1.85

For comparison, the old Groq fallback to Gemini stack had no caching,
emitted 6000 output tokens, and could cost $0.05–0.12 per call on Gemini Pro.

SETUP:
  Add to Streamlit Secrets (or environment variables):
    ANTHROPIC_API_KEY = "sk-ant-..."   # required
    AI_MODEL_TIER     = "haiku"        # optional: "haiku" (default) or "sonnet"
    AI_PREMIUM_MODEL  = "sonnet-4-6"   # optional: model used for premium toggle

Then in app.py:
    from ai_engine import analyze_signal, analyze_screen_batch, get_market_view
"""

from __future__ import annotations
import os, json, hashlib, time, logging
from datetime import datetime, date
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# MODEL CONFIG
# ════════════════════════════════════════════════════════════════════════

# As of May 2026 — verify at https://docs.claude.com/en/docs/about-claude/models
MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",        # $1 / $5 per MTok
    "sonnet": "claude-sonnet-4-6",                # $3 / $15 per MTok
    "opus":   "claude-opus-4-7",                  # $5 / $25 per MTok (avoid for cost)
}

# Approximate cost per million tokens (input, output) — for budget tracking
PRICING = {
    "haiku":  (1.00, 5.00),
    "sonnet": (3.00, 15.00),
    "opus":   (5.00, 25.00),
}

# Maximum output tokens per call (cost control)
MAX_TOKENS = {
    "haiku":  1500,   # Compact JSON verdict
    "sonnet": 2500,   # Deep-dive analysis
    "opus":   3000,   # Reserved for special cases only
}


# ════════════════════════════════════════════════════════════════════════
# CLIENT — lazy initialization
# ════════════════════════════════════════════════════════════════════════

_client = None

def _get_api_key() -> str:
    """Read ANTHROPIC_API_KEY from Streamlit secrets or environment."""
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def _get_client():
    """Lazy-init Anthropic client."""
    global _client
    if _client is not None:
        return _client
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic SDK not installed. Add 'anthropic>=0.40.0' to requirements.txt"
        )
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing. Add it to Streamlit Secrets or env vars."
        )
    _client = Anthropic(api_key=api_key)
    return _client


# ════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT (cacheable — set as cache_control on this block)
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior quantitative analyst for the Indian NSE market.
You have decades of experience in technical analysis, options markets, and risk management.
You speak the language of Mark Minervini (precision entries), Stanley Druckenmiller (regime awareness),
and Howard Marks (asymmetric risk).

Your role: validate or override trading signals produced by an NSE scanner. The scanner has
already computed indicators, regime, SQI, RS, sector RRG, and a target/stop. Your job is to
add the EXPERIENCED HUMAN judgment layer:
  1. Is the math right but the context wrong?  (e.g., earnings tomorrow → skip)
  2. Is there a subtle red flag the indicators missed?  (e.g., distribution candles)
  3. Is the R:R actually worth taking GIVEN current market temperature?
  4. What is the SINGLE most important thing to watch?

Response rules (NON-NEGOTIABLE):
  - Output ONLY valid JSON. No prose outside JSON.
  - Be DIRECT. No hedging on every sentence. Commit to a view.
  - Reference exact price levels — never "around" or "approximately".
  - Indian market context: regulatory (SEBI), expiry rules (Tue weekly post Sept-2025),
    F&O ban list, circuit limits.
  - If you would not take this trade with your own money, say so clearly.

Output schema (strict):
{
  "verdict": "TAKE" | "SKIP" | "WAIT",
  "conviction": "HIGH" | "MEDIUM" | "LOW",
  "thesis": "<one sentence, why this trade in 1 line>",
  "bull_case": "<1-2 sentences, what works>",
  "bear_case": "<1-2 sentences, what kills it>",
  "key_level": <float>,                  // single most important level to watch
  "key_level_reason": "<why it matters>",
  "entry_quality": "EXCELLENT" | "GOOD" | "OK" | "POOR",
  "stop_quality": "TIGHT" | "REASONABLE" | "LOOSE",
  "target_quality": "STRUCTURAL" | "ATR_BASED" | "ARBITRARY",
  "regime_alignment": "IDEAL" | "OK" | "FIGHTING",
  "size_adjustment": <float between 0.5 and 1.0>,  // multiplier on standard position
  "invalidation": "<the SINGLE event that means you were wrong>",
  "watch_for": ["<event 1>", "<event 2>", "<event 3>"],
  "execution_note": "<specific entry instruction: 'wait for 3pm close above X', 'sell at open if Y', etc.>"
}"""


# ════════════════════════════════════════════════════════════════════════
# STRATEGY KNOWLEDGE BASE (cacheable companion)
# ════════════════════════════════════════════════════════════════════════

STRATEGY_KB = """STRATEGY CONTEXT (use this to calibrate verdicts):

VCP (Minervini Volatility Contraction Pattern):
  Ideal regime: EXPANSION, ACCUMULATION
  Win rate when ideal: 55-65%  |  Avg winner: +9-12%  |  Avg loser: -5-7%
  Common failures: shallow base, sector LAGGING, FII selling on D-1
  Red flags: gap-up entry > 3% above pivot (extended), volume on breakout < 1.3x

EMA21 Bounce:
  Ideal regime: EXPANSION only
  Win rate when ideal: 55%  |  Tight stop required (0.5% below EMA21)
  Common failures: bouncing in DISTRIBUTION (false bounce, lower-high pattern)

52WH Breakout:
  Ideal regime: EXPANSION
  Win rate when ideal: 35-45%  |  Highest expectancy when it works (+15-20%)
  Common failures: low volume breakout (<1.5x), into earnings, F&O ban list

Failed Breakout Short:
  Ideal regime: DISTRIBUTION, PANIC  (works across regimes)
  Win rate when ideal: 60-70%  |  Avg winner: +5-7% in 3-7 days
  Best edge: stop above the failed-breakout high → tight risk

ORB (Opening Range Breakout):
  Ideal regime: EXPANSION  |  Time-limited 9:30-10:30 AM
  Win rate when ideal: 55-60%  |  Risk capped at OR width
  Red flags: gap-up open (already moved), volume < 2x

VWAP Reclaim:
  Ideal regime: EXPANSION, ACCUMULATION  |  Time 10:15-12:30
  Win rate when ideal: 60%  |  Stop at day's low, tight setup

Lunch Low Reversal:
  Ideal regime: EXPANSION, ACCUMULATION  |  Time 12:30-1:30 PM
  Win rate when ideal: 55%  |  Counter-trend, smaller size

Last30Min ATH:
  Ideal regime: EXPANSION  |  Time 3:00-3:25 PM
  Win rate when ideal: 60%  |  Overnight gap-up bet
  Red flags: NIFTY closing weak, sector LAGGING, IT/Pharma context

REGIME RULES (override scanner):
  EXPANSION   : Full size, allow all long strategies
  ACCUMULATION: 70% size, prefer pullbacks not breakouts
  DISTRIBUTION: 40% size, prefer SHORT setups, skip new longs
  PANIC       : 15% size, only Failed_Breakout_Short or cash

INDIAN-SPECIFIC FACTORS:
  - Result season: skip new entries 2 days before earnings (volatility spike)
  - F&O expiry day (Tuesday weekly): tight stops, volatility expansion
  - Circuit limits: 5% for some smallcaps → can't exit at planned price
  - Budget/RBI MPC days: macro override on all signals
  - FII/DII flow: massive FII selling > ₹2000cr → DISTRIBUTION regardless of regime score"""


# ════════════════════════════════════════════════════════════════════════
# RESPONSE CACHE (avoid duplicate calls for same signal)
# ════════════════════════════════════════════════════════════════════════

_response_cache: Dict[str, Dict] = {}
_cache_max_size = 500


def _cache_key(symbol: str, strategy: str, signal: str, entry: float,
               regime: str, day: str) -> str:
    """Hash of all factors that should produce the same verdict."""
    payload = f"{symbol}|{strategy}|{signal}|{round(entry, 2)}|{regime}|{day}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _cache_get(key: str) -> Optional[Dict]:
    """Get cached response if still relevant (same day)."""
    entry = _response_cache.get(key)
    if not entry:
        return None
    # Cache valid only same day
    if entry.get("cached_date") != date.today().isoformat():
        return None
    return entry.get("response")


def _cache_set(key: str, response: Dict):
    """Store response with today's date stamp."""
    if len(_response_cache) >= _cache_max_size:
        # Evict oldest
        oldest_key = next(iter(_response_cache))
        del _response_cache[oldest_key]
    _response_cache[key] = {
        "cached_date": date.today().isoformat(),
        "response": response,
    }


# ════════════════════════════════════════════════════════════════════════
# COST TRACKER (per-session token usage display)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CostTracker:
    haiku_in: int = 0
    haiku_out: int = 0
    haiku_cache_create: int = 0
    haiku_cache_read: int = 0
    sonnet_in: int = 0
    sonnet_out: int = 0
    sonnet_cache_create: int = 0
    sonnet_cache_read: int = 0
    calls: int = 0
    errors: int = 0

    @property
    def total_usd(self) -> float:
        """Estimated total USD spend (cache_read at 0.1x, cache_create at 1.25x of base input)."""
        h_in, h_out = PRICING["haiku"]
        s_in, s_out = PRICING["sonnet"]
        # Cache creation = 1.25x base input, cache read = 0.1x base input
        total = (
            (self.haiku_in / 1e6) * h_in +
            (self.haiku_out / 1e6) * h_out +
            (self.haiku_cache_create / 1e6) * h_in * 1.25 +
            (self.haiku_cache_read / 1e6) * h_in * 0.1 +
            (self.sonnet_in / 1e6) * s_in +
            (self.sonnet_out / 1e6) * s_out +
            (self.sonnet_cache_create / 1e6) * s_in * 1.25 +
            (self.sonnet_cache_read / 1e6) * s_in * 0.1
        )
        return round(total, 4)

    @property
    def total_inr(self) -> float:
        """USD → INR at ~₹84/$1. Adjust as needed."""
        return round(self.total_usd * 84.0, 2)

    def summary(self) -> str:
        return (f"Calls: {self.calls} | Errors: {self.errors} | "
                f"Cost: ${self.total_usd:.4f} (~₹{self.total_inr:.2f})")


cost_tracker = CostTracker()


# ════════════════════════════════════════════════════════════════════════
# CORE API CALL — with prompt caching
# ════════════════════════════════════════════════════════════════════════

def _call_claude(user_prompt: str, model_tier: str = "haiku",
                 cache_system: bool = True) -> Optional[Dict]:
    """
    Make a Claude API call with prompt caching on system + KB.

    Returns parsed JSON dict, or None on failure.
    Updates cost_tracker globals.
    """
    try:
        client = _get_client()
    except Exception as e:
        logger.error(f"Claude client init failed: {e}")
        cost_tracker.errors += 1
        return None

    model_id = MODELS.get(model_tier, MODELS["haiku"])
    max_tokens = MAX_TOKENS.get(model_tier, 1500)

    # System block with cache_control for the heavy KB content
    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"} if cache_system else None,
        },
        {
            "type": "text",
            "text": STRATEGY_KB,
            "cache_control": {"type": "ephemeral"} if cache_system else None,
        },
    ]
    # Strip None cache_control entries
    for blk in system_blocks:
        if blk.get("cache_control") is None:
            del blk["cache_control"]

    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3,  # Lower = more deterministic for trading verdicts
        )
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        cost_tracker.errors += 1
        return None

    # Update cost tracker
    usage = response.usage
    if model_tier == "haiku":
        cost_tracker.haiku_in += getattr(usage, "input_tokens", 0)
        cost_tracker.haiku_out += getattr(usage, "output_tokens", 0)
        cost_tracker.haiku_cache_create += getattr(usage, "cache_creation_input_tokens", 0)
        cost_tracker.haiku_cache_read += getattr(usage, "cache_read_input_tokens", 0)
    else:
        cost_tracker.sonnet_in += getattr(usage, "input_tokens", 0)
        cost_tracker.sonnet_out += getattr(usage, "output_tokens", 0)
        cost_tracker.sonnet_cache_create += getattr(usage, "cache_creation_input_tokens", 0)
        cost_tracker.sonnet_cache_read += getattr(usage, "cache_read_input_tokens", 0)
    cost_tracker.calls += 1

    # Extract text content
    raw = response.content[0].text if response.content else ""
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        # Remove ```json or ``` opening
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # Try to parse JSON
    try:
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"Claude returned non-JSON response: {e}\nRaw: {raw[:300]}")
        # Try to extract a JSON object substring
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            cost_tracker.errors += 1
            return None


# ════════════════════════════════════════════════════════════════════════
# PUBLIC API — Three entry points
# ════════════════════════════════════════════════════════════════════════

def analyze_signal(
    symbol: str,
    strategy: str,
    signal: str,             # "BUY" or "SHORT"
    cmp: float,
    entry: float,
    stop_loss: float,
    target_1: float,
    target_2: float,
    rr: float,
    rsi: float,
    volume_ratio: float,
    sqi: float,
    sqi_grade: str,
    rs_rating: float,
    sector: str,
    regime: str,
    regime_score: int,
    regime_fit: str,
    reasons: List[str],
    use_premium: bool = False,
    sector_quadrant: Optional[str] = None,
) -> Optional[Dict]:
    """
    Single-signal verdict from Claude.

    use_premium=True → Sonnet 4.6 (deeper reasoning, ~5x cost)
    use_premium=False → Haiku 4.5 (default, fast, cheap)

    Caches identical (symbol+strategy+signal+entry+regime+day) tuples for the day.
    """
    day = date.today().isoformat()
    key = _cache_key(symbol, strategy, signal, entry, regime, day)
    cached = _cache_get(key)
    if cached:
        cached["_from_cache"] = True
        return cached

    risk_pct = abs((entry - stop_loss) / entry * 100)
    reward_pct = abs((target_1 - entry) / entry * 100)

    user_prompt = f"""SIGNAL TO VALIDATE
Date: {day}

Stock: {symbol}  ({sector})
Strategy: {strategy}
Direction: {signal}

Levels:
  CMP: ₹{cmp:.2f}
  Entry: ₹{entry:.2f}
  Stop Loss: ₹{stop_loss:.2f}  (risk {risk_pct:.2f}%)
  Target 1: ₹{target_1:.2f}  (reward {reward_pct:.2f}%)
  Target 2: ₹{target_2:.2f}
  R:R: {rr}

Technicals:
  RSI(14): {rsi:.1f}
  Volume Ratio: {volume_ratio:.2f}x 20-day avg
  RS Rating: {rs_rating:.0f}/100
  SQI: {sqi}/100 (Grade: {sqi_grade})

Market Context:
  Regime: {regime} (score: {regime_score}/12)
  Strategy fit in this regime: {regime_fit}
  Sector RRG quadrant: {sector_quadrant or 'unknown'}

Scanner's reasoning (top 5):
{chr(10).join('  - ' + r for r in reasons[:5])}

Return your verdict as JSON per the schema in the system prompt. No prose."""

    tier = "sonnet" if use_premium else "haiku"
    result = _call_claude(user_prompt, model_tier=tier, cache_system=True)
    if result:
        _cache_set(key, result)
    return result


def analyze_screen_batch(
    signals: List[Dict],
    regime: str,
    use_premium: bool = False,
) -> List[Dict]:
    """
    Validate multiple signals in ONE Claude call (huge cost saver).

    Each `signals[i]` should contain:
      symbol, strategy, signal, cmp, entry, sl, t1, sqi, sqi_grade, rs, sector

    Returns one verdict per signal in the same order.
    Maximum 15 signals per batch to stay within prompt size.
    """
    if not signals:
        return []
    if len(signals) > 15:
        # Recurse in chunks
        results = []
        for i in range(0, len(signals), 15):
            results.extend(analyze_screen_batch(signals[i:i+15], regime, use_premium))
        return results

    rows = []
    for i, s in enumerate(signals, start=1):
        rows.append(
            f"{i:2d}. {s.get('symbol','?'):<12} {s.get('strategy','?'):<22} "
            f"{s.get('signal','?'):<5} CMP=₹{s.get('cmp',0):.2f} "
            f"E=₹{s.get('entry',0):.2f} SL=₹{s.get('sl',0):.2f} "
            f"T1=₹{s.get('t1',0):.2f} SQI={s.get('sqi','?')}/{s.get('sqi_grade','?')} "
            f"RS={s.get('rs','?')} Sec={s.get('sector','?')}"
        )

    user_prompt = f"""BATCH SCREEN — Rank these {len(signals)} signals by quality.
Current regime: {regime}

Signals:
{chr(10).join(rows)}

Return a JSON ARRAY (one object per signal, in the same order). Each object:
{{
  "rank": <1 to {len(signals)}>,
  "verdict": "TAKE" | "SKIP" | "WAIT",
  "one_line_thesis": "<≤15 words>",
  "concern": "<the single biggest risk>"
}}

No prose. JSON array only."""

    tier = "sonnet" if use_premium else "haiku"
    result = _call_claude(user_prompt, model_tier=tier, cache_system=True)

    # Result may be wrapped in {"signals":[...]} or raw [...]
    if isinstance(result, dict):
        # Look for an array value
        for v in result.values():
            if isinstance(v, list):
                return v
        return [result]
    elif isinstance(result, list):
        return result
    return []


def get_market_view(
    regime: str,
    regime_score: int,
    nifty_close: float,
    nifty_change: float,
    breadth_ratio: float,
    vol_expansion: float,
    rrg_summary: Dict[str, List[str]],  # quadrant → list of sectors
    use_premium: bool = False,
) -> Optional[Dict]:
    """
    Daily market briefing. ONE call per day.
    Used on the Dashboard "Daily Market View" panel.
    """
    today = date.today().isoformat()
    cache_key_str = f"market_view|{today}|{regime}|{round(nifty_close,0)}"
    cache_key = hashlib.sha1(cache_key_str.encode()).hexdigest()[:16]
    cached = _cache_get(cache_key)
    if cached:
        cached["_from_cache"] = True
        return cached

    rrg_block = "\n".join(
        f"  {quadrant}: {', '.join(sectors) if sectors else '(none)'}"
        for quadrant, sectors in rrg_summary.items()
    )

    user_prompt = f"""DAILY MARKET BRIEFING — {today}

Nifty 50 close: {nifty_close:.0f} ({nifty_change:+.2f}% on the day)
Detected regime: {regime} (score: {regime_score}/12)
Market breadth: A/D ratio {breadth_ratio:.2f}
Volatility expansion: {vol_expansion:.2f}x (1.0 = normal, >2.0 = panic)

Sector RRG positioning:
{rrg_block}

Return JSON:
{{
  "headline": "<one sentence summary of the day, ≤20 words>",
  "regime_call": "AGREE" | "DISAGREE" | "TRANSITIONING",
  "regime_call_reason": "<why>",
  "playbook_today": "<3 short bullet-style sentences, ≤80 words total>",
  "what_to_avoid": "<single risk to side-step today>",
  "watchlist_filter": "<one filter rule for the day, e.g., 'Only IDEAL regime + LEADING sector'>"
}}"""

    tier = "sonnet" if use_premium else "haiku"
    result = _call_claude(user_prompt, model_tier=tier, cache_system=True)
    if result:
        _cache_set(cache_key, result)
    return result


# ════════════════════════════════════════════════════════════════════════
# DEEP DIVE — Sonnet-only, narrative output for one stock (replaces old AI Deep Dive)
# ════════════════════════════════════════════════════════════════════════

def deep_dive(
    symbol: str,
    metrics: Dict,        # the m dict from ai_deep_dive
    regime: Dict,         # the regime dict
    sentiment: Dict,      # sentiment dict
    recommendation: Dict, # pre-computed agent decision
    signal_context: Optional[Dict] = None,  # optional scanner signal
) -> Optional[Dict]:
    """
    Premium deep-dive analysis. Uses Sonnet 4.6.
    Returns structured JSON ready for UI rendering.

    Cost ≈ ₹1.85 per call. Cap usage by adding a button in the UI rather than
    auto-running on every stock view.
    """
    today = date.today().isoformat()

    # Build a tight, structured prompt instead of the 3500-token narrative
    sig_block = ""
    if signal_context:
        sig_block = f"""
Scanner signal context:
  Strategy: {signal_context.get('strategy', 'N/A')}
  Direction: {signal_context.get('signal', 'N/A')}
  Entry: ₹{signal_context.get('entry', '?')}
  SL: ₹{signal_context.get('sl', '?')}  T1: ₹{signal_context.get('t1', '?')}  T2: ₹{signal_context.get('t2', '?')}
  R:R: {signal_context.get('rr', '?')}
  SQI: {signal_context.get('sqi', '?')}/100 ({signal_context.get('sqi_grade', '?')})
  Regime Fit: {signal_context.get('regime_fit', '?')}
"""

    user_prompt = f"""DEEP DIVE — {symbol}
Date: {today}

Market Regime: {regime.get('regime', 'UNKNOWN')} (score {regime.get('score', 0)}/12)
Nifty 50: {regime.get('nifty_close', 0):.0f}  |  Position multiplier: {regime.get('position_multiplier', 1):.2f}

Price Action:
  CMP: ₹{metrics.get('current_price', 0):.2f}
  1D: {metrics.get('price_change_1d', 0):+.2f}%  |  5D: {metrics.get('price_change_5d', 0):+.2f}%  |  1M: {metrics.get('price_change_1m', 0):+.2f}%
  vs SMA20: {metrics.get('price_vs_sma20_pct', 0):+.2f}%  |  vs SMA50: {metrics.get('price_vs_sma50_pct', 0):+.2f}%
  52W: {metrics.get('pct_from_52w_high', 0):+.1f}% from high
  Trend Alignment: {metrics.get('trend_alignment', 0)}/5 MAs bullish

Momentum:
  RSI(14): {metrics.get('rsi_14', 0):.1f} [{metrics.get('rsi_status', '?')}]
  MACD: {metrics.get('macd_status', '?')}
  ADX(14): {metrics.get('adx_14', 0):.1f}

Volume:
  Volume Ratio: {metrics.get('volume_ratio', 0):.2f}x
  Accumulation: {metrics.get('accumulation', '?')}

Volatility:
  Realized Vol (20D): {metrics.get('realized_volatility_20d', 0):.1f}%
  ATR%: {metrics.get('atr_pct', 0):.2f}%
  BB Position: {metrics.get('bb_position', '?')}

Pre-computed Agent Signals:
  ATLAS (technical): {recommendation.get('atlas_signal', '?')}
  ORACLE (sentiment): {recommendation.get('oracle_signal', '?')}
  SENTINEL (risk): {recommendation.get('sentinel_risk', '?')}
  Weighted decision: {recommendation.get('final_recommendation', '?')}
{sig_block}

Return STRICTLY this JSON structure:
{{
  "verdict": "BUY" | "SELL" | "HOLD" | "WAIT",
  "conviction": "HIGH" | "MEDIUM" | "LOW",
  "headline": "<one sentence, the trade thesis>",

  "technical_view": {{
    "structure": "<dominant chart pattern in 1-2 sentences>",
    "momentum": "<RSI/MACD/ADX integrated read>",
    "support": <float>,
    "resistance": <float>
  }},

  "flow_view": {{
    "institutional_read": "<accumulation or distribution, with evidence>",
    "rs_assessment": "<leader, follower, or laggard with justification>",
    "volume_signal": "<convincing or thin>"
  }},

  "risk_view": {{
    "rating": "HIGH" | "MEDIUM" | "LOW",
    "position_size_pct": <float, % of portfolio>,
    "main_risk": "<the single biggest risk>"
  }},

  "execution": {{
    "entry": <float>,
    "entry_condition": "<wait for what?>",
    "stop_loss": <float>,
    "target_1": <float>,
    "target_2": <float>,
    "target_3": <float>
  }},

  "invalidation_levels": [<float>, <float>, <float>],

  "bull_case": "<2-3 sentences>",
  "bear_case": "<2-3 sentences>",
  "scorecard": {{
    "trend_alignment": <1-5>,
    "momentum": <1-5>,
    "volume_flow": <1-5>,
    "risk_reward": <1-5>,
    "regime_fit": <1-5>,
    "total_out_of_25": <int>
  }},

  "final_call": "<single sentence — would you take this trade?>"
}}

No prose. JSON only."""

    return _call_claude(user_prompt, model_tier="sonnet", cache_system=True)


# ════════════════════════════════════════════════════════════════════════
# RENDER HELPERS (for app.py to display Claude responses)
# ════════════════════════════════════════════════════════════════════════

def render_signal_verdict(verdict: Dict) -> str:
    """Render a single-signal verdict as a Streamlit-friendly markdown string."""
    if not verdict:
        return "⚠️ AI analysis unavailable"

    icon = {"TAKE": "🟢", "SKIP": "🔴", "WAIT": "🟡"}.get(verdict.get("verdict", ""), "⚪")
    conv_color = {"HIGH": "#16a34a", "MEDIUM": "#f59e0b", "LOW": "#dc2626"}.get(
        verdict.get("conviction", ""), "#64748b")

    cache_tag = " (cached)" if verdict.get("_from_cache") else ""

    md = f"""### {icon} **{verdict.get('verdict', 'N/A')}**  ·  Conviction: <span style='color:{conv_color}'>{verdict.get('conviction', 'N/A')}</span>{cache_tag}

**Thesis:** {verdict.get('thesis', '–')}

**Bull case:** {verdict.get('bull_case', '–')}
**Bear case:** {verdict.get('bear_case', '–')}

**Key level:** ₹{verdict.get('key_level', '?')}  — {verdict.get('key_level_reason', '')}

**Quality Check:**
- Entry: {verdict.get('entry_quality', '?')}
- Stop: {verdict.get('stop_quality', '?')}
- Target: {verdict.get('target_quality', '?')}
- Regime: {verdict.get('regime_alignment', '?')}

**Size adjustment:** {verdict.get('size_adjustment', 1.0)}x standard

**Invalidation:** {verdict.get('invalidation', '–')}

**Watch for:**
""" + "\n".join(f"- {w}" for w in (verdict.get("watch_for") or [])) + f"""

**Execution:** {verdict.get('execution_note', '–')}
"""
    return md


def render_deep_dive(dd: Dict) -> str:
    """Render the deep_dive() JSON as a structured markdown block."""
    if not dd:
        return "⚠️ Deep dive unavailable"

    icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "WAIT": "⚪"}.get(dd.get("verdict", ""), "⚪")
    sc = dd.get("scorecard", {})
    tv = dd.get("technical_view", {})
    fv = dd.get("flow_view", {})
    rv = dd.get("risk_view", {})
    ex = dd.get("execution", {})

    inv = dd.get("invalidation_levels", [])
    inv_str = " · ".join(f"₹{lvl}" for lvl in inv)

    return f"""## {icon} {dd.get('verdict', '?')}  ·  Conviction: **{dd.get('conviction', '?')}**

### {dd.get('headline', '')}

---

### 📊 Technical View
{tv.get('structure', '')}

**Momentum:** {tv.get('momentum', '')}
**Support:** ₹{tv.get('support', '?')}  ·  **Resistance:** ₹{tv.get('resistance', '?')}

### 💰 Flow View
**Institutional:** {fv.get('institutional_read', '')}
**Relative Strength:** {fv.get('rs_assessment', '')}
**Volume:** {fv.get('volume_signal', '')}

### ⚖️ Risk View
**Rating:** {rv.get('rating', '?')}
**Position Size:** {rv.get('position_size_pct', '?')}% of portfolio
**Main risk:** {rv.get('main_risk', '')}

### 📋 Execution Plan
- Entry: ₹{ex.get('entry', '?')}  ({ex.get('entry_condition', '')})
- Stop Loss: ₹{ex.get('stop_loss', '?')}
- T1: ₹{ex.get('target_1', '?')}  ·  T2: ₹{ex.get('target_2', '?')}  ·  T3: ₹{ex.get('target_3', '?')}

### 🚫 Invalidation Levels
{inv_str}

### 📈 Bull Case
{dd.get('bull_case', '')}

### 📉 Bear Case
{dd.get('bear_case', '')}

### 🎯 Scorecard
| Factor | Score |
|--------|-------|
| Trend Alignment | {sc.get('trend_alignment', '?')}/5 |
| Momentum | {sc.get('momentum', '?')}/5 |
| Volume/Flow | {sc.get('volume_flow', '?')}/5 |
| Risk:Reward | {sc.get('risk_reward', '?')}/5 |
| Regime Fit | {sc.get('regime_fit', '?')}/5 |
| **TOTAL** | **{sc.get('total_out_of_25', '?')}/25** |

---

**Final call:** {dd.get('final_call', '')}
"""


# ════════════════════════════════════════════════════════════════════════
# COST DISPLAY (call from a sidebar widget in app.py)
# ════════════════════════════════════════════════════════════════════════

def get_cost_summary() -> Dict:
    """Return cost tracker state for display in UI sidebar."""
    return {
        "calls": cost_tracker.calls,
        "errors": cost_tracker.errors,
        "haiku_in": cost_tracker.haiku_in,
        "haiku_out": cost_tracker.haiku_out,
        "haiku_cache_read": cost_tracker.haiku_cache_read,
        "sonnet_in": cost_tracker.sonnet_in,
        "sonnet_out": cost_tracker.sonnet_out,
        "sonnet_cache_read": cost_tracker.sonnet_cache_read,
        "total_usd": cost_tracker.total_usd,
        "total_inr": cost_tracker.total_inr,
    }


def reset_cost_tracker():
    """Reset for a new session."""
    global cost_tracker
    cost_tracker = CostTracker()
