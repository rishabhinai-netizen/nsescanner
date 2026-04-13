"""
AI DEEP DIVE — Multi-Agent Trading System v7.0
NSE Scanner Pro Integration Module
Zero hard external deps — all AI imports are lazy (inside functions).
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── These are already in requirements.txt and installed by the main app ──────
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0).rolling(n).mean()
    l = (-d.where(d < 0, 0)).rolling(n).mean()
    rs = g / l.replace(0, 1e-10)
    return (100 - 100 / (1 + rs)).fillna(50).clip(0, 100)


def _macd(s: pd.Series):
    e1 = s.ewm(span=12, adjust=False).mean()
    e2 = s.ewm(span=26, adjust=False).mean()
    m = e1 - e2
    sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig


def _adx(hi, lo, cl, n=14) -> float:
    try:
        pdm = hi.diff().clip(lower=0)
        ndm = (-lo.diff()).clip(lower=0)
        tr  = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(1)
        atr = tr.rolling(n).mean().replace(0, 1e-10)
        pdi = 100 * pdm.rolling(n).mean() / atr
        ndi = 100 * ndm.rolling(n).mean() / atr
        dx  = (pdi - ndi).abs() / (pdi + ndi).replace(0, 1e-10) * 100
        return float(dx.rolling(n).mean().iloc[-1])
    except Exception:
        return 20.0


def _compute_metrics(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or len(df) < 30:
        return None
    try:
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        m: Dict[str, Any] = {}
        m["current_price"]       = float(c.iloc[-1])
        m["price_change_1d"]     = (c.iloc[-1] / c.iloc[-2] - 1) * 100 if len(c) >= 2 else 0
        m["price_change_5d"]     = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) >= 6 else 0
        m["high_52w"]            = float(h.max())
        m["low_52w"]             = float(l.min())
        m["pct_from_52w_high"]   = (m["current_price"] / m["high_52w"] - 1) * 100
        m["pct_from_52w_low"]    = (m["current_price"] / m["low_52w"]  - 1) * 100
        avg_vol = v.iloc[-20:].mean()
        m["volume_ratio"]        = float(v.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
        rsi_s = _rsi(c)
        m["rsi_14"]              = float(rsi_s.iloc[-1])
        m["rsi_status"]          = ("OVERSOLD" if m["rsi_14"] < 30 else
                                    "OVERBOUGHT" if m["rsi_14"] > 70 else "NEUTRAL")
        ml, ms, mh = _macd(c)
        if len(mh) >= 2:
            if   mh.iloc[-1] > 0 and mh.iloc[-2] <= 0: m["macd_status"] = "BULLISH_CROSSOVER"
            elif mh.iloc[-1] < 0 and mh.iloc[-2] >= 0: m["macd_status"] = "BEARISH_CROSSOVER"
            else: m["macd_status"] = "BULLISH" if mh.iloc[-1] > 0 else "BEARISH"
        else:
            m["macd_status"] = "NEUTRAL"
        sma20 = c.rolling(20).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        m["price_vs_sma20_pct"]  = (m["current_price"] / sma20.iloc[-1] - 1) * 100
        if   c.iloc[-1] > sma20.iloc[-1] > ema50.iloc[-1]: m["trend_short_term"] = "BULLISH"
        elif c.iloc[-1] < sma20.iloc[-1] < ema50.iloc[-1]: m["trend_short_term"] = "BEARISH"
        else:                                                m["trend_short_term"] = "NEUTRAL"
        m["trend_medium_term"]   = "BULLISH" if ema50.iloc[-1] > ema50.iloc[-10] else "BEARISH"
        adx_val                  = _adx(h, l, c)
        m["adx_14"]              = adx_val
        m["trend_strength"]      = "STRONG" if adx_val > 25 else ("MODERATE" if adx_val > 20 else "WEAK")
        tr2 = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(1)
        atr14 = tr2.rolling(14).mean()
        m["atr_pct"]             = float(atr14.iloc[-1]) / m["current_price"] * 100
        m["realized_volatility_20d"] = float(c.pct_change().iloc[-20:].std() * (252 ** 0.5) * 100)
        # store for chart
        df["_rsi"]   = rsi_s
        df["_sma20"] = sma20
        df["_ema50"] = ema50
        return m
    except Exception as e:
        print(f"[AI Deep Dive] _compute_metrics error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# REGIME
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_regime() -> Dict:
    try:
        vix_h  = yf.Ticker("^INDIAVIX").history(period="5d")
        nft_h  = yf.Ticker("^NSEI").history(period="5d")
        vix    = float(vix_h["Close"].iloc[-1]) if not vix_h.empty else None
        nifty  = float(nft_h["Close"].iloc[-1]) if not nft_h.empty else None
        nchg   = (nft_h["Close"].iloc[-1] / nft_h["Close"].iloc[-2] - 1) * 100 if len(nft_h) >= 2 else 0
        adv = dec = 0
        for t in ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
                  "SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS","LT.NS"]:
            try:
                h2 = yf.Ticker(t).history(period="2d")
                if len(h2) >= 2:
                    if h2["Close"].iloc[-1] > h2["Close"].iloc[-2]: adv += 1
                    else: dec += 1
            except Exception: pass
        breadth = adv / (adv + dec) if (adv + dec) > 0 else 0.5
    except Exception:
        return dict(regime="UNKNOWN", code="UNKNOWN", confidence=0,
                    description="Data unavailable", strategy="Wait",
                    atlas_weight=33, oracle_weight=33, sentinel_weight=34,
                    vix="N/A", breadth=0.5, nifty_level="N/A", nifty_change=0)

    if vix is None:
        r = dict(regime="UNKNOWN", code="UNKNOWN", confidence=0, description="VIX unavailable",
                 strategy="Wait", atlas_weight=33, oracle_weight=33, sentinel_weight=34)
    elif vix > 22:
        r = dict(regime="LIQUIDITY VACUUM", code="STRESS",
                 confidence=min(95, 70 + (vix - 22) * 2),
                 description="High fear — Risk-off mode",
                 strategy="Reduce exposure, cash 60-70%",
                 atlas_weight=10, oracle_weight=20, sentinel_weight=70)
    elif vix < 13:
        r = dict(regime="COMPRESSION", code="ACCUMULATE",
                 confidence=min(90, 65 + (13 - vix) * 3),
                 description="Low volatility — Market coiling",
                 strategy="Prepare for breakout, mean-reversion",
                 atlas_weight=25, oracle_weight=35, sentinel_weight=40)
    elif 14 <= vix <= 19 and breadth > 0.6:
        r = dict(regime="MOMENTUM CASCADE", code="TREND", confidence=75,
                 description="Strong directional trend",
                 strategy="Follow momentum, trail stops",
                 atlas_weight=50, oracle_weight=20, sentinel_weight=30)
    elif 13 <= vix <= 17 and 0.4 <= breadth <= 0.6:
        r = dict(regime="LIQUIDITY DRIFT", code="GOLDILOCKS", confidence=70,
                 description="Optimal alpha-generation environment",
                 strategy="Active trading, balanced exposure",
                 atlas_weight=35, oracle_weight=40, sentinel_weight=25)
    else:
        r = dict(regime="REGIME TRANSITION", code="UNCERTAINTY", confidence=55,
                 description="Mixed signals — transition period",
                 strategy="Reduce sizes, wait for clarity",
                 atlas_weight=20, oracle_weight=30, sentinel_weight=50)

    r.update(vix=vix, breadth=breadth, nifty_level=nifty, nifty_change=nchg)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT + DECISION
# ══════════════════════════════════════════════════════════════════════════════

def _sentiment(m: Dict) -> Dict:
    score, factors = 0, []
    vr, chg1 = m.get("volume_ratio", 1.0), m.get("price_change_1d", 0)
    if vr > 2.0:
        if chg1 > 0: score += 3; factors.append(f"🔥 INSTITUTIONAL BUY: {vr:.2f}x volume on up day (+3)")
        else:         score -= 2; factors.append(f"⚠️ {vr:.2f}x volume on DOWN day — distribution? (-2)")
    elif vr > 1.5:
        if chg1 > 0: score += 2; factors.append(f"✅ Above-avg vol {vr:.2f}x + positive price (+2)")
        else:         score -= 1; factors.append(f"Vol {vr:.2f}x on down day (-1)")
    elif vr < 0.7:   factors.append(f"📉 Below-avg vol {vr:.2f}x — weak conviction (0)")
    ph = m.get("pct_from_52w_high", -100)
    if ph > -5:    score += 2; factors.append(f"💪 Near 52W high ({ph:+.1f}%) — strong RS (+2)")
    elif ph > -10: score += 1; factors.append(f"📈 Close to 52W high ({ph:+.1f}%) (+1)")
    pl = m.get("pct_from_52w_low", 0)
    if pl < 10:    score -= 2; factors.append(f"⚠️ Near 52W low (+{pl:.1f}%) — weak RS (-2)")
    c5 = m.get("price_change_5d", 0)
    if c5 > 5 and chg1 > 0:   score += 1; factors.append(f"🚀 Sustained uptrend: 5D +{c5:.1f}%, 1D +{chg1:.1f}% (+1)")
    elif c5 < -5 and chg1 < 0: score -= 1; factors.append(f"📉 Sustained downtrend: 5D {c5:.1f}%, 1D {chg1:.1f}% (-1)")
    vol = m.get("realized_volatility_20d", 0)
    if vol > 40: score -= 1; factors.append(f"⚡ High volatility {vol:.1f}% (-1)")
    overall = ("STRONG BULLISH" if score >= 3 else "BULLISH" if score >= 1 else
               "STRONG BEARISH" if score <= -3 else "BEARISH" if score <= -1 else "NEUTRAL")
    return dict(overall=overall, score=score, factors=factors)


def _decision(m: Dict, regime: Dict, sent: Dict) -> Dict:
    a = 0
    if m.get("trend_short_term") == "BULLISH": a += 2
    elif m.get("trend_short_term") == "BEARISH": a -= 2
    if m.get("macd_status") == "BULLISH_CROSSOVER": a += 1
    elif m.get("macd_status") == "BEARISH_CROSSOVER": a -= 1
    rsi = m.get("rsi_14", 50)
    if rsi < 30: a += 1
    elif rsi > 70: a -= 1
    atlas = "BUY" if a >= 2 else ("SELL" if a <= -2 else "HOLD")

    os = sent.get("score", 0)
    oracle = "BUY" if os >= 2 else ("SELL" if os <= -2 else "HOLD")
    oracle_conf = min(85, 50 + abs(os) * 10)

    rs = 0
    if m.get("realized_volatility_20d", 0) > 35: rs += 2
    elif m.get("realized_volatility_20d", 0) < 20: rs -= 1
    if m.get("atr_pct", 0) > 3: rs += 1
    ph = m.get("pct_from_52w_high", -50)
    if ph > -10: rs += 1
    elif ph < -30: rs -= 1
    sentinel = "HIGH" if rs >= 2 else ("LOW" if rs <= 0 else "MEDIUM")

    aw = regime.get("atlas_weight", 33) / 100
    ow = regime.get("oracle_weight", 33) / 100
    sw = regime.get("sentinel_weight", 34) / 100
    sm = {"BUY": 1, "HOLD": 0, "SELL": -1}
    rp = {"LOW": 0, "MEDIUM": 0.2, "HIGH": 0.5}
    ws = (sm[atlas] * aw + sm[oracle] * ow) * (1 - rp[sentinel] * sw)

    if ws >= 0.4:    final, trend = "BUY", "BULLISH"
    elif ws <= -0.4: final, trend = "SELL", "BEARISH"
    else:            final, trend = "HOLD", "NEUTRAL"

    return dict(atlas_signal=atlas, atlas_score=a,
                oracle_signal=oracle, oracle_confidence=oracle_conf, oracle_score=os,
                sentinel_risk=sentinel, sentinel_score=rs,
                weighted_score=round(ws, 3), final_recommendation=final, trend=trend)


# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════

def _chart(df: pd.DataFrame, sym: str, sig: Optional[Dict] = None) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{sym} — Price Action", "RSI (14)", "Volume"))
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)
    if "_sma20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["_sma20"], name="SMA 20",
                                 line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    if "_ema50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["_ema50"], name="EMA 50",
                                 line=dict(color="#42a5f5", width=1.5)), row=1, col=1)
    if sig:
        for price, label, color in [(sig.get("entry"), "Entry", "#42a5f5"),
                                    (sig.get("sl"),    "SL",    "#ef5350"),
                                    (sig.get("t1"),    "T1",    "#26a69a"),
                                    (sig.get("t2"),    "T2",    "#66bb6a")]:
            if price:
                try:
                    fig.add_hline(y=float(price), line_dash="dash", line_color=color,
                                  annotation_text=f"{label} {_fp(price)}", row=1, col=1)
                except Exception: pass
    if "_rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["_rsi"], name="RSI",
                                 line=dict(color="#ab47bc", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=2, col=1)
    colors = ["#ef5350" if r["Close"] < r["Open"] else "#26a69a" for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors), row=3, col=1)
    fig.update_layout(height=740, xaxis_rangeslider_visible=False, template="plotly_dark",
                      showlegend=True, hovermode="x unified",
                      margin=dict(l=50, r=60, t=55, b=30),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="monospace", size=11))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# AI CALLS — fully lazy imports
# ══════════════════════════════════════════════════════════════════════════════

_SYS = """You are the central intelligence of a Multi-Agent Swarm Trading System for NSE India.
Analyse through THREE agents: ATLAS (technical), ORACLE (price-action/sentiment), SENTINEL (risk).
Be precise, data-driven, concise. Always complete all 5 sections."""


def _build_prompt(ticker, m, regime, sent, rec, sig):
    sc = ""
    if sig:
        sc = f"""
╔══ NSE SCANNER PRO SIGNAL ══╗
Strategy : {sig.get("strategy","N/A")}   Signal: {sig.get("signal","N/A")}
Entry ₹{sig.get("entry","?")}  SL ₹{sig.get("sl","?")}  T1 ₹{sig.get("t1","?")}  T2 ₹{sig.get("t2","?")}
R:R {sig.get("rr","?")}   SQI {sig.get("sqi","?")} ({sig.get("sqi_grade","?")})   Sector: {sig.get("sector","?")}
╚════════════════════════════╝"""

    return f"""Analyse {ticker}. {sc}

REGIME: {regime.get("regime")} [{regime.get("code")}] {regime.get("confidence")}% conf
VIX {regime.get("vix","?")} | Nifty {regime.get("nifty_level","?")} ({regime.get("nifty_change",0):+.2f}%) | Breadth {regime.get("breadth","?")}
Agent weights → ATLAS {regime.get("atlas_weight")}% | ORACLE {regime.get("oracle_weight")}% | SENTINEL {regime.get("sentinel_weight")}%

TECHNICALS:
Price ₹{m.get("current_price",0):,.2f}  1D {m.get("price_change_1d",0):+.2f}%  5D {m.get("price_change_5d",0):+.2f}%
RSI {m.get("rsi_14",0):.1f} [{m.get("rsi_status")}]  ADX {m.get("adx_14",0):.1f} [{m.get("trend_strength")}]  MACD {m.get("macd_status")}
Trend S/M: {m.get("trend_short_term")} / {m.get("trend_medium_term")}
vs SMA20 {m.get("price_vs_sma20_pct",0):+.2f}%  VolRatio {m.get("volume_ratio",0):.2f}x
52W High ₹{m.get("high_52w",0):,.2f} ({m.get("pct_from_52w_high",0):+.1f}%)  52W Low ₹{m.get("low_52w",0):,.2f} (+{m.get("pct_from_52w_low",0):.1f}%)
Volatility {m.get("realized_volatility_20d",0):.1f}%  ATR% {m.get("atr_pct",0):.2f}%

SENTIMENT: {sent.get("overall")} (score {sent.get("score")})
{chr(10).join("• " + f for f in sent.get("factors",[]))}

PRE-COMPUTED: ATLAS={rec.get("atlas_signal")} | ORACLE={rec.get("oracle_signal")} ({rec.get("oracle_confidence")}%) | SENTINEL={rec.get("sentinel_risk")}
Weighted score {rec.get("weighted_score")} → FINAL: {rec.get("final_recommendation")}

Provide analysis in EXACTLY this format:

## 1. MARKET REGIME
[2-3 sentences on regime implication for this trade]

## 2. AGENT ANALYSIS

### AGENT A — ATLAS | {rec.get("atlas_signal")} | {regime.get("atlas_weight")}% weight
**Reasoning:** [trend structure, MACD, RSI, key levels]
**Risk flags:** [technical concerns]

### AGENT B — ORACLE | {rec.get("oracle_signal")} | {regime.get("oracle_weight")}% weight
**Reasoning:** [institutional volume, relative strength vs 52W]
**Risk flags:** [sentiment concerns]

### AGENT C — SENTINEL | {rec.get("sentinel_risk")} risk | {regime.get("sentinel_weight")}% weight
**Reasoning:** [volatility, position sizing, regime risk]
**Risk flags:** [hard stops, invalidation levels]

## 3. SWARM CONSENSUS
**Decision:** {rec.get("final_recommendation")} | **Conviction:** [High/Medium/Low]
**Weighted score:** {rec.get("weighted_score")} (threshold ±0.40)
{('**Scanner validation:** Does AI confirm Entry ₹' + str(sig.get("entry","?")) + ' / SL ₹' + str(sig.get("sl","?")) + '? Comment on R:R.') if sig else ""}

## 4. EXECUTION PLAN
[Entry trigger, position size % of capital, stop management, exit rules]

## 5. INVALIDATION TRIGGERS
[Specific price levels that kill this trade idea]"""


def _call_groq(key: str, prompt: str) -> str:
    from openai import OpenAI  # lazy import
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": _SYS},
                  {"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=4000,
    )
    return r.choices[0].message.content


def _call_gemini(key: str, model: str, prompt: str) -> str:
    import google.generativeai as genai  # lazy import
    genai.configure(api_key=key)
    m = genai.GenerativeModel(model_name=model, system_instruction=_SYS)
    return m.generate_content(prompt).text


def _run_ai(ticker, m, regime, sent, rec, sig, groq_key, gemini_key, gemini_model) -> Optional[str]:
    prompt = _build_prompt(ticker, m, regime, sent, rec, sig)
    if groq_key:
        try:
            with st.spinner("🚀 Analysing with Groq (llama-3.3-70b)…"):
                out = _call_groq(groq_key, prompt)
            st.toast("✅ Groq analysis complete", icon="🚀")
            return out
        except Exception as e:
            st.warning(f"⚠️ Groq failed: {e} — trying Gemini…")
    if gemini_key:
        try:
            with st.spinner(f"🔄 Analysing with {gemini_model}…"):
                out = _call_gemini(gemini_key, gemini_model, prompt)
            st.toast("✅ Gemini analysis complete", icon="🔄")
            return out
        except Exception as e:
            st.error(f"❌ Both APIs failed. Last error: {e}")
            return None
    st.error("❌ No API keys. Add GROQ_API_KEY or GOOGLE_API_KEY to Streamlit Secrets.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE SIGNAL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_open_signals() -> List[Dict]:
    try:
        from signal_tracker import _get_supabase  # local import — safe
        sb = _get_supabase()
        if not sb:
            return []
        res = (sb.table("signals")
               .select("symbol,strategy,signal,cmp,entry,sl,t1,t2,rr,sqi,sqi_grade,sector,regime,created_at")
               .eq("status", "OPEN")
               .order("created_at", desc=True)
               .limit(150)
               .execute())
        return res.data or []
    except Exception as e:
        print(f"[AI Deep Dive] Supabase load error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fp(v) -> str:
    try:
        v = float(v)
        return f"₹{v:,.0f}" if v >= 10000 else f"₹{v:,.1f}" if v >= 100 else f"₹{v:,.2f}"
    except Exception:
        return str(v)

def _sig_col(s):
    return {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#ff9800"}.get(s, "#9e9e9e")

def _risk_col(r):
    return {"HIGH": "#ef5350", "MEDIUM": "#ff9800", "LOW": "#26a69a"}.get(r, "#9e9e9e")

def _regime_bg(code):
    return {"stress": "#c0392b", "accumulate": "#6c3483", "trend": "#0e6655",
            "goldilocks": "#7d3c98", "uncertainty": "#555", "unknown": "#333"
            }.get(code.lower(), "#333")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_ai_deep_dive():
    st.markdown("""
    <style>
    .dd-title{font-size:1.5rem;font-weight:800;letter-spacing:3px;
      background:linear-gradient(120deg,#00d4ff,#FF6B35,#ff4757);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .agent-box{background:rgba(255,255,255,0.04);border-radius:10px;
      border-left:4px solid #7b2cbf;padding:14px 16px;margin:6px 0;}
    .agent-box.bull{border-left-color:#26a69a;}
    .agent-box.bear{border-left-color:#ef5350;}
    .agent-box.neut{border-left-color:#ff9800;}
    .sbadge{display:inline-block;padding:3px 12px;border-radius:20px;
      font-weight:700;font-size:.82rem;letter-spacing:1px;color:#fff;}
    .ctx-box{background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.22);
      border-radius:10px;padding:12px 18px;margin:10px 0;font-size:.88rem;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dd-title">🧠 AI DEEP DIVE</div>', unsafe_allow_html=True)
    st.caption("Multi-Agent Swarm Analysis (ATLAS · ORACLE · SENTINEL) — Groq primary / Gemini fallback")

    # ── API keys from secrets ────────────────────────────────────────────────
    try:
        groq_key   = st.secrets.get("GROQ_API_KEY",   "")
        gemini_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        groq_key = gemini_key = ""

    if not groq_key and not gemini_key:
        st.error("❌ No API keys found. Add `GROQ_API_KEY` and/or `GOOGLE_API_KEY` in Streamlit Secrets → Settings.")
        st.info("Both keys are already in your Streamlit Secrets — if you see this error, try rebooting the app from Streamlit Cloud dashboard.")
        return

    # ── Load open signals from Supabase ──────────────────────────────────────
    with st.spinner("Loading open scanner signals…"):
        open_sigs = _load_open_signals()

    sig_lookup: Dict[str, Dict] = {}
    for s in open_sigs:
        label = (f"{s['symbol']}  ·  {s['strategy']}  ·  {s['signal']}"
                 f"  ·  SQI {s.get('sqi','?')} ({s.get('sqi_grade','?')})"
                 f"  ·  RR {s.get('rr','?')}")
        sig_lookup[label] = s

    # ── Controls ─────────────────────────────────────────────────────────────
    st.divider()
    col_l, col_r = st.columns([3, 1])
    with col_l:
        if sig_lookup:
            options = ["— Enter ticker manually —"] + list(sig_lookup.keys())
            picked = st.selectbox("📡 Pick a live OPEN signal", options, key="dd_pick")
        else:
            picked = None
            st.info("No OPEN signals in Supabase — enter ticker manually.")
        manual = st.text_input("Or type any ticker (e.g. RELIANCE or RELIANCE.NS)", key="dd_manual")
    with col_r:
        gemini_model = st.selectbox("Gemini fallback model",
                                    ["gemini-2.0-flash", "gemini-2.5-flash"], key="dd_gem")
        run_btn = st.button("🚀 Run AI Analysis", type="primary",
                            use_container_width=True, key="dd_run")

    # ── Idle: show open signals table ────────────────────────────────────────
    if not run_btn:
        if open_sigs:
            st.markdown("#### 📋 Open Signals — pick one above to deep-dive")
            rows = [{"Symbol": s["symbol"], "Strategy": s["strategy"], "Signal": s["signal"],
                     "CMP": _fp(s.get("cmp",0)), "Entry": _fp(s.get("entry",0)),
                     "SL": _fp(s.get("sl",0)), "T1": _fp(s.get("t1",0)),
                     "R:R": s.get("rr","—"), "SQI": s.get("sqi","—"),
                     "Grade": s.get("sqi_grade","—"), "Sector": s.get("sector","—")}
                    for s in open_sigs]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=320)
        else:
            st.info("Run a scan first to generate signals, then use this page for AI deep-dive analysis.")
        return

    # ── Resolve ticker ────────────────────────────────────────────────────────
    sig_ctx: Optional[Dict] = None
    ticker = ""
    if picked and picked != "— Enter ticker manually —":
        sig_ctx = sig_lookup[picked]
        ticker  = sig_ctx["symbol"]
    elif manual.strip():
        ticker = manual.strip().upper().replace(".NS", "")
    if not ticker:
        st.error("Please pick a signal or type a ticker symbol.")
        return

    ns_ticker = ticker + ".NS"

    if sig_ctx:
        st.markdown(f"""<div class="ctx-box">
            <b>📡 Scanner Signal</b> &nbsp;·&nbsp; <b>{sig_ctx['symbol']}</b>
            &nbsp;·&nbsp; {sig_ctx['strategy']} &nbsp;·&nbsp; <b>{sig_ctx['signal']}</b>
            &nbsp;·&nbsp; SQI <b>{sig_ctx.get('sqi','?')} ({sig_ctx.get('sqi_grade','?')})</b><br>
            Entry <b>{_fp(sig_ctx.get('entry','?'))}</b> &nbsp;|&nbsp;
            SL <b>{_fp(sig_ctx.get('sl','?'))}</b> &nbsp;|&nbsp;
            T1 <b>{_fp(sig_ctx.get('t1','?'))}</b> &nbsp;|&nbsp;
            T2 <b>{_fp(sig_ctx.get('t2','?'))}</b> &nbsp;|&nbsp;
            R:R <b>{sig_ctx.get('rr','?')}</b>
        </div>""", unsafe_allow_html=True)

    # ── Fetch & compute ───────────────────────────────────────────────────────
    with st.spinner(f"Fetching data for {ns_ticker}…"):
        try:
            df = yf.Ticker(ns_ticker).history(period="6mo")
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            return

    if df is None or df.empty or len(df) < 30:
        st.error(f"Insufficient data for **{ns_ticker}**. Check the ticker symbol.")
        return

    regime = _fetch_regime()
    m = _compute_metrics(df)
    if not m:
        st.error("Unable to compute metrics.")
        return

    sent = _sentiment(m)
    rec  = _decision(m, regime, sent)

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.divider()
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Price",     _fp(m["current_price"]),      f"{m['price_change_1d']:+.2f}%")
    c2.metric("RSI (14)",  f"{m['rsi_14']:.1f}",         m["rsi_status"])
    c3.metric("Volume",    f"{m['volume_ratio']:.2f}×",  "vs 20d avg")
    c4.metric("Sentiment", sent["overall"])
    c5.metric("AI Signal", rec["final_recommendation"])
    c6.metric("Wtd Score", f"{rec['weighted_score']:+.3f}")

    # ── Regime banner ─────────────────────────────────────────────────────────
    bg = _regime_bg(regime.get("code", "unknown"))
    st.markdown(f"""<div style="background:{bg};padding:10px 18px;border-radius:10px;
        color:#fff;margin:10px 0;display:flex;align-items:center;gap:14px;">
        <div><b style="font-size:1.05rem">{regime['regime']}</b>
        <span style="opacity:.8;font-size:.82rem;margin-left:8px">{regime['description']}</span></div>
        <div style="margin-left:auto;font-size:.8rem;text-align:right;opacity:.9">
        VIX <b>{regime['vix']}</b> &nbsp;|&nbsp;
        Nifty <b>{regime.get('nifty_level','?')}</b> ({regime.get('nifty_change',0):+.2f}%)
        &nbsp;|&nbsp; Breadth <b>{regime.get('breadth','?')}</b></div></div>
    """, unsafe_allow_html=True)

    # ── Agent cards ───────────────────────────────────────────────────────────
    a1, a2, a3 = st.columns(3)
    with a1:
        cls = "bull" if rec["atlas_signal"]=="BUY" else ("bear" if rec["atlas_signal"]=="SELL" else "neut")
        st.markdown(f"""<div class="agent-box {cls}">
            <div style="font-size:.72rem;color:#888;letter-spacing:1px">AGENT A · ATLAS · {regime['atlas_weight']}% wt</div>
            <div style="margin-top:7px"><span class="sbadge" style="background:{_sig_col(rec['atlas_signal'])}">{rec['atlas_signal']}</span></div>
            <div style="font-size:.78rem;margin-top:6px;color:#bbb">
            {m.get('trend_short_term')} trend · MACD {m.get('macd_status')} · ADX {m['adx_14']:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with a2:
        cls = "bull" if rec["oracle_signal"]=="BUY" else ("bear" if rec["oracle_signal"]=="SELL" else "neut")
        st.markdown(f"""<div class="agent-box {cls}">
            <div style="font-size:.72rem;color:#888;letter-spacing:1px">AGENT B · ORACLE · {regime['oracle_weight']}% wt</div>
            <div style="margin-top:7px">
            <span class="sbadge" style="background:{_sig_col(rec['oracle_signal'])}">{rec['oracle_signal']}</span>
            <span style="font-size:.75rem;color:#888;margin-left:8px">{rec['oracle_confidence']}% conf</span></div>
            <div style="font-size:.78rem;margin-top:6px;color:#bbb">{sent['overall']} (score {sent['score']})</div>
        </div>""", unsafe_allow_html=True)
    with a3:
        cls = "bear" if rec["sentinel_risk"]=="HIGH" else ("bull" if rec["sentinel_risk"]=="LOW" else "neut")
        st.markdown(f"""<div class="agent-box {cls}">
            <div style="font-size:.72rem;color:#888;letter-spacing:1px">AGENT C · SENTINEL · {regime['sentinel_weight']}% wt</div>
            <div style="margin-top:7px"><span class="sbadge" style="background:{_risk_col(rec['sentinel_risk'])}">RISK: {rec['sentinel_risk']}</span></div>
            <div style="font-size:.78rem;margin-top:6px;color:#bbb">
            Vol {m['realized_volatility_20d']:.1f}% · ATR% {m['atr_pct']:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.plotly_chart(_chart(df, ticker, sig_ctx), use_container_width=True)

    # ── AI Analysis ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"🤖 Multi-Agent AI Analysis — {ticker}")
    analysis = _run_ai(ticker, m, regime, sent, rec, sig_ctx,
                       groq_key, gemini_key, gemini_model)
    if analysis:
        st.markdown(analysis)

    if sent.get("factors"):
        with st.expander("📊 Sentiment Breakdown", expanded=False):
            for f in sent["factors"]:
                st.markdown(f"- {f}")
