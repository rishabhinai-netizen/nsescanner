"""
Fundamental Quality Gate — Prevents buying fundamentally weak stocks
===================================================================
Fetches basic fundamental data from yfinance and applies CANSLIM-inspired filters.

Filters:
- EPS growth QoQ > 0% (current earnings positive)
- Revenue growth > 0% (business not shrinking) 
- Market cap > ₹500 Cr (liquidity requirement)
- PE ratio < 100 (not hyper-speculative)
- Debt/Equity < 2.0 (not over-leveraged)

Signals are tagged, NOT blocked — the gate adds warnings and reduces SQI.
This ensures the scanner never recommends VCP breakout on a company with
declining earnings (classic "dead cat" trap).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import yfinance as yf

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False
    class _MockSt:
        @staticmethod
        def cache_data(ttl=None, show_spinner=False):
            def decorator(func): return func
            return decorator
    st = _MockSt()

logger = logging.getLogger(__name__)


@dataclass
class FundamentalGate:
    """Fundamental quality assessment for a stock."""
    passes: bool  # True if stock passes quality gate
    score: int  # 0-5 (count of passed checks)
    max_score: int  # 5
    grade: str  # "A" / "B" / "C" / "F"
    grade_icon: str
    market_cap_cr: float  # Market cap in Crores
    pe_ratio: float
    eps_growth: Optional[float]  # QoQ EPS growth %
    revenue_growth: Optional[float]  # YoY revenue growth %
    debt_equity: Optional[float]
    warnings: list  # List of failed checks
    details: list  # List of all checks
    sqi_penalty: float  # How much to reduce SQI (0-30)


# Thresholds
MIN_MARKET_CAP_CR = 500  # ₹500 Crore minimum
MAX_PE_RATIO = 100  # No hyper-speculative
MIN_EPS_GROWTH = 0  # EPS must be growing (QoQ)
MIN_REVENUE_GROWTH = 0  # Revenue must not be declining
MAX_DEBT_EQUITY = 2.0  # Leverage limit


@st.cache_data(ttl=86400, show_spinner=False)  # Cache 24 hours
def _fetch_fundamentals(symbol: str) -> Dict:
    """Fetch basic fundamentals from yfinance. Cached for 24 hours."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        if not info or info.get("regularMarketPrice") is None:
            return {}
        
        # Extract key metrics
        result = {
            "market_cap": info.get("marketCap", 0),  # In INR
            "pe_ratio": info.get("trailingPE", 0) or info.get("forwardPE", 0) or 0,
            "eps_trailing": info.get("trailingEps", 0) or 0,
            "revenue_growth": info.get("revenueGrowth", None),  # YoY as decimal
            "earnings_growth": info.get("earningsGrowth", None),  # QoQ as decimal  
            "debt_to_equity": info.get("debtToEquity", None),  # As percentage (e.g., 45.2 = 45.2%)
            "profit_margins": info.get("profitMargins", None),
            "return_on_equity": info.get("returnOnEquity", None),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }
        return result
    except Exception as e:
        logger.debug(f"Fundamentals fetch failed for {symbol}: {e}")
        return {}


def check_fundamental_quality(symbol: str) -> FundamentalGate:
    """
    Run fundamental quality gate on a stock.
    Returns quality assessment with pass/fail and SQI penalty.
    """
    data = _fetch_fundamentals(symbol)
    
    if not data:
        # Can't fetch fundamentals — give benefit of doubt with warning
        return FundamentalGate(
            passes=True, score=0, max_score=5,
            grade="?", grade_icon="❓",
            market_cap_cr=0, pe_ratio=0,
            eps_growth=None, revenue_growth=None, debt_equity=None,
            warnings=["Fundamental data unavailable — proceed with caution"],
            details=["❓ Unable to fetch fundamental data from yfinance"],
            sqi_penalty=5,  # Small penalty for unknown fundamentals
        )
    
    score = 0
    warnings = []
    details = []
    
    # 1. Market Cap Check
    market_cap_cr = data.get("market_cap", 0) / 1e7  # Convert to Crores
    if market_cap_cr >= MIN_MARKET_CAP_CR:
        score += 1
        details.append(f"✅ Market Cap ₹{market_cap_cr:,.0f} Cr (min ₹{MIN_MARKET_CAP_CR} Cr)")
    elif market_cap_cr > 0:
        warnings.append(f"Small cap (₹{market_cap_cr:,.0f} Cr) — higher risk")
        details.append(f"⚠️ Market Cap ₹{market_cap_cr:,.0f} Cr — below ₹{MIN_MARKET_CAP_CR} Cr threshold")
    else:
        details.append("❓ Market cap data unavailable")
    
    # 2. PE Ratio Check
    pe = data.get("pe_ratio", 0)
    if pe > 0 and pe <= MAX_PE_RATIO:
        score += 1
        details.append(f"✅ PE Ratio {pe:.1f} (max {MAX_PE_RATIO})")
    elif pe > MAX_PE_RATIO:
        warnings.append(f"PE {pe:.1f} — speculative valuation")
        details.append(f"⚠️ PE Ratio {pe:.1f} exceeds {MAX_PE_RATIO} — speculative")
    elif pe < 0:
        warnings.append("Negative PE — company losing money")
        details.append("❌ Negative PE ratio — loss-making company")
    else:
        details.append("❓ PE ratio data unavailable")
    
    # 3. Earnings Growth Check
    earnings_growth = data.get("earnings_growth")
    if earnings_growth is not None:
        eps_growth_pct = earnings_growth * 100
        if eps_growth_pct > MIN_EPS_GROWTH:
            score += 1
            details.append(f"✅ Earnings Growth {eps_growth_pct:+.1f}% QoQ")
        else:
            warnings.append(f"Earnings declining ({eps_growth_pct:+.1f}%) — earnings trap risk")
            details.append(f"❌ Earnings Growth {eps_growth_pct:+.1f}% — DECLINING")
    else:
        details.append("❓ Earnings growth data unavailable")
    
    # 4. Revenue Growth Check
    rev_growth = data.get("revenue_growth")
    if rev_growth is not None:
        rev_growth_pct = rev_growth * 100
        if rev_growth_pct > MIN_REVENUE_GROWTH:
            score += 1
            details.append(f"✅ Revenue Growth {rev_growth_pct:+.1f}% YoY")
        else:
            warnings.append(f"Revenue declining ({rev_growth_pct:+.1f}%) — business shrinking")
            details.append(f"❌ Revenue Growth {rev_growth_pct:+.1f}% — DECLINING")
    else:
        details.append("❓ Revenue growth data unavailable")
    
    # 5. Debt/Equity Check
    de = data.get("debt_to_equity")
    if de is not None:
        de_ratio = de / 100  # yfinance gives as percentage
        if de_ratio <= MAX_DEBT_EQUITY:
            score += 1
            details.append(f"✅ Debt/Equity {de_ratio:.2f} (max {MAX_DEBT_EQUITY})")
        else:
            warnings.append(f"High leverage (D/E {de_ratio:.2f}) — financial risk")
            details.append(f"⚠️ Debt/Equity {de_ratio:.2f} exceeds {MAX_DEBT_EQUITY}")
    else:
        details.append("❓ Debt/Equity data unavailable")
    
    # Grade and SQI penalty
    if score >= 5:
        grade, grade_icon = "A", "🟢"
        sqi_penalty = 0
    elif score >= 4:
        grade, grade_icon = "B+", "🟢"
        sqi_penalty = 0
    elif score >= 3:
        grade, grade_icon = "B", "🟡"
        sqi_penalty = 5
    elif score >= 2:
        grade, grade_icon = "C", "🟠"
        sqi_penalty = 10
    else:
        grade, grade_icon = "F", "🔴"
        sqi_penalty = 20
    
    # Extra penalty for specific red flags
    if earnings_growth is not None and earnings_growth < -0.2:
        sqi_penalty += 10  # Big earnings decline
    if pe and pe < 0:
        sqi_penalty += 10  # Loss-making
    
    passes = score >= 2 and len(warnings) <= 2
    
    return FundamentalGate(
        passes=passes,
        score=score,
        max_score=5,
        grade=grade,
        grade_icon=grade_icon,
        market_cap_cr=round(market_cap_cr, 0),
        pe_ratio=round(pe, 1) if pe else 0,
        eps_growth=round(earnings_growth * 100, 1) if earnings_growth is not None else None,
        revenue_growth=round(rev_growth * 100, 1) if rev_growth is not None else None,
        debt_equity=round(de / 100, 2) if de is not None else None,
        warnings=warnings,
        details=details,
        sqi_penalty=min(sqi_penalty, 30),
    )


def batch_fundamental_check(symbols: list, progress_callback=None) -> Dict[str, FundamentalGate]:
    """Check fundamentals for multiple symbols. Returns dict of results."""
    results = {}
    for i, sym in enumerate(symbols):
        if progress_callback and i % 10 == 0:
            progress_callback(i / len(symbols), f"Checking fundamentals: {sym}...")
        try:
            results[sym] = check_fundamental_quality(sym)
        except Exception:
            results[sym] = FundamentalGate(
                passes=True, score=0, max_score=5,
                grade="?", grade_icon="❓",
                market_cap_cr=0, pe_ratio=0,
                eps_growth=None, revenue_growth=None, debt_equity=None,
                warnings=["Check failed"], details=["❓ Error"], sqi_penalty=5,
            )
    return results
