"""
IPO Scanner Module — NSE Scanner Pro v15
==========================================
Detects IPO base formations and breakout signals.

Research foundation (O'Neil Institute study, 250 IPOs 2010-2020):
- IPO base: consolidation 15-30% depth, 10-14 days minimum
- High-volume breakout: ≥150% of 50-day avg volume
- Breakout hit rate: 57%, alpha: +2.79% over 63 days vs Nifty
- Average winner: +19.9%, Average loser: -14.5% (favorable R:R)
- 8-week hold rule: if +20% in 3 weeks → hold 8 full weeks minimum

Indian-specific:
- GMP correlation ~0.8 with listing returns (most useful pre-listing)
- Anchor investor lock-up: Day 30 (76% stocks decline avg 2.6%)
- Public lock-up: Day 90 (50% of supply unlocks)
- PE/VC lock-up: Day 180 (avg -5 to -6% drag)
- Promoter lock-up: Day 540 (largest supply event)

Entry rules:
- Close > IPO base left-side high (pivot)
- Volume ≥ 150% of 50-day average on breakout day
- RS ≥ 80 (stock must be outperforming Nifty since listing)
- Enter within 5% of pivot (not extended)
- Stop loss: 7-8% below entry (IPO stocks are volatile)

Risk management:
- Max 0.75% portfolio risk per IPO trade
- Max 5% position size per IPO
- Max 25% total IPO allocation (3-5 positions)

Data strategy:
- NSE website scraping for: listing date, issue price, subscription data
- yfinance for: price history post-listing
- Breeze API for: intraday confirmation on breakout day
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta, date

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    _HAS_STREAMLIT = True
    def _cache(ttl=3600):
        return st.cache_data(ttl=ttl, show_spinner=False)
except ImportError:
    _HAS_STREAMLIT = False
    def _cache(ttl=3600):
        def decorator(func): return func
        return decorator


# ============================================================================
# IPO DATA STRUCTURES
# ============================================================================

@dataclass
class IPOMetadata:
    """Core IPO information scraped from NSE/public sources."""
    symbol: str
    company_name: str
    listing_date: str          # YYYY-MM-DD
    issue_price: float         # ₹ per share
    listing_price: float       # Opening price on listing day
    listing_gain_pct: float    # (listing_price - issue_price) / issue_price * 100

    # Subscription data
    qib_sub: float             # QIB subscription (times)
    hni_sub: float             # HNI subscription (times)
    retail_sub: float          # Retail subscription (times)
    overall_sub: float         # Total oversubscription (times)

    # Company basics
    issue_size_cr: float       # Issue size in Crores
    market_cap_cr: float       # Market cap at listing

    # Lock-up calendar
    day30_date: str            # Anchor lock-up expiry
    day90_date: str            # Public lock-up expiry
    day180_date: str           # PE/VC lock-up expiry
    day540_date: str           # Promoter lock-up expiry

    sector: str = ""
    exchange: str = "NSE"


@dataclass
class IPOBaseSignal:
    """Result of IPO base detection and scoring."""
    symbol: str
    company_name: str
    listing_date: str
    issue_price: float
    cmp: float                 # Current market price

    # Quality score (0-100)
    quality_score: float
    quality_grade: str         # "STRONG_BUY" / "BUY" / "WATCH" / "AVOID"
    quality_icon: str

    # Base analysis
    base_forming: bool
    base_depth_pct: float      # How deep the base is (15-30% = ideal)
    base_days: int             # Days in base
    pivot_price: float         # Left-side high = breakout level
    weeks_since_listing: int

    # Breakout signal
    breakout_confirmed: bool
    breakout_volume_ratio: float  # Current vol / 50d avg
    rs_rating: float

    # 8-week hold rule
    eight_week_hold_active: bool
    gain_since_listing: float

    # Lock-up alerts
    lock_up_alerts: List[str]
    next_lockup_date: str
    next_lockup_days: int
    next_lockup_type: str

    # Entry parameters
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float

    # Score breakdown
    listing_score: float       # 15%
    subscription_score: float  # 15%
    volume_score: float        # 15%
    base_score: float          # 15%
    fundamental_score: float   # 15%
    institutional_score: float # 10%
    sector_score: float        # 10%
    rs_score: float            # 5%

    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[IPOMetadata] = None


# ============================================================================
# NSE IPO DATA SCRAPER
# ============================================================================

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/new-issues/ipo",
    "X-Requested-With": "XMLHttpRequest",
}


def scrape_nse_ipo_list() -> List[Dict]:
    """
    Scrape recent IPO listings from NSE India.
    Returns last 2 years of listings with issue price and dates.
    Falls back to empty list if scraping fails (yfinance still works for price data).
    """
    try:
        session = requests.Session()
        # First, establish session cookies
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)

        # Fetch recent IPO data
        url = "https://www.nseindia.com/api/ipo-current-allotment"
        resp = session.get(url, headers=NSE_HEADERS, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            listings = []
            for item in data:
                try:
                    listings.append({
                        "symbol": item.get("symbol", ""),
                        "company_name": item.get("companyName", ""),
                        "listing_date": item.get("listingDate", ""),
                        "issue_price": float(item.get("issuePrice", 0) or 0),
                        "listing_price": float(item.get("listingOpenPrice", 0) or 0),
                        "qib_sub": float(item.get("qibSubscribed", 0) or 0),
                        "hni_sub": float(item.get("niiSubscribed", 0) or 0),
                        "retail_sub": float(item.get("retailSubscribed", 0) or 0),
                        "overall_sub": float(item.get("totalSubscribed", 0) or 0),
                        "issue_size_cr": float(item.get("issueSizeCrore", 0) or 0),
                    })
                except Exception:
                    continue
            if listings:
                return listings
    except Exception as e:
        logger.debug(f"NSE IPO scrape failed: {e}")

    # Fallback: try NSE capital markets API
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        url = "https://www.nseindia.com/api/reportsmf?index=IPO"
        resp = session.get(url, headers=NSE_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json() if isinstance(resp.json(), list) else []
    except Exception:
        pass

    return []


def scrape_subscription_data(symbol: str) -> Dict:
    """
    Try to get subscription data for a specific IPO from NSE.
    Returns empty dict if unavailable.
    """
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        url = f"https://www.nseindia.com/api/ipo-detail?symbol={symbol}"
        resp = session.get(url, headers=NSE_HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "qib_sub": float(data.get("qibSubscribed", 0) or 0),
                "hni_sub": float(data.get("niiSubscribed", 0) or 0),
                "retail_sub": float(data.get("retailSubscribed", 0) or 0),
                "overall_sub": float(data.get("totalSubscribed", 0) or 0),
                "issue_price": float(data.get("issuePrice", 0) or 0),
                "listing_date": data.get("listingDate", ""),
                "listing_price": float(data.get("listingOpenPrice", 0) or 0),
            }
    except Exception:
        pass
    return {}


def get_ipo_price_data(symbol: str, listing_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch post-listing price history from yfinance.
    Returns DataFrame with OHLCV data from listing date.
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        start = listing_date  # "YYYY-MM-DD"
        df = ticker.history(start=start, auto_adjust=True)
        if df.empty or len(df) < 5:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df["symbol"] = symbol
        return df
    except Exception as e:
        logger.debug(f"IPO price fetch failed for {symbol}: {e}")
        return None


def build_ipo_metadata(raw: Dict) -> Optional[IPOMetadata]:
    """Build IPOMetadata from raw scraped data."""
    try:
        symbol = raw.get("symbol", "").strip().upper()
        listing_date_str = raw.get("listing_date", "")

        if not symbol or not listing_date_str:
            return None

        # Parse listing date
        for fmt in ["%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
            try:
                listing_dt = datetime.strptime(listing_date_str, fmt)
                listing_date_str = listing_dt.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
        else:
            return None

        listing_dt = datetime.strptime(listing_date_str, "%Y-%m-%d")
        issue_price = float(raw.get("issue_price", 0) or 0)
        listing_price = float(raw.get("listing_price", 0) or 0)

        if listing_price > 0 and issue_price > 0:
            listing_gain = (listing_price - issue_price) / issue_price * 100
        else:
            listing_gain = 0.0

        # Lock-up dates
        def lockup_date(days: int) -> str:
            return (listing_dt + timedelta(days=days)).strftime("%Y-%m-%d")

        return IPOMetadata(
            symbol=symbol,
            company_name=raw.get("company_name", symbol),
            listing_date=listing_date_str,
            issue_price=issue_price,
            listing_price=listing_price,
            listing_gain_pct=round(listing_gain, 2),
            qib_sub=float(raw.get("qib_sub", 0) or 0),
            hni_sub=float(raw.get("hni_sub", 0) or 0),
            retail_sub=float(raw.get("retail_sub", 0) or 0),
            overall_sub=float(raw.get("overall_sub", 0) or 0),
            issue_size_cr=float(raw.get("issue_size_cr", 0) or 0),
            market_cap_cr=0.0,
            day30_date=lockup_date(30),
            day90_date=lockup_date(90),
            day180_date=lockup_date(180),
            day540_date=lockup_date(540),
            sector=raw.get("sector", ""),
            exchange="NSE",
        )
    except Exception as e:
        logger.debug(f"build_ipo_metadata failed: {e}")
        return None


# ============================================================================
# IPO BASE DETECTION
# ============================================================================

def detect_ipo_base(df: pd.DataFrame, issue_price: float) -> Dict:
    """
    Detect IPO base formation.

    IPO base criteria (O'Neil Institute):
    - Depth: 15-30% from left-side high (too deep = broken, too shallow = not a base)
    - Duration: ≥ 10 trading days (≥ 14 preferred)
    - No prior uptrend required (differs from VCP)
    - Left-side high = breakout pivot
    - Volume: should dry up during base (< 50-day avg)
    """
    result = {
        "base_forming": False,
        "base_depth_pct": 0.0,
        "base_days": 0,
        "pivot_price": 0.0,
        "volume_dry_up": False,
        "quality": "NONE",
        "score": 0,
    }

    if df is None or len(df) < 10:
        return result

    # Left-side high: highest close in first 50% of available data (or first 30 days)
    lookback = min(len(df), 60)
    recent = df.iloc[-lookback:]

    # Find the left-side high (ideally the early pop after listing)
    peak_idx = recent["high"].idxmax()
    peak_loc = recent.index.get_loc(peak_idx) if hasattr(recent.index, 'get_loc') else len(recent) // 2

    if peak_loc < 3:  # Peak too recent = not an established base
        peak_loc = min(peak_loc + 5, len(recent) - 1)

    left_high = float(recent["high"].iloc[:peak_loc + 1].max())
    base_start_idx = peak_loc

    # Base period: from peak to now
    base_period = recent.iloc[base_start_idx:]
    if len(base_period) < 5:
        return result

    base_low = float(base_period["low"].min())
    base_high = float(base_period["high"].max())
    current_price = float(df["close"].iloc[-1])

    # Base depth from left-side high
    base_depth = (left_high - base_low) / left_high * 100 if left_high > 0 else 0
    base_days = len(base_period)

    # Volume dry-up in base
    base_vol_avg = base_period["volume"].mean()
    full_vol_avg = df["volume"].mean()
    vol_dry_up = base_vol_avg < full_vol_avg * 0.7

    # Base quality assessment
    ideal_depth = 15 <= base_depth <= 30
    min_duration = base_days >= 10
    preferred_duration = base_days >= 14

    pivot = round(left_high, 2)

    if ideal_depth and preferred_duration:
        quality = "EXCELLENT"
        score = 90
    elif ideal_depth and min_duration:
        quality = "GOOD"
        score = 75
    elif base_depth <= 35 and min_duration:
        quality = "FAIR"
        score = 55
    elif base_depth > 35:
        quality = "TOO_DEEP"  # Likely broken IPO
        score = 20
    else:
        quality = "TOO_EARLY"  # Not enough time in base
        score = 30

    base_forming = quality in ("EXCELLENT", "GOOD", "FAIR") and current_price >= pivot * 0.95

    result.update({
        "base_forming": base_forming,
        "base_depth_pct": round(base_depth, 1),
        "base_days": base_days,
        "pivot_price": pivot,
        "volume_dry_up": vol_dry_up,
        "quality": quality,
        "score": score,
        "base_low": round(base_low, 2),
    })
    return result


def check_breakout_signal(df: pd.DataFrame, pivot: float) -> Dict:
    """
    Check if current day has a confirmed IPO breakout.
    Requirements: close > pivot AND volume ≥ 150% of 50-day average.
    """
    if df is None or len(df) < 10:
        return {"breakout": False, "vol_ratio": 0, "entry_gap": 0}

    latest = df.iloc[-1]
    cmp = float(latest["close"])

    # 50-day volume average
    vol_avg_50 = df["volume"].iloc[-min(50, len(df)):].mean()
    vol_ratio = float(latest["volume"]) / (vol_avg_50 + 1)

    # Breakout condition
    breakout_price = cmp >= pivot * 0.998  # Within 0.2% tolerance
    breakout_volume = vol_ratio >= 1.5     # 150% of 50d avg

    breakout = breakout_price and breakout_volume

    # Entry gap from pivot (must be within 5% to avoid chasing)
    entry_gap = (cmp - pivot) / pivot * 100 if pivot > 0 else 0

    return {
        "breakout": breakout,
        "vol_ratio": round(vol_ratio, 2),
        "entry_gap": round(entry_gap, 2),
        "within_buy_range": -1 <= entry_gap <= 5,
    }


def check_eight_week_hold(df: pd.DataFrame, issue_price: float) -> Dict:
    """
    Check 8-week hold rule: if stock gains ≥20% within first 3 weeks → hold 8 weeks.
    """
    if df is None or len(df) < 5:
        return {"active": False, "gain_3w": 0, "weeks_held": 0}

    listing_price = float(df["close"].iloc[0])  # First trading day close
    current = float(df["close"].iloc[-1])

    # 3-week high from listing
    three_week_data = df.iloc[:min(15, len(df))]
    high_3w = float(three_week_data["high"].max())

    gain_3w = (high_3w - listing_price) / listing_price * 100 if listing_price > 0 else 0
    gain_current = (current - listing_price) / listing_price * 100 if listing_price > 0 else 0

    weeks_since_listing = len(df) / 5  # Approximate trading weeks

    # Rule: if 20%+ gain in 3 weeks → hold until 8 weeks
    eight_week_active = gain_3w >= 20 and weeks_since_listing < 8

    return {
        "active": eight_week_active,
        "gain_3w": round(gain_3w, 1),
        "gain_current": round(gain_current, 1),
        "weeks_held": round(weeks_since_listing, 1),
        "hold_until_weeks": 8 if eight_week_active else None,
    }


def compute_lockup_alerts(metadata: IPOMetadata) -> List[Dict]:
    """
    Compute upcoming lock-up expiry alerts.
    Returns list of alerts sorted by date.
    """
    today = datetime.now().date()
    alerts = []

    lockups = [
        ("Day 30 Anchor Lock-up", metadata.day30_date, "76% of stocks decline avg 2.6%", "HIGH"),
        ("Day 90 Public Lock-up", metadata.day90_date, "50% of total supply unlocks", "HIGH"),
        ("Day 180 PE/VC Lock-up", metadata.day180_date, "PE/VC exit pressure (-5 to -6%)", "MEDIUM"),
        ("Day 540 Promoter Lock-up", metadata.day540_date, "Largest supply event", "LOW"),
    ]

    for name, date_str, impact, severity in lockups:
        try:
            lockup_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            days_away = (lockup_dt - today).days

            # Show alert 5 days before
            if -5 <= days_away <= 30:  # Between 5 days before and 30 days after
                alerts.append({
                    "name": name,
                    "date": date_str,
                    "days_away": days_away,
                    "impact": impact,
                    "severity": severity,
                    "status": "IMMINENT" if days_away <= 0 else ("UPCOMING" if days_away <= 5 else "WATCH"),
                })
        except Exception:
            continue

    return sorted(alerts, key=lambda x: x["days_away"])


# ============================================================================
# 8-FACTOR QUALITY SCORING
# ============================================================================

def compute_ipo_quality_score(
    df: pd.DataFrame,
    metadata: IPOMetadata,
    base_data: Dict,
    nifty_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute 8-factor IPO quality score (0-100).

    Factors:
    1. Listing Performance (15%) — strong listing = institutional validation
    2. Subscription Quality (15%) — QIB > 10x = smart money approved
    3. Volume Profile (15%) — accumulation pattern post-listing
    4. Base Formation (15%) — ideal depth and duration
    5. Fundamentals (15%) — market cap, profitability
    6. Institutional Activity (10%) — buying post-listing
    7. Sector (10%) — sector in leading quadrant
    8. RS vs Nifty (5%) — outperforming benchmark
    """
    scores = {}
    details = []

    # 1. Listing Performance (15%)
    listing_gain = metadata.listing_gain_pct if metadata else 0
    if listing_gain >= 50:
        listing_score = 95
        details.append(f"✅ Stellar listing: +{listing_gain:.1f}% — strong institutional demand")
    elif listing_gain >= 20:
        listing_score = 80
        details.append(f"✅ Strong listing: +{listing_gain:.1f}%")
    elif listing_gain >= 10:
        listing_score = 65
        details.append(f"✅ Positive listing: +{listing_gain:.1f}%")
    elif listing_gain >= 0:
        listing_score = 45
        details.append(f"⚠️ Flat listing: +{listing_gain:.1f}%")
    else:
        listing_score = max(0, 30 + listing_gain)  # Decreases with bigger losses
        details.append(f"❌ Weak listing: {listing_gain:.1f}% — avoid unless recovering")
    scores["listing"] = round(listing_score, 1)

    # 2. Subscription Quality (15%)
    qib = metadata.qib_sub if metadata else 0
    overall = metadata.overall_sub if metadata else 0
    if qib >= 50:
        sub_score = 95
        details.append(f"✅ QIB {qib:.0f}x — exceptional institutional appetite")
    elif qib >= 30:
        sub_score = 85
        details.append(f"✅ QIB {qib:.0f}x — strong institutional interest")
    elif qib >= 10:
        sub_score = 70
        details.append(f"✅ QIB {qib:.0f}x — good institutional coverage")
    elif qib >= 5:
        sub_score = 50
        details.append(f"⚠️ QIB {qib:.0f}x — moderate institutional interest")
    elif overall >= 30:
        sub_score = 55
        details.append(f"⚠️ QIB low but overall {overall:.0f}x subscribed")
    else:
        sub_score = 25
        details.append(f"❌ QIB {qib:.0f}x — weak institutional interest, high risk")
    scores["subscription"] = round(sub_score, 1)

    # 3. Volume Profile (15%)
    if df is not None and len(df) > 10:
        recent_vol = df["volume"].iloc[-5:].mean()
        early_vol = df["volume"].iloc[:5].mean()
        if early_vol > 0:
            vol_trend = recent_vol / early_vol
            if 0.3 <= vol_trend <= 0.7:
                vol_score = 80  # Volume drying up = good base forming
                details.append(f"✅ Volume contracting to {vol_trend:.1f}x listing vol — healthy base")
            elif vol_trend < 0.3:
                vol_score = 60  # Very dry = some concern about interest
                details.append(f"⚠️ Very low volume ({vol_trend:.1f}x listing) — watch for breakout")
            elif vol_trend > 2:
                vol_score = 85  # Unusual volume = activity
                details.append(f"✅ Volume picking up ({vol_trend:.1f}x) — potential breakout forming")
            else:
                vol_score = 55
                details.append(f"⚪ Normal volume ({vol_trend:.1f}x listing vol)")
        else:
            vol_score = 50
    else:
        vol_score = 50
    scores["volume"] = round(vol_score, 1)

    # 4. Base Formation (15%)
    base_score = base_data.get("score", 50)
    base_quality = base_data.get("quality", "NONE")
    if base_quality == "EXCELLENT":
        details.append(f"✅ IPO base: Excellent depth {base_data.get('base_depth_pct', 0):.1f}%, {base_data.get('base_days', 0)} days")
    elif base_quality == "GOOD":
        details.append(f"✅ IPO base: Good depth {base_data.get('base_depth_pct', 0):.1f}%")
    elif base_quality == "TOO_DEEP":
        details.append(f"❌ Base too deep ({base_data.get('base_depth_pct', 0):.1f}%) — potential distribution")
    else:
        details.append(f"⚪ Base forming ({base_data.get('base_depth_pct', 0):.1f}% depth, {base_data.get('base_days', 0)} days)")
    scores["base"] = round(base_score, 1)

    # 5. Fundamentals (15%) — use market cap as proxy if available
    issue_size = metadata.issue_size_cr if metadata else 0
    if issue_size >= 2000:
        fund_score = 80
        details.append(f"✅ Large issue size ₹{issue_size:,.0f} Cr — strong fundamental backing")
    elif issue_size >= 500:
        fund_score = 65
        details.append(f"✅ Decent issue size ₹{issue_size:,.0f} Cr")
    elif issue_size > 0:
        fund_score = 50
        details.append(f"⚪ Smaller issue ₹{issue_size:,.0f} Cr — higher risk")
    else:
        fund_score = 50
        details.append("❓ Fundamental data unavailable")
    scores["fundamental"] = round(fund_score, 1)

    # 6. Institutional Activity post-listing (10%)
    # Proxy: price above listing price shows institutional support
    if df is not None and len(df) > 5 and metadata and metadata.listing_price > 0:
        cmp = float(df["close"].iloc[-1])
        vs_listing = (cmp / metadata.listing_price - 1) * 100
        if vs_listing > 20:
            inst_score = 85
            details.append(f"✅ Trading {vs_listing:.1f}% above listing price — institutional accumulation")
        elif vs_listing > 5:
            inst_score = 70
            details.append(f"✅ Trading {vs_listing:.1f}% above listing — holding well")
        elif vs_listing > -10:
            inst_score = 50
            details.append(f"⚪ Near listing price ({vs_listing:+.1f}%)")
        else:
            inst_score = 30
            details.append(f"❌ Below listing by {vs_listing:.1f}% — weak post-listing demand")
    else:
        inst_score = 50
    scores["institutional"] = round(inst_score, 1)

    # 7. Sector (10%) — using sector string for now, RRG integration possible
    # Default neutral if no RRG data
    sector_score = 60  # Slightly positive (IPOs usually in hot sectors)
    scores["sector"] = sector_score

    # 8. RS vs Nifty (5%)
    if df is not None and nifty_df is not None and len(df) >= 10 and len(nifty_df) >= 10:
        from data_engine import Indicators
        rs = Indicators.relative_strength(df, nifty_df, period=min(len(df) - 1, 21))
        if rs >= 80:
            rs_score = 90
            details.append(f"✅ RS {rs:.0f} — outperforming Nifty strongly")
        elif rs >= 60:
            rs_score = 65
            details.append(f"⚪ RS {rs:.0f} — near-market performance")
        else:
            rs_score = 35
            details.append(f"⚠️ RS {rs:.0f} — underperforming Nifty")
        scores["rs"] = round(rs_score, 1)
    else:
        scores["rs"] = 55

    # Composite with weights
    weights = {
        "listing": 0.15, "subscription": 0.15, "volume": 0.15,
        "base": 0.15, "fundamental": 0.15, "institutional": 0.10,
        "sector": 0.10, "rs": 0.05,
    }
    composite = sum(scores[k] * weights[k] for k in weights)
    composite = round(min(max(composite, 0), 100), 1)

    if composite >= 80:
        grade, icon = "STRONG_BUY", "🏆"
    elif composite >= 60:
        grade, icon = "BUY", "💪"
    elif composite >= 40:
        grade, icon = "WATCH", "👀"
    else:
        grade, icon = "AVOID", "⛔"

    return {
        "composite": composite,
        "grade": grade,
        "icon": icon,
        "scores": scores,
        "details": details,
    }


# ============================================================================
# MAIN SCANNER
# ============================================================================

def scan_ipo_universe(
    nifty_df: Optional[pd.DataFrame] = None,
    max_weeks: int = 52,
    min_quality: float = 40,
    rrg_data: Dict = None,
) -> List[IPOBaseSignal]:
    """
    Scan all recent IPOs for base formations and breakout signals.

    Args:
        nifty_df: Nifty data for RS calculation
        max_weeks: Only scan IPOs listed within this many weeks
        min_quality: Minimum quality score to include (0-100)
        rrg_data: RRG sector rotation data for sector scoring

    Returns:
        List of IPOBaseSignal, sorted by quality score descending.
    """
    results = []

    # Get IPO list from NSE
    raw_ipos = scrape_nse_ipo_list()

    if not raw_ipos:
        logger.warning("NSE IPO scrape returned empty. Using fallback approach.")
        # Return empty — UI will show scrape failure message
        return []

    cutoff_date = datetime.now() - timedelta(weeks=max_weeks)

    for raw in raw_ipos:
        try:
            meta = build_ipo_metadata(raw)
            if not meta:
                continue

            # Only recent IPOs
            listing_dt = datetime.strptime(meta.listing_date, "%Y-%m-%d")
            if listing_dt < cutoff_date:
                continue

            # Get price history
            df = get_ipo_price_data(meta.symbol, meta.listing_date)
            if df is None or len(df) < 5:
                continue

            cmp = float(df["close"].iloc[-1])
            weeks_since = round(len(df) / 5, 1)

            # Base analysis
            base_data = detect_ipo_base(df, meta.issue_price)

            # Quality score
            quality_data = compute_ipo_quality_score(df, meta, base_data, nifty_df)

            if quality_data["composite"] < min_quality:
                continue

            # Breakout check
            pivot = base_data.get("pivot_price", cmp)
            breakout_data = check_breakout_signal(df, pivot)

            # 8-week hold
            eight_week = check_eight_week_hold(df, meta.issue_price)

            # Lock-up alerts
            lockup_alerts = compute_lockup_alerts(meta)
            lockup_strs = []
            next_lockup_date = ""
            next_lockup_days = 999
            next_lockup_type = ""

            for alert in lockup_alerts:
                if alert["days_away"] <= 30:
                    status = alert["status"]
                    icon = "🚨" if status == "IMMINENT" else "⚠️"
                    lockup_strs.append(
                        f"{icon} {alert['name']} in {alert['days_away']} days ({alert['date']}): {alert['impact']}"
                    )
                    if alert["days_away"] < next_lockup_days:
                        next_lockup_days = alert["days_away"]
                        next_lockup_date = alert["date"]
                        next_lockup_type = alert["name"]

            # RS rating
            rs = 55.0
            if nifty_df is not None and len(df) >= 10:
                from data_engine import Indicators
                rs = Indicators.relative_strength(df, nifty_df, period=min(len(df) - 1, 21))

            # Entry parameters
            entry = pivot if not breakout_data["breakout"] else cmp
            stop_loss = round(entry * 0.92, 2)  # 8% stop for IPOs
            risk = entry - stop_loss
            target_1 = round(entry + 2 * risk, 2)
            target_2 = round(entry + 3.5 * risk, 2)
            rr = round((target_1 - entry) / risk, 1) if risk > 0 else 0

            # Gain since listing
            gain_since_listing = (cmp - meta.issue_price) / meta.issue_price * 100 if meta.issue_price > 0 else 0

            sig = IPOBaseSignal(
                symbol=meta.symbol,
                company_name=meta.company_name,
                listing_date=meta.listing_date,
                issue_price=meta.issue_price,
                cmp=round(cmp, 2),
                quality_score=quality_data["composite"],
                quality_grade=quality_data["grade"],
                quality_icon=quality_data["icon"],
                base_forming=base_data["base_forming"],
                base_depth_pct=base_data["base_depth_pct"],
                base_days=base_data["base_days"],
                pivot_price=pivot,
                weeks_since_listing=int(weeks_since),
                breakout_confirmed=breakout_data["breakout"],
                breakout_volume_ratio=breakout_data["vol_ratio"],
                rs_rating=round(rs, 1),
                eight_week_hold_active=eight_week["active"],
                gain_since_listing=round(gain_since_listing, 1),
                lock_up_alerts=lockup_strs,
                next_lockup_date=next_lockup_date,
                next_lockup_days=next_lockup_days,
                next_lockup_type=next_lockup_type,
                entry_price=round(entry, 2),
                stop_loss=round(stop_loss, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                risk_reward=rr,
                listing_score=quality_data["scores"].get("listing", 50),
                subscription_score=quality_data["scores"].get("subscription", 50),
                volume_score=quality_data["scores"].get("volume", 50),
                base_score=quality_data["scores"].get("base", 50),
                fundamental_score=quality_data["scores"].get("fundamental", 50),
                institutional_score=quality_data["scores"].get("institutional", 50),
                sector_score=quality_data["scores"].get("sector", 60),
                rs_score=quality_data["scores"].get("rs", 55),
                reasons=quality_data["details"],
                warnings=lockup_strs,
                metadata=meta,
            )
            results.append(sig)

        except Exception as e:
            logger.debug(f"IPO scan error for {raw.get('symbol', '?')}: {e}")
            continue

    results.sort(key=lambda x: (x.breakout_confirmed, x.quality_score), reverse=True)
    return results


def scan_single_ipo(symbol: str, listing_date: str, issue_price: float,
                    nifty_df: Optional[pd.DataFrame] = None,
                    sub_data: Dict = None) -> Optional[IPOBaseSignal]:
    """
    Scan a single manually-specified IPO.
    Useful as fallback when NSE scraping fails for specific stocks.
    """
    try:
        df = get_ipo_price_data(symbol, listing_date)
        if df is None or len(df) < 5:
            return None

        raw = {
            "symbol": symbol,
            "company_name": symbol,
            "listing_date": listing_date,
            "issue_price": issue_price,
            "listing_price": float(df["open"].iloc[0]) if not df.empty else 0,
            **(sub_data or {}),
        }

        # Estimate listing gain from first day open
        listing_price = float(df["open"].iloc[0]) if not df.empty else issue_price
        listing_gain = (listing_price - issue_price) / issue_price * 100 if issue_price > 0 else 0
        raw["listing_gain_pct"] = listing_gain
        raw["listing_price"] = listing_price

        meta = build_ipo_metadata(raw)
        if not meta:
            return None

        base_data = detect_ipo_base(df, issue_price)
        quality_data = compute_ipo_quality_score(df, meta, base_data, nifty_df)

        cmp = float(df["close"].iloc[-1])
        pivot = base_data.get("pivot_price", cmp)
        breakout_data = check_breakout_signal(df, pivot)
        eight_week = check_eight_week_hold(df, issue_price)
        lockup_alerts = compute_lockup_alerts(meta)
        lockup_strs = [
            f"⚠️ {a['name']} in {a['days_away']} days" for a in lockup_alerts if a["days_away"] <= 30
        ]

        rs = 55.0
        if nifty_df is not None and len(df) >= 10:
            from data_engine import Indicators
            rs = Indicators.relative_strength(df, nifty_df, period=min(len(df) - 1, 21))

        entry = pivot if not breakout_data["breakout"] else cmp
        stop_loss = round(entry * 0.92, 2)
        risk = entry - stop_loss
        target_1 = round(entry + 2 * risk, 2)
        target_2 = round(entry + 3.5 * risk, 2)
        rr = round((target_1 - entry) / risk, 1) if risk > 0 else 0
        gain_since = (cmp - issue_price) / issue_price * 100 if issue_price > 0 else 0
        weeks_since = round(len(df) / 5, 1)

        next_lockup_date = ""
        next_lockup_days = 999
        next_lockup_type = ""
        for alert in lockup_alerts:
            if alert["days_away"] < next_lockup_days:
                next_lockup_days = alert["days_away"]
                next_lockup_date = alert["date"]
                next_lockup_type = alert["name"]

        return IPOBaseSignal(
            symbol=symbol, company_name=symbol,
            listing_date=listing_date, issue_price=issue_price,
            cmp=round(cmp, 2),
            quality_score=quality_data["composite"],
            quality_grade=quality_data["grade"],
            quality_icon=quality_data["icon"],
            base_forming=base_data["base_forming"],
            base_depth_pct=base_data["base_depth_pct"],
            base_days=base_data["base_days"],
            pivot_price=pivot,
            weeks_since_listing=int(weeks_since),
            breakout_confirmed=breakout_data["breakout"],
            breakout_volume_ratio=breakout_data["vol_ratio"],
            rs_rating=round(rs, 1),
            eight_week_hold_active=eight_week["active"],
            gain_since_listing=round(gain_since, 1),
            lock_up_alerts=lockup_strs,
            next_lockup_date=next_lockup_date,
            next_lockup_days=next_lockup_days,
            next_lockup_type=next_lockup_type,
            entry_price=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            risk_reward=rr,
            listing_score=quality_data["scores"].get("listing", 50),
            subscription_score=quality_data["scores"].get("subscription", 50),
            volume_score=quality_data["scores"].get("volume", 50),
            base_score=quality_data["scores"].get("base", 50),
            fundamental_score=quality_data["scores"].get("fundamental", 50),
            institutional_score=quality_data["scores"].get("institutional", 50),
            sector_score=quality_data["scores"].get("sector", 60),
            rs_score=quality_data["scores"].get("rs", 55),
            reasons=quality_data["details"],
            warnings=lockup_strs,
            metadata=meta,
        )
    except Exception as e:
        logger.warning(f"scan_single_ipo({symbol}) failed: {e}")
        return None
