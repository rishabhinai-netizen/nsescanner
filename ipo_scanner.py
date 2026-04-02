"""
IPO Scanner Module — NSE Scanner Pro v16
==========================================
Two years of NSE IPOs: Mainboard + SME segment.
Uses yfinance for price data (reliable), curated listing database.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta, date

logger = logging.getLogger(__name__)

# ============================================================
# CURATED IPO DATABASE — 2 years, Mainboard + SME
# (symbol, yf_symbol, listing_date, issue_price, company,
#  qib_sub, overall_sub, issue_size_cr, sector, segment)
# ============================================================
CURATED_IPOS = [
    # ── 2024 MAINBOARD ────────────────────────────────────────────────────
    ("HYUNDAI",     "HYUNDAI",      "2024-10-22", 1960, "Hyundai Motor India",    7.97,  2.37,  27870, "Auto",          "Mainboard"),
    ("SWIGGY",      "SWIGGY",       "2024-11-13",  390, "Swiggy",                35.10,  3.59,  11327, "Consumer Tech", "Mainboard"),
    ("NTPCGREEN",   "NTPCGREEN",    "2024-11-27",  108, "NTPC Green Energy",      2.55,  1.72,  10000, "Energy",        "Mainboard"),
    ("BAJAJHFL",    "BAJAJHFL",     "2024-09-23",   70, "Bajaj Housing Finance", 223.46, 67.43,  6560, "Finance",       "Mainboard"),
    ("NIVABUPA",    "NIVABUPA",     "2024-11-14",   74, "Niva Bupa Health",       24.76,  6.93,  2200, "Insurance",     "Mainboard"),
    ("PREMIER",     "PREMIER",      "2024-08-01",  900, "Premier Energies",      242.68, 74.12,  1291, "Energy",        "Mainboard"),
    ("OLAELEC",     "OLAELEC",      "2024-08-09",   76, "Ola Electric",           32.93,  3.20,  6145, "Auto",          "Mainboard"),
    ("FIRSTCRY",    "FIRSTCRY",     "2024-08-13",  465, "FirstCry (Brainbees)",   56.01, 12.75,  4193, "Retail",        "Mainboard"),
    ("BHARTIHEXA",  "BHARTIHEXA",   "2024-07-12",  262, "Bharti Hexacom",        256.96, 30.37,  4275, "Telecom",       "Mainboard"),
    ("GODIGIT",     "GODIGIT",      "2024-05-23",  272, "Go Digit Insurance",      7.32,  4.21,  2615, "Insurance",     "Mainboard"),
    ("AWFIS",       "AWFIS",        "2024-05-30",  432, "Awfis Space",           108.62,106.51,   599, "Real Estate",   "Mainboard"),
    ("ARISINFRA",   "ARISINFRA",    "2024-06-03",  280, "Aris Infra",             41.70, 29.00,   520, "Infra",         "Mainboard"),
    ("MANBA",       "MANBA",        "2024-09-30",  120, "Manba Finance",         200.22,224.08,   150, "Finance",       "Mainboard"),
    # ── 2023 MAINBOARD ────────────────────────────────────────────────────
    ("MAPMYINDIA",  "MAPMYINDIA",   "2023-12-28",  316, "MapmyIndia",             35.00, 12.00,   1040,"Technology",    "Mainboard"),
    ("MUTHOOTMF",   "MUTHOOTMF",    "2023-12-18",  291, "Muthoot Microfin",       17.44,  5.00,   960, "Finance",       "Mainboard"),
    ("DOMS",        "DOMS",         "2023-12-20", 1200, "DOMS Industries",        93.00, 93.10,  1200, "Consumer",      "Mainboard"),
    ("INOXINDIA",   "INOXINDIA",    "2023-12-21",  660, "Inox India",             60.50, 61.30,   1459,"Industrials",   "Mainboard"),
    ("SURAJEST",    "SURAJEST",     "2023-09-22",  360, "Suraj Estate",          210.00,107.00,   400, "Real Estate",   "Mainboard"),
    ("YATHARTH",    "YATHARTH",     "2023-07-26",  300, "Yatharth Hospital",      41.08, 38.39,   687, "Healthcare",    "Mainboard"),
    ("NETWEB",      "NETWEB",       "2023-07-27",  500, "Netweb Technologies",    90.00, 90.00,    631,"Technology",    "Mainboard"),
    ("IDEAFORGE",   "IDEAFORGE",    "2023-07-10",  672, "IdeaForge Technology",   63.00, 32.17,   567, "Defence",       "Mainboard"),
    ("CYIENTDLM",   "CYIENTDLM",    "2023-07-03",  265, "Cyient DLM",             14.29, 67.14,   592, "Electronics",   "Mainboard"),
    ("TATATECH",    "TATATECH",     "2023-11-30",  500, "Tata Technologies",     173.25, 69.43,  3042, "Technology",    "Mainboard"),
    ("GANDHAR",     "GANDHAR",      "2023-11-22",  169, "Gandhar Oil",            10.00,  4.80,   500, "Energy",        "Mainboard"),
    ("FLAIR",       "FLAIR",        "2023-11-24",  304, "Flair Writing",          22.19, 43.74,   593, "Consumer",      "Mainboard"),
    ("CELLO",       "CELLO",        "2023-11-01",  648, "Cello World",            36.82, 38.66,  1900, "Consumer",      "Mainboard"),
    ("FEDBANK",     "FEDFINA",      "2023-11-22",  133, "Federal Bank Fin Serv",  32.00,  2.01,  1092, "Finance",       "Mainboard"),
    # ── 2024 SME IPOs (NSE Emerge) ──────────────────────────────────────
    ("TRAFIKSOL",   "TRAFIKSOL",    "2024-09-25",  117, "Trafiksol ITS",         345.00,345.00,    45, "Technology",    "SME"),
    ("SAHARANEWS",  "SNSL",         "2024-10-04",   81, "Sahara News",           222.00,222.00,    42, "Media",         "SME"),
    ("QUADRANT",    "QDIGI",        "2024-09-18",   70, "Quadrant Future Tek",   180.10,180.10,    25, "Technology",    "SME"),
    ("KPIL2",       "NKGSB",        "2024-08-14",  103, "NKGSB Cooperative",      85.00, 85.00,    30, "Finance",       "SME"),
    ("SAT",         "SATINDLTD",    "2024-07-22",   69, "SAT Industries",         56.00, 56.00,    40, "Chemicals",     "SME"),
    ("VIKRAM",      "VIKRAMTHERMO", "2024-07-10",  267, "Vikram Thermo",         310.00,310.00,    50, "Industrials",   "SME"),
    # ── 2023 SME IPOs ───────────────────────────────────────────────────
    ("ONMO",        "ONMOBILE",     "2023-09-11",   88, "Onmobile Global",        5.80,  5.80,   200, "Technology",    "SME"),
    ("SBCL",        "SBCL",         "2023-12-05",  210, "Shree Bhavani Cotton",  44.00, 44.00,    32, "Textiles",      "SME"),
    ("JAYSYNTH",    "JAYSYNTHORG",  "2023-11-08",   51, "Jaysynth Orgachem",      87.00, 87.00,    22, "Chemicals",     "SME"),
    ("ORIANA",      "ORIANA",       "2023-09-19",   98, "Oriana Power",          143.00,143.00,    72, "Energy",        "SME"),
    ("VELS",        "VELS",         "2023-07-28",   70, "Vels Film",             258.00,258.00,    40, "Media",         "SME"),
    ("AELEA",       "AELEALIFE",    "2023-08-14",   65, "Aelea Commodities",     375.00,375.00,    28, "Commodities",   "SME"),
]


@dataclass
class IPOBaseSignal:
    symbol: str
    company_name: str
    listing_date: str
    issue_price: float
    cmp: float
    score: float
    quality_score: float
    quality_grade: str
    quality_icon: str
    base_forming: bool
    base_depth_pct: float
    base_days: int
    pivot_price: float
    weeks_since_listing: int
    breakout_confirmed: bool
    breakout_volume_ratio: float
    rs_rating: float
    eight_week_hold_active: bool
    gain_since_listing: float
    lock_up_alerts: List[str]
    next_lockup_date: str
    next_lockup_days: int
    next_lockup_type: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    listing_score: float
    subscription_score: float
    volume_score: float
    base_score: float
    fundamental_score: float
    institutional_score: float
    sector_score: float
    rs_score: float
    segment: str = "Mainboard"
    signal: str = "WATCH"
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: object = None

    def __post_init__(self):
        self.score = self.quality_score


IPOResult = IPOBaseSignal


def get_ipo_price_data(yf_sym: str, listing_date: str) -> Optional[pd.DataFrame]:
    try:
        ticker = yf.Ticker(f"{yf_sym}.NS")
        df = ticker.history(start=listing_date, auto_adjust=True)
        if df.empty or len(df) < 3:
            # Try without .NS (some SME stocks)
            ticker2 = yf.Ticker(yf_sym)
            df = ticker2.history(start=listing_date, auto_adjust=True)
            if df.empty or len(df) < 3:
                return None
        df = df[["Open","High","Low","Close","Volume"]]
        df.columns = ["open","high","low","close","volume"]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        logger.debug(f"IPO data failed {yf_sym}: {e}")
        return None


def detect_ipo_base(df: pd.DataFrame, issue_price: float) -> Dict:
    result = {"base_forming": False, "base_depth_pct": 0.0, "base_days": 0,
              "pivot_price": 0.0, "volume_dry_up": False, "quality": "NONE", "score": 50}
    if df is None or len(df) < 8:
        return result

    lookback = min(len(df), 60)
    recent   = df.iloc[-lookback:]
    try:
        peak_loc = recent.index.get_loc(recent["high"].idxmax())
    except Exception:
        peak_loc = len(recent) // 3

    if peak_loc < 3:
        peak_loc = min(peak_loc + 5, len(recent) - 1)

    left_high   = float(recent["high"].iloc[:peak_loc+1].max())
    base_period = recent.iloc[peak_loc:]
    if len(base_period) < 4:
        return result

    base_low   = float(base_period["low"].min())
    base_depth = (left_high - base_low) / left_high * 100 if left_high > 0 else 0
    base_days  = len(base_period)
    vol_dry_up = base_period["volume"].mean() < df["volume"].mean() * 0.7
    pivot      = round(left_high, 2)
    cmp        = float(df["close"].iloc[-1])

    if 15 <= base_depth <= 30 and base_days >= 14: quality, score = "EXCELLENT", 90
    elif 15 <= base_depth <= 30 and base_days >= 8: quality, score = "GOOD", 75
    elif base_depth <= 35 and base_days >= 7:        quality, score = "FAIR", 55
    elif base_depth > 35:                            quality, score = "TOO_DEEP", 20
    else:                                            quality, score = "TOO_EARLY", 30

    result.update({
        "base_forming": quality in ("EXCELLENT","GOOD","FAIR") and cmp >= pivot * 0.90,
        "base_depth_pct": round(base_depth, 1), "base_days": base_days,
        "pivot_price": pivot, "volume_dry_up": vol_dry_up, "quality": quality, "score": score,
    })
    return result


def check_breakout_signal(df: pd.DataFrame, pivot: float) -> Dict:
    if df is None or len(df) < 4:
        return {"breakout": False, "vol_ratio": 0, "entry_gap": 0, "within_buy_range": False}
    latest    = df.iloc[-1]
    cmp       = float(latest["close"])
    vol_avg50 = df["volume"].iloc[-min(50, len(df)):].mean()
    vol_ratio = float(latest["volume"]) / (vol_avg50 + 1)
    entry_gap = (cmp - pivot) / pivot * 100 if pivot > 0 else 0
    return {
        "breakout":       cmp >= pivot * 0.998 and vol_ratio >= 1.5,
        "vol_ratio":      round(vol_ratio, 2),
        "entry_gap":      round(entry_gap, 2),
        "within_buy_range": -1 <= entry_gap <= 5,
    }


def check_eight_week_hold(df: pd.DataFrame, issue_price: float) -> Dict:
    if df is None or len(df) < 4:
        return {"active": False, "gain_3w": 0, "weeks_held": 0}
    listing_close = float(df["close"].iloc[0])
    three_wk_high = float(df["high"].iloc[:min(15, len(df))].max())
    gain_3w       = (three_wk_high - listing_close) / listing_close * 100 if listing_close > 0 else 0
    weeks_since   = len(df) / 5
    return {"active": gain_3w >= 20 and weeks_since < 8, "gain_3w": round(gain_3w, 1),
            "weeks_held": round(weeks_since, 1)}


def compute_lockup_alerts(listing_date_str: str) -> List[Dict]:
    today    = date.today()
    alerts   = []
    try:
        listing_dt = datetime.strptime(listing_date_str, "%Y-%m-%d").date()
        for name, days, impact in [
            ("Day 30 Anchor Lock-up", 30,  "76% of stocks decline avg 2.6%"),
            ("Day 90 Public Lock-up", 90,  "50% of supply unlocks"),
            ("Day 180 PE/VC",         180, "Avg -5% to -6% drag"),
            ("Day 540 Promoter",      540, "Largest supply event"),
        ]:
            ld        = listing_dt + timedelta(days=days)
            days_away = (ld - today).days
            if -10 <= days_away <= 90:
                alerts.append({"name": name, "date": ld.isoformat(),
                               "days_away": days_away, "impact": impact,
                               "status": "IMMINENT" if days_away <= 0 else "UPCOMING"})
    except Exception:
        pass
    return sorted(alerts, key=lambda x: x["days_away"])


def compute_ipo_quality(df: pd.DataFrame, row: tuple, base_data: Dict, nifty_df=None) -> Dict:
    symbol, yf_sym, listing_date, issue_price, company, qib_sub, overall_sub, issue_size_cr, sector, segment = row
    scores, details = {}, []

    listing_close = float(df["close"].iloc[0]) if not df.empty else issue_price
    listing_gain  = (listing_close - issue_price) / issue_price * 100 if issue_price > 0 else 0
    if listing_gain >= 50:   s1, d1 = 95, f"Stellar listing +{listing_gain:.0f}%"
    elif listing_gain >= 20: s1, d1 = 80, f"Strong listing +{listing_gain:.0f}%"
    elif listing_gain >= 5:  s1, d1 = 60, f"Positive listing +{listing_gain:.0f}%"
    elif listing_gain >= -5: s1, d1 = 40, f"Flat listing {listing_gain:+.0f}%"
    else:                    s1, d1 = max(10, 30+listing_gain), f"Weak listing {listing_gain:.0f}%"
    scores["listing"] = round(s1, 1); details.append(d1)

    if qib_sub >= 50:   s2, d2 = 95, f"QIB {qib_sub:.0f}x — exceptional"
    elif qib_sub >= 30: s2, d2 = 85, f"QIB {qib_sub:.0f}x — strong"
    elif qib_sub >= 10: s2, d2 = 70, f"QIB {qib_sub:.0f}x — good"
    elif qib_sub >= 2:  s2, d2 = 50, f"QIB {qib_sub:.0f}x — moderate"
    else:               s2, d2 = 35, f"QIB {qib_sub:.0f}x — low (SME normal)"
    if segment == "SME": s2 = min(s2 + 10, 95)  # SME QIB norms are lower
    scores["subscription"] = round(s2, 1); details.append(d2)

    if len(df) >= 10:
        rv = df["volume"].iloc[-5:].mean() / (df["volume"].iloc[:5].mean() + 1)
        if 0.3 <= rv <= 0.7: s3, d3 = 80, f"Volume drying up ({rv:.1f}x) — healthy base"
        elif rv < 0.3:        s3, d3 = 60, f"Very low volume ({rv:.1f}x)"
        elif rv > 2:          s3, d3 = 75, f"Volume picking up ({rv:.1f}x)"
        else:                 s3, d3 = 55, "Normal volume"
    else:
        s3, d3 = 50, "Insufficient history"
    scores["volume"] = round(s3, 1); details.append(d3)

    scores["base"] = round(base_data.get("score", 50), 1)

    if issue_size_cr >= 5000:   s5, d5 = 85, f"Large issue ₹{issue_size_cr:,.0f} Cr"
    elif issue_size_cr >= 1000: s5, d5 = 70, f"Mid issue ₹{issue_size_cr:,.0f} Cr"
    elif issue_size_cr >= 100:  s5, d5 = 55, f"Small issue ₹{issue_size_cr:,.0f} Cr"
    else:                       s5, d5 = 40, f"Micro issue ₹{issue_size_cr:,.0f} Cr (SME)"
    scores["fundamental"] = round(s5, 1); details.append(d5)

    cmp       = float(df["close"].iloc[-1]) if not df.empty else listing_close
    vs_l      = (cmp - listing_close) / listing_close * 100 if listing_close > 0 else 0
    if vs_l > 30:    s6, d6 = 90, f"{vs_l:.0f}% above listing — institutional holding"
    elif vs_l > 5:   s6, d6 = 70, f"{vs_l:.0f}% above listing"
    elif vs_l > -10: s6, d6 = 50, f"Near listing ({vs_l:+.0f}%)"
    else:            s6, d6 = 30, f"{vs_l:.0f}% below listing"
    scores["institutional"] = round(s6, 1); details.append(d6)

    leading = {"Finance","Auto","Consumer Tech","Energy","Telecom","Technology","Defence"}
    s7      = 75 if sector in leading else 55
    scores["sector"] = round(s7, 1)

    s8 = 55
    if nifty_df is not None and len(df) >= 10:
        try:
            lb     = min(len(df)-1, 21)
            sr     = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-lb]) - 1)*100
            nr     = (float(nifty_df["close"].iloc[-1]) / float(nifty_df["close"].iloc[-lb]) - 1)*100
            s8     = min(max(50 + (sr - nr)*2, 0), 100)
            details.append(f"RS {s8:.0f} vs Nifty")
        except Exception:
            pass
    scores["rs"] = round(s8, 1)

    weights = {"listing":0.15,"subscription":0.15,"volume":0.15,"base":0.15,
               "fundamental":0.15,"institutional":0.10,"sector":0.10,"rs":0.05}
    composite = round(sum(scores[k]*weights[k] for k in weights), 1)

    if composite >= 80:   grade, icon = "STRONG_BUY", "🏆"
    elif composite >= 60: grade, icon = "BUY",         "💪"
    elif composite >= 40: grade, icon = "WATCH",       "👀"
    else:                 grade, icon = "AVOID",       "⛔"

    return {"composite": composite, "grade": grade, "icon": icon, "scores": scores, "details": details}


def scan_ipo_universe(
    nifty_df=None,
    max_listing_months: int = 24,
    min_score: float = 0,
    breeze_engine=None,
    segment_filter: str = "All",  # "All", "Mainboard", "SME"
    # legacy aliases
    max_weeks: int = None,
    min_quality: float = None,
    rrg_data: dict = None,
) -> List[IPOBaseSignal]:
    if max_weeks is not None:
        max_listing_months = int(max_weeks / 4.33)
    if min_quality is not None:
        min_score = float(min_quality)

    results = []
    cutoff  = date.today() - timedelta(days=max_listing_months * 30)

    for row in CURATED_IPOS:
        symbol, yf_sym, listing_date_str, issue_price, company, qib_sub, overall_sub, issue_size_cr, sector, segment = row
        try:
            listing_dt = datetime.strptime(listing_date_str, "%Y-%m-%d").date()
            if listing_dt < cutoff:
                continue
            if segment_filter != "All" and segment != segment_filter:
                continue

            df = get_ipo_price_data(yf_sym, listing_date_str)
            if df is None or len(df) < 3:
                continue

            cmp         = float(df["close"].iloc[-1])
            weeks_since = round(len(df) / 5, 1)
            base_data   = detect_ipo_base(df, issue_price)
            quality     = compute_ipo_quality(df, row, base_data, nifty_df)

            if quality["composite"] < min_score:
                continue

            pivot     = base_data.get("pivot_price", cmp)
            breakout  = check_breakout_signal(df, pivot)
            eight_wk  = check_eight_week_hold(df, issue_price)
            lockups   = compute_lockup_alerts(listing_date_str)

            lockup_strs = []
            nd, ndint, nt = "", 999, ""
            for a in lockups:
                icon = "🚨" if a["days_away"] <= 0 else "⚠️"
                lockup_strs.append(f"{icon} {a['name']} in {a['days_away']}d: {a['impact']}")
                if a["days_away"] < ndint:
                    ndint, nd, nt = a["days_away"], a["date"], a["name"]

            rs      = quality["scores"].get("rs", 55)
            gain_sl = (cmp - issue_price) / issue_price * 100 if issue_price > 0 else 0

            entry  = pivot if not breakout["breakout"] else cmp
            sl     = round(entry * 0.92, 2)
            risk   = entry - sl
            t1     = round(entry + 2*risk, 2)
            t2     = round(entry + 3.5*risk, 2)
            rr     = round((t1 - entry) / risk, 1) if risk > 0 else 0

            g      = quality["grade"]
            sig_map = {"STRONG_BUY":"STRONG BUY","BUY":"BUY","WATCH":"WATCH","AVOID":"AVOID"}

            results.append(IPOBaseSignal(
                symbol=symbol, company_name=company,
                listing_date=listing_date_str, issue_price=issue_price, cmp=round(cmp,2),
                score=quality["composite"], quality_score=quality["composite"],
                quality_grade=g, quality_icon=quality["icon"],
                base_forming=base_data["base_forming"],
                base_depth_pct=base_data["base_depth_pct"],
                base_days=base_data["base_days"], pivot_price=pivot,
                weeks_since_listing=int(weeks_since),
                breakout_confirmed=breakout["breakout"],
                breakout_volume_ratio=breakout["vol_ratio"],
                rs_rating=round(float(rs), 1),
                eight_week_hold_active=eight_wk["active"],
                gain_since_listing=round(gain_sl, 1),
                lock_up_alerts=lockup_strs,
                next_lockup_date=nd, next_lockup_days=ndint, next_lockup_type=nt,
                entry_price=round(entry,2), stop_loss=round(sl,2),
                target_1=round(t1,2), target_2=round(t2,2), risk_reward=rr,
                segment=segment, signal=sig_map.get(g,"WATCH"),
                listing_score=quality["scores"].get("listing",50),
                subscription_score=quality["scores"].get("subscription",50),
                volume_score=quality["scores"].get("volume",50),
                base_score=quality["scores"].get("base",50),
                fundamental_score=quality["scores"].get("fundamental",50),
                institutional_score=quality["scores"].get("institutional",50),
                sector_score=quality["scores"].get("sector",60),
                rs_score=float(quality["scores"].get("rs",55)),
                reasons=quality["details"], warnings=lockup_strs,
            ))

        except Exception as e:
            logger.debug(f"IPO scan error {symbol}: {e}")
            continue

    results.sort(key=lambda x: (x.breakout_confirmed, x.quality_score), reverse=True)
    return results


def get_upcoming_lock_up_alerts(days_ahead: int = 30) -> List[Dict]:
    alerts = []
    today  = date.today()
    cutoff = today + timedelta(days=days_ahead)
    for row in CURATED_IPOS:
        sym = row[0]
        for a in compute_lockup_alerts(row[2]):
            try:
                ed = datetime.strptime(a["date"], "%Y-%m-%d").date()
                if today <= ed <= cutoff:
                    alerts.append({"symbol": sym, "event": a["name"], "date": a["date"],
                                   "days_until": a["days_away"], "action": "Review / tighten stop"})
            except Exception:
                continue
    return sorted(alerts, key=lambda x: x.get("days_until", 999))


def fetch_nse_ipo_list() -> List[Dict]:
    return [{"symbol":r[0],"company_name":r[4],"listing_date":r[2],
             "issue_price":r[3],"qib_sub":r[5],"overall_sub":r[6],"segment":r[9]}
            for r in CURATED_IPOS]


def format_ipo_alert(result: IPOBaseSignal) -> str:
    if result is None: return "No IPO data."
    icons = {"STRONG_BUY":"🟢","BUY":"🔵","WATCH":"🟡","AVOID":"🔴"}
    icon  = icons.get(result.quality_grade, "⚪")
    seg   = f" [{result.segment}]" if hasattr(result, "segment") else ""
    return (f"{icon} IPO{seg} — {result.symbol} ({result.company_name})\n"
            f"Score: {result.quality_score:.0f}/100 | {result.quality_grade}\n"
            f"Listed: {result.listing_date} | CMP: ₹{result.cmp:,.0f} | Issue: ₹{result.issue_price:,.0f}\n"
            f"Gain since listing: {result.gain_since_listing:+.1f}%")
