"""
AI DEEP DIVE v3.0 — NSE Scanner Pro Integration
================================================
CHANGES FROM v2:
- Exhaustive AI prompt: RS, SQI breakdown, strategy PF, regime fit, sector RRG,
  weekly alignment, scanner reasons, Bollinger, ADX, stochastic, volume analysis
- 1000+ stock universe via NSE live CSV + extended fallback list
- Full standalone analysis for ANY stock (not just scanner signals)
- On-demand SQI computation for manual ticker input
- Cached regime + breadth data (5 min TTL)
- Strategy context block in prompt (win rate, PF, ideal regimes)
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# STOCK UNIVERSE — NSE Top 1000+
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def _get_nse_universe() -> List[str]:
    """
    Fetch NSE 500 from official NSE CSV. Augment with additional large/mid caps
    to reach ~1000 stocks. Falls back to hardcoded list.
    """
    import requests as req, csv, io

    # Primary: NSE official Nifty 500 CSV
    base_symbols = []
    try:
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
        r = req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code == 200:
            reader = csv.DictReader(io.StringIO(r.text))
            for row in reader:
                sym = row.get("Symbol", "").strip()
                if sym: base_symbols.append(sym)
    except Exception:
        pass

    # Secondary: NSE Nifty 1000 smallcap/midcap additions
    extra_symbols = []
    for index_url in [
        "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
    ]:
        try:
            r = req.get(index_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code == 200:
                reader = csv.DictReader(io.StringIO(r.text))
                for row in reader:
                    sym = row.get("Symbol", "").strip()
                    if sym and sym not in base_symbols: extra_symbols.append(sym)
        except Exception:
            pass

    combined = base_symbols + extra_symbols
    if len(combined) > 200:
        return combined  # Up to ~900 unique NSE stocks

    # Hardcoded fallback — Nifty 500 core + additional liquid mid/smallcaps
    return _FALLBACK_1000


# Hardcoded fallback with 500 known liquid NSE symbols
_FALLBACK_1000 = [
    # Nifty 50
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "ITC","KOTAKBANK","LT","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI","HCLTECH",
    "SUNPHARMA","TITAN","ULTRACEMCO","NTPC","WIPRO","NESTLEIND","BAJAJFINSV","POWERGRID",
    "M&M","ONGC","JSWSTEEL","ADANIPORTS","TATASTEEL","COALINDIA","HDFCLIFE","SBILIFE",
    "TECHM","GRASIM","DRREDDY","CIPLA","BPCL","APOLLOHOSP","DIVISLAB","EICHERMOT",
    "HEROMOTOCO","INDUSINDBK","TATACONSUM","BRITANNIA","BAJAJ-AUTO","HINDALCO","ADANIENT",
    "SHRIRAMFIN","LTIM","TRENT",
    # Nifty Next 50
    "ABB","ACC","ADANIGREEN","ADANIPOWER","AMBUJACEM","ATGL","AUROPHARMA","BANKBARODA",
    "BEL","BERGEPAINT","BOSCHLTD","CANBK","CHOLAFIN","COLPAL","CONCOR","DABUR","DLF",
    "DMART","GAIL","GODREJCP","HAL","HAVELLS","ICICIPRULI","INDIGO","IOC","IRCTC",
    "IRFC","JIOFIN","JINDALSTEL","LICI","LODHA","LUPIN","MARICO","MOTHERSON","NAUKRI",
    "NHPC","OBEROIRLTY","PERSISTENT","PIDILITIND","PIIND","PNB","POLYCAB","SBICARD",
    "SIEMENS","TORNTPHARM","TATAPOWER","VEDL","ZOMATO","ZYDUSLIFE","RECLTD",
    # Nifty Midcap 150
    "ABCAPITAL","ABFRL","AEGISLOG","AFFLE","AJANTPHARM","ALKEM","ALOKINDS","AMBER",
    "ANANDRATHI","ANANTRAJ","ANGELONE","APLAPOLLO","APLLTD","APARINDS","APTUS",
    "ASTRAL","ASTER","ASTRAZEN","ATUL","AUBANK","BAJAJHFL","BALKRISIND","BANDHANBNK",
    "BATAINDIA","BAYERCROP","BIKAJI","BIOCON","BLUESTARCO","BSOFT","BRIGADE","BSE",
    "CANFINHOME","CAPLIPOINT","CARBORUNIV","CEATLTD","CDSL","CENTURYPLY","CERA",
    "CHAMBLFERT","CHOLAHLDNG","CLEAN","COFORGE","COROMANDEL","CREDITACC","CROMPTON",
    "CUMMINSIND","CYIENT","DATAPATTNS","DEEPAKFERT","DEEPAKNTR","DELHIVERY","DEVYANI",
    "DIXON","EMAMILTD","ENDURANCE","ESCORTS","FEDERALBNK","FIVESTAR","FORCEMOT",
    "FORTIS","GICRE","GILLETTE","GLAND","GLENMARK","GLAXO","GPIL","GODREJAGRO",
    "GODREJIND","GODREJPROP","GRANULES","GRAPHITE","GRAVITA","GESHIP","FLUOROCHEM",
    "GUJGASLTD","GSPL","HEG","HAPPSTMNDS","HFCL","HINDCOPPER","HINDZINC","HOMEFIRST",
    "HUDCO","HYUNDAI","ICICIGI","IDBI","IDFCFIRSTB","IFCI","IIFL","IRB","IRCON",
    "ITC","IGL","INDUSTOWER","INTELLECT","IPCALAB","JBCHEPHARM","JKCEMENT","JBMA",
    "JMFINANCIL","JSWENERGY","JSWINFRA","JPPOWER","JINDALSAW","JSL","JUBLFOOD",
    "JUBLINGREA","JUBLPHARMA","JWL","JYOTHYLAB","KAJARIACER","KPIL","KALYANKJIL",
    "KARURVYSYA","KAYNES","KEC","KFINTECH","KOTAKBANK","KEI","KPITTECH","LATENTVIEW",
    "LAURUSLABS","LICHSGFIN","LINDEINDIA","LLOYDSME","LTF","LTTS","LTFOODS",
    "MANAPPURAM","MRPL","MANKIND","MFSL","MAXHEALTH","MAZDOCK","METROPOLIS",
    "MINDACORP","MOTILALOFS","MPHASIS","MCX","MUTHOOTFIN","NATCOPHARM","NBCC","NCC",
    "NLCINDIA","NMDC","NAM-INDIA","NUVOCO","ONGC","OFSS","PVRINOX","PAGEIND",
    "PATANJALI","PETRONET","PFIZER","PHOENIXLTD","PPLPHARMA","POLYMED","POONAWALLA",
    "PFC","POWERGRID","PRAJIND","PRESTIGE","PGHH","RBLBANK","RADICO","RVNL","RAILTEL",
    "RAINBOW","RKFORGE","RCF","REDINGTON","RHIM","RITES","RRKABEL","SRF","SAGILITY",
    "SAMMAANCAP","SAPPHIRE","SARDAEN","SAREGAMA","SCHAEFFLER","SCHNEIDER","SHREECEM",
    "SHYAMMETL","SIEMENS","SIGNATURE","SOBHA","SOLARINDS","SONACOMS","SONATSOFTW",
    "STARHEALTH","SUMICHEM","SUNTV","SUNDARMFIN","SUNDRMFAST","SUPREMEIND","SUZLON",
    "SWIGGY","SYNGENE","TBOTEK","TVSMOTOR","TATACHEM","TATACOMM","TATAELXSI",
    "TATAINVEST","TATATECH","TECHM","THERMAX","TIMKEN","TITAGARH","TORNTPOWER",
    "TARIL","TRIDENT","TRIVENI","TRITURBINE","TIINDIA","UNOMINDA","UPL","UTIAMC",
    "UNIONBANK","UBL","UNITDSPR","VGUARD","VTL","VBL","VEDL","VIJAYA","IDEA","VOLTAS",
    "WAAREEENER","WELCORP","WHIRLPOOL","WIPRO","WOCKPHARMA","YESBANK","ZEEL","ZENTEC",
    "ZENSARTECH","ECLERX","CGPOWER","BDL","CDSL","CAMS","CONCORDBIO","CRAFTSMAN",
    "DCMSHRIRAM","EMCURE","ERIS","ELECON","ELGIEQUIP","ENGINERSIN","FACT","FINCABLES",
    "FINPIPE","FSL","GMDCLTD","GMRAIRPORT","GRSE","GODIGIT","GODREJCP","GODFRYPHLP",
    "HINDPETRO","HONASA","HONAUT","IEX","INDIAMART","INDIANB","IOB","IREDA",
    "ITCHOTELS","ITI","INDGN","JKTYRE","JOUBLFOOD","JSWCEMENT","KARURVYSYA",
    "KIRLOSBROS","KIRLOSENG","KIMS","LALPATHLAB","MAHABANK","MAHSEAMLES","M&MFIN",
    "MAHDOCK","MANAPPURAM","MANYAVAR","MAXHEALTH","MINDACORP","MSUMI","MGL",
    "MAHSCOOTER","MRPL","NATIONALUM","NAVA","NAVINFLUOR","NETWEB","NEULANDLAB",
    "NEWGEN","NIVABUPA","NUVAMA","NBCC","NTPCGREEN","NH","OLAELEC","OLECTRA","PAYTM",
    "PCBL","PGEL","POLICYBZR","PNBHOUSING","PTCIL","PREMIERENE","RECLTD","RELINFRA",
    "RPOWER","SBFC","SAIL","SJVN","SAILIFE","SAREGAMA","TATAMOTORS","BHEL","BEML",
    "BHARAT","HINDCOPPER","HINDALCO","HAL","GRSE","BDL","MAZDOCK","COCHINSHIP",
    "ABBOTINDIA","AKUMS","AKZOINDIA","ASTERDM","AGARWALEYE","BLUEJET","CAPLIPOINT",
    "COHANCE","CONCORDBIO","DRREDDY","DIVISLAB","EMCURE","ERIS","FORTIS","GLAND",
    "GLAXO","GLENMARK","GRANULES","IPCALAB","JBCHEPHARM","JUBLPHARMA","LAURUSLABS",
    "LUPIN","MANKIND","MAXHEALTH","MEDANTA","METROPOLIS","NATCOPHARM","NEULANDLAB",
    "ONESOURCE","PPLPHARMA","PFIZER","POLYMED","RAINBOW","SAILIFE","SUNPHARMA",
    "SYNGENE","TORNTPHARM","VIJAYA","WOCKPHARMA","ZYDUSLIFE","CIPLA","APOLLOHOSP",
]


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0).rolling(n).mean()
    l = (-d.where(d < 0, 0)).rolling(n).mean()
    return (100 - 100 / (1 + g / l.replace(0, 1e-10))).fillna(50).clip(0, 100)

def _macd(s: pd.Series) -> Tuple:
    e1 = s.ewm(span=12, adjust=False).mean()
    e2 = s.ewm(span=26, adjust=False).mean()
    m = e1 - e2; sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig

def _adx(hi, lo, cl, n=14) -> float:
    try:
        pdm = hi.diff().clip(lower=0); ndm = (-lo.diff()).clip(lower=0)
        tr = pd.concat([hi-lo,(hi-cl.shift()).abs(),(lo-cl.shift()).abs()],axis=1).max(1)
        atr = tr.rolling(n).mean().replace(0,1e-10)
        pdi = 100*pdm.rolling(n).mean()/atr; ndi = 100*ndm.rolling(n).mean()/atr
        dx = (pdi-ndi).abs()/(pdi+ndi).replace(0,1e-10)*100
        return float(dx.rolling(n).mean().iloc[-1])
    except: return 20.0

def _stoch(hi, lo, cl, k=14, d=3) -> Tuple[float, float]:
    try:
        ll = lo.rolling(k).min(); hh = hi.rolling(k).max()
        k_line = 100*(cl-ll)/(hh-ll).replace(0,1e-10)
        d_line = k_line.rolling(d).mean()
        return float(k_line.iloc[-1]), float(d_line.iloc[-1])
    except: return 50.0, 50.0

def _bollinger(s: pd.Series, n=20, std=2) -> Tuple:
    sma = s.rolling(n).mean(); sigma = s.rolling(n).std()
    upper = sma + std*sigma; lower = sma - std*sigma
    bb_pct = (s - lower) / (upper - lower).replace(0, 1e-10)
    bb_width = (upper - lower) / sma.replace(0, 1e-10) * 100
    return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1]), \
           float(bb_pct.iloc[-1]), float(bb_width.iloc[-1])

def _compute_metrics(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or len(df) < 30:
        return None
    try:
        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        m: Dict[str, Any] = {}

        # Price
        m["current_price"]          = float(c.iloc[-1])
        m["price_change_1d"]        = (c.iloc[-1]/c.iloc[-2]-1)*100 if len(c)>=2 else 0
        m["price_change_5d"]        = (c.iloc[-1]/c.iloc[-6]-1)*100 if len(c)>=6 else 0
        m["price_change_1m"]        = (c.iloc[-1]/c.iloc[-22]-1)*100 if len(c)>=22 else 0
        m["price_change_3m"]        = (c.iloc[-1]/c.iloc[-66]-1)*100 if len(c)>=66 else 0

        # 52W
        m["high_52w"]               = float(h.rolling(252,min_periods=50).max().iloc[-1])
        m["low_52w"]                = float(l.rolling(252,min_periods=50).min().iloc[-1])
        m["pct_from_52w_high"]      = (m["current_price"]/m["high_52w"]-1)*100
        m["pct_from_52w_low"]       = (m["current_price"]/m["low_52w"]-1)*100

        # Volume
        avg_vol                     = v.iloc[-20:].mean()
        avg_vol_50                  = v.iloc[-50:].mean()
        m["volume_ratio"]           = float(v.iloc[-1]/avg_vol) if avg_vol>0 else 1.0
        m["volume_ratio_50d"]       = float(v.iloc[-1]/avg_vol_50) if avg_vol_50>0 else 1.0
        m["volume_trend"]           = "EXPANDING" if avg_vol > avg_vol_50*1.1 else ("CONTRACTING" if avg_vol < avg_vol_50*0.9 else "STABLE")

        # RSI
        rsi_s = _rsi(c); rsi9 = _rsi(c, 9)
        m["rsi_14"]                 = float(rsi_s.iloc[-1])
        m["rsi_9"]                  = float(rsi9.iloc[-1])
        m["rsi_divergence"]         = "BULLISH" if m["rsi_14"] > m["rsi_9"] else "BEARISH"
        m["rsi_status"]             = ("OVERSOLD" if m["rsi_14"]<30 else "OVERBOUGHT" if m["rsi_14"]>70 else "NEUTRAL")

        # MACD
        ml, ms, mh                  = _macd(c)
        m["macd_line"]              = float(ml.iloc[-1])
        m["macd_signal_line"]       = float(ms.iloc[-1])
        m["macd_histogram"]         = float(mh.iloc[-1])
        m["macd_prev_histogram"]    = float(mh.iloc[-2]) if len(mh)>=2 else 0
        if mh.iloc[-1]>0 and mh.iloc[-2]<=0:     m["macd_status"] = "BULLISH_CROSSOVER"
        elif mh.iloc[-1]<0 and mh.iloc[-2]>=0:   m["macd_status"] = "BEARISH_CROSSOVER"
        elif mh.iloc[-1]>0 and mh.iloc[-1]>mh.iloc[-2]: m["macd_status"] = "BULLISH_EXPANDING"
        elif mh.iloc[-1]>0:                       m["macd_status"] = "BULLISH_FADING"
        elif mh.iloc[-1]<0 and mh.iloc[-1]<mh.iloc[-2]: m["macd_status"] = "BEARISH_EXPANDING"
        else:                                     m["macd_status"] = "BEARISH_FADING"

        # MAs
        sma20  = c.rolling(20).mean();   sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean();  ema9  = c.ewm(span=9,adjust=False).mean()
        ema21  = c.ewm(span=21,adjust=False).mean()
        m["sma_20"]                 = float(sma20.iloc[-1])
        m["sma_50"]                 = float(sma50.iloc[-1])
        m["sma_200"]                = float(sma200.iloc[-1]) if len(c)>=200 else None
        m["ema_21"]                 = float(ema21.iloc[-1])
        m["price_vs_sma20_pct"]     = (m["current_price"]/sma20.iloc[-1]-1)*100
        m["price_vs_sma50_pct"]     = (m["current_price"]/sma50.iloc[-1]-1)*100
        m["price_vs_sma200_pct"]    = (m["current_price"]/sma200.iloc[-1]-1)*100 if m["sma_200"] else None
        m["sma20_vs_sma50"]         = "ABOVE" if sma20.iloc[-1]>sma50.iloc[-1] else "BELOW"
        m["golden_cross"]           = (sma50.iloc[-1]>sma200.iloc[-1]) if m["sma_200"] else None

        if   c.iloc[-1]>sma20.iloc[-1]>ema21.iloc[-1]: m["trend_short_term"] = "BULLISH"
        elif c.iloc[-1]<sma20.iloc[-1]<ema21.iloc[-1]: m["trend_short_term"] = "BEARISH"
        else:                                            m["trend_short_term"] = "NEUTRAL"
        m["trend_medium_term"]      = "BULLISH" if ema21.iloc[-1]>sma50.iloc[-1] else "BEARISH"
        if m["sma_200"]:
            m["trend_long_term"]    = "BULLISH" if sma50.iloc[-1]>sma200.iloc[-1] else "BEARISH"
        else:
            m["trend_long_term"]    = "UNKNOWN"

        # ADX
        adx_val                     = _adx(h, l, c)
        m["adx_14"]                 = adx_val
        m["trend_strength"]         = "VERY STRONG" if adx_val>35 else ("STRONG" if adx_val>25 else ("MODERATE" if adx_val>20 else "WEAK"))

        # Stochastic
        stk, std_d                  = _stoch(h, l, c)
        m["stoch_k"]                = stk
        m["stoch_d"]                = std_d
        m["stoch_status"]           = ("OVERSOLD" if stk<20 else "OVERBOUGHT" if stk>80 else "NEUTRAL")

        # Bollinger Bands
        bb_up, bb_mid, bb_lo, bb_pct, bb_w = _bollinger(c)
        m["bb_upper"]               = bb_up
        m["bb_lower"]               = bb_lo
        m["bb_pct"]                 = bb_pct  # 0=at lower, 1=at upper
        m["bb_width"]               = bb_w
        m["bb_squeeze"]             = bb_w < 4.0  # volatility compression
        m["bb_position"]            = ("AT_UPPER" if bb_pct>0.9 else "AT_LOWER" if bb_pct<0.1 else
                                       "UPPER_HALF" if bb_pct>0.5 else "LOWER_HALF")

        # ATR + Volatility
        tr2 = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
        atr14 = tr2.rolling(14).mean()
        m["atr_pct"]                = float(atr14.iloc[-1])/m["current_price"]*100
        m["realized_volatility_20d"] = float(c.pct_change().iloc[-20:].std()*(252**0.5)*100)
        m["realized_volatility_5d"]  = float(c.pct_change().iloc[-5:].std()*(252**0.5)*100)

        # RS Rating (1M+3M momentum rank proxy — actual RS vs Nifty needs Nifty data)
        # We compute relative strength score using price performance buckets
        perf_3m = m["price_change_3m"]
        perf_1m = m["price_change_1m"]
        rs_proxy = min(100, max(0, 50 + (perf_3m*0.6 + perf_1m*0.4) * 1.2))
        m["rs_proxy"]               = round(rs_proxy, 1)

        # Trend alignment score (0-5)
        ta = sum([
            c.iloc[-1] > ema21.iloc[-1],
            c.iloc[-1] > sma50.iloc[-1],
            c.iloc[-1] > sma200.iloc[-1] if m["sma_200"] else False,
            sma20.iloc[-1] > sma50.iloc[-1],
            sma50.iloc[-1] > sma200.iloc[-1] if m["sma_200"] else False,
        ])
        m["trend_alignment"]        = ta
        m["trend_alignment_str"]    = f"{ta}/5 MAs aligned bullish"

        # Volume accumulation (last 5 up days vs down days volume)
        try:
            returns_5 = c.pct_change().iloc[-10:]
            vols_5 = v.iloc[-10:]
            up_vol = vols_5[returns_5 > 0].sum()
            dn_vol = vols_5[returns_5 <= 0].sum()
            m["accumulation_ratio"] = round(up_vol/dn_vol, 2) if dn_vol > 0 else 2.0
            m["accumulation"]       = "ACCUMULATING" if m["accumulation_ratio"]>1.2 else ("DISTRIBUTING" if m["accumulation_ratio"]<0.8 else "NEUTRAL")
        except:
            m["accumulation_ratio"] = 1.0
            m["accumulation"]       = "NEUTRAL"

        # Store for chart
        df["_rsi"]   = rsi_s
        df["_sma20"] = sma20
        df["_ema21"] = ema21
        df["_sma50"] = sma50

        return m
    except Exception as e:
        print(f"[AI Deep Dive] metrics error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# REGIME
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_regime() -> Dict:
    try:
        vix_h  = yf.Ticker("^INDIAVIX").history(period="5d")
        nft_h  = yf.Ticker("^NSEI").history(period="20d")
        vix    = float(vix_h["Close"].iloc[-1]) if not vix_h.empty else None
        nifty  = float(nft_h["Close"].iloc[-1]) if not nft_h.empty else None
        nchg   = (nft_h["Close"].iloc[-1]/nft_h["Close"].iloc[-2]-1)*100 if len(nft_h)>=2 else 0
        nchg_5d = (nft_h["Close"].iloc[-1]/nft_h["Close"].iloc[-6]-1)*100 if len(nft_h)>=6 else 0
        nchg_20d = (nft_h["Close"].iloc[-1]/nft_h["Close"].iloc[-21]-1)*100 if len(nft_h)>=21 else 0
        adv = dec = 0
        for t in ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
                  "SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS","LT.NS",
                  "AXISBANK.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS","HCLTECH.NS"]:
            try:
                h2 = yf.Ticker(t).history(period="2d")
                if len(h2)>=2:
                    if h2["Close"].iloc[-1]>h2["Close"].iloc[-2]: adv+=1
                    else: dec+=1
            except: pass
        breadth = adv/(adv+dec) if (adv+dec)>0 else 0.5
    except Exception:
        return dict(regime="UNKNOWN",code="UNKNOWN",confidence=0,
                    description="Data unavailable",strategy="Wait",
                    atlas_weight=33,oracle_weight=33,sentinel_weight=34,
                    vix="N/A",breadth=0.5,nifty_level="N/A",nifty_change=0,
                    nifty_change_5d=0,nifty_change_20d=0,advances=0,declines=0)

    if vix is None:
        r = dict(regime="UNKNOWN",code="UNKNOWN",confidence=0,description="VIX unavailable",
                 strategy="Wait",atlas_weight=33,oracle_weight=33,sentinel_weight=34)
    elif vix > 22:
        r = dict(regime="LIQUIDITY VACUUM",code="STRESS",
                 confidence=min(95,70+(vix-22)*2),
                 description="High fear — Risk-off. Sell rallies, protect capital.",
                 strategy="Cash 60-70%, hedge existing longs, no new breakouts",
                 atlas_weight=10,oracle_weight=20,sentinel_weight=70)
    elif vix < 13:
        r = dict(regime="COMPRESSION",code="ACCUMULATE",
                 confidence=min(90,65+(13-vix)*3),
                 description="Low volatility — Market coiling for directional move.",
                 strategy="Mean-reversion trades, prepare breakout watchlist",
                 atlas_weight=25,oracle_weight=35,sentinel_weight=40)
    elif 14<=vix<=19 and breadth>0.6:
        r = dict(regime="MOMENTUM CASCADE",code="TREND",confidence=75,
                 description="Strong directional trend — momentum is your friend.",
                 strategy="Follow momentum, trail stops, max position size",
                 atlas_weight=50,oracle_weight=20,sentinel_weight=30)
    elif 13<=vix<=17 and 0.4<=breadth<=0.6:
        r = dict(regime="LIQUIDITY DRIFT",code="GOLDILOCKS",confidence=70,
                 description="Optimal alpha-generation — stock-picker's market.",
                 strategy="Balanced exposure, active trading, sector rotation",
                 atlas_weight=35,oracle_weight=40,sentinel_weight=25)
    else:
        r = dict(regime="REGIME TRANSITION",code="UNCERTAINTY",confidence=55,
                 description="Mixed signals — avoid overcommitting.",
                 strategy="Half position sizes, wait for regime clarity",
                 atlas_weight=20,oracle_weight=30,sentinel_weight=50)

    r.update(vix=round(vix,2) if vix else "N/A",breadth=round(breadth,2),
             nifty_level=round(nifty,2) if nifty else "N/A",
             nifty_change=round(nchg,2),nifty_change_5d=round(nchg_5d,2),
             nifty_change_20d=round(nchg_20d,2),advances=adv,declines=dec)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT + DECISION
# ══════════════════════════════════════════════════════════════════════════════

def _sentiment(m: Dict) -> Dict:
    score, factors = 0, []
    vr, chg1 = m.get("volume_ratio",1.0), m.get("price_change_1d",0)

    # Volume analysis
    if vr>2.0:
        if chg1>0: score+=3; factors.append(f"🔥 INSTITUTIONAL BUY: {vr:.2f}× volume spike on up day (+3)")
        else:       score-=2; factors.append(f"⚠️ {vr:.2f}× volume on DOWN day — distribution signal (-2)")
    elif vr>1.5:
        if chg1>0: score+=2; factors.append(f"✅ Strong volume {vr:.2f}× + positive price action (+2)")
        else:       score-=1; factors.append(f"High vol {vr:.2f}× on down day — caution (-1)")
    elif vr<0.7: factors.append(f"📉 Low vol {vr:.2f}× — weak institutional conviction (neutral)")

    # Accumulation/Distribution
    accum = m.get("accumulation","NEUTRAL")
    ar = m.get("accumulation_ratio",1.0)
    if accum=="ACCUMULATING": score+=1; factors.append(f"💚 A/D ratio {ar:.2f} — 10-day accumulation pattern (+1)")
    elif accum=="DISTRIBUTING": score-=1; factors.append(f"🔴 A/D ratio {ar:.2f} — distribution over 10 days (-1)")

    # Relative strength (52W position)
    ph = m.get("pct_from_52w_high",-100)
    if ph>-5:    score+=2; factors.append(f"💪 Near 52W high ({ph:+.1f}%) — top RS quartile (+2)")
    elif ph>-10: score+=1; factors.append(f"📈 Within 10% of 52W high ({ph:+.1f}%) (+1)")
    elif ph<-30: score-=1; factors.append(f"📉 Deep correction ({ph:+.1f}%) — weak RS (-1)")
    pl = m.get("pct_from_52w_low",0)
    if pl<10:    score-=2; factors.append(f"⚠️ Near 52W low (+{pl:.1f}%) — extreme weakness (-2)")

    # BB position
    bp = m.get("bb_position","")
    if bp=="AT_UPPER":  score+=1; factors.append(f"🎯 Price at BB upper band — breakout momentum (+1)")
    elif bp=="AT_LOWER": score-=1; factors.append(f"🎯 Price at BB lower band — potential reversal zone (-1)")
    if m.get("bb_squeeze"): factors.append(f"🔒 Bollinger Band squeeze — breakout pending (watch closely)")

    # Momentum consistency
    c5 = m.get("price_change_5d",0)
    c1m = m.get("price_change_1m",0)
    if c5>5 and chg1>0 and c1m>8:  score+=1; factors.append(f"🚀 Multi-timeframe uptrend: 1M +{c1m:.1f}%, 5D +{c5:.1f}% (+1)")
    elif c5<-5 and chg1<0 and c1m<-8: score-=1; factors.append(f"📉 Multi-timeframe downtrend: 1M {c1m:.1f}%, 5D {c5:.1f}% (-1)")

    vol20 = m.get("realized_volatility_20d",0)
    if vol20>50: score-=1; factors.append(f"⚡ Very high volatility {vol20:.1f}% — erratic price action (-1)")

    overall = ("STRONG BULLISH" if score>=3 else "BULLISH" if score>=1 else
               "STRONG BEARISH" if score<=-3 else "BEARISH" if score<=-1 else "NEUTRAL")
    return dict(overall=overall,score=score,factors=factors)


def _decision(m: Dict, regime: Dict, sent: Dict) -> Dict:
    a = 0
    if m.get("trend_short_term")=="BULLISH": a+=2
    elif m.get("trend_short_term")=="BEARISH": a-=2
    if m.get("macd_status") in ("BULLISH_CROSSOVER","BULLISH_EXPANDING"): a+=1
    elif m.get("macd_status") in ("BEARISH_CROSSOVER","BEARISH_EXPANDING"): a-=1
    rsi = m.get("rsi_14",50)
    if rsi<35: a+=1
    elif rsi>65: a-=1
    if m.get("trend_alignment",0)>=4: a+=1
    elif m.get("trend_alignment",0)<=1: a-=1
    atlas = "BUY" if a>=2 else ("SELL" if a<=-2 else "HOLD")

    os = sent.get("score",0)
    oracle = "BUY" if os>=2 else ("SELL" if os<=-2 else "HOLD")
    oracle_conf = min(90,50+abs(os)*10)

    rs = 0
    vol20 = m.get("realized_volatility_20d",0)
    if vol20>40: rs+=2
    elif vol20>25: rs+=1
    elif vol20<18: rs-=1
    if m.get("atr_pct",0)>3: rs+=1
    ph = m.get("pct_from_52w_high",-50)
    if ph>-10: rs+=1
    elif ph<-35: rs-=1
    sentinel = "HIGH" if rs>=2 else ("LOW" if rs<=0 else "MEDIUM")

    aw = regime.get("atlas_weight",33)/100
    ow = regime.get("oracle_weight",33)/100
    sw = regime.get("sentinel_weight",34)/100
    sm = {"BUY":1,"HOLD":0,"SELL":-1}
    rp = {"LOW":0,"MEDIUM":0.2,"HIGH":0.5}
    ws = (sm[atlas]*aw + sm[oracle]*ow) * (1 - rp[sentinel]*sw)

    if ws>=0.4:    final,trend = "BUY","BULLISH"
    elif ws<=-0.4: final,trend = "SELL","BEARISH"
    else:          final,trend = "HOLD","NEUTRAL"

    return dict(atlas_signal=atlas,atlas_score=a,oracle_signal=oracle,
                oracle_confidence=oracle_conf,oracle_score=os,
                sentinel_risk=sentinel,sentinel_score=rs,
                weighted_score=round(ws,3),final_recommendation=final,trend=trend)


# ══════════════════════════════════════════════════════════════════════════════
# EXHAUSTIVE AI PROMPT
# ══════════════════════════════════════════════════════════════════════════════

# Strategy knowledge base for prompt context
_STRATEGY_KB = {
    "EMA21_Bounce": {"win_rate":40,"pf":1.96,"expectancy":2.14,"hold":"5-15 days",
                     "edge":"Best long strategy by backtest. Price bounces off rising 21 EMA in strong uptrend.",
                     "ideal":"EXPANSION","ok":"ACCUMULATION","blocked":"PANIC"},
    "VCP":          {"win_rate":45,"pf":1.80,"expectancy":3.50,"hold":"15-40 days",
                     "edge":"Minervini VCP. Tight base, vol dry-up, breakout confirmation. High reward outliers.",
                     "ideal":"EXPANSION,ACCUMULATION","ok":"","blocked":"PANIC,DISTRIBUTION"},
    "52WH_Breakout":{"win_rate":35,"pf":1.86,"expectancy":5.82,"hold":"20-60 days",
                     "edge":"52-week high breakout. Rare but highest reward. Best in bull markets.",
                     "ideal":"EXPANSION","ok":"ACCUMULATION","blocked":"PANIC,DISTRIBUTION"},
    "Failed_Breakout_Short":{"win_rate":24,"pf":1.60,"expectancy":3.12,"hold":"3-10 days",
                     "edge":"Most consistent across regimes. Breakout failure = trapped longs = fast decline.",
                     "ideal":"DISTRIBUTION,PANIC","ok":"ACCUMULATION","blocked":""},
    "Last30Min_ATH":{"win_rate":68,"pf":2.10,"expectancy":0.89,"hold":"Overnight BTST",
                     "edge":"Highest win rate. Stock at ATH in last 30 min — overnight gap-up momentum.",
                     "ideal":"EXPANSION","ok":"ACCUMULATION","blocked":"PANIC,DISTRIBUTION"},
}

_SYS = """You are the central intelligence of an autonomous Multi-Agent Swarm Trading System for NSE India.
You think like a combination of Mark Minervini (technical precision), Stanley Druckenmiller (macro + momentum),
and Howard Marks (risk management). You analyse every trade through THREE specialized agents with regime-adaptive weighting.
Be direct, specific, data-driven. Use exact price levels. Never hedge every sentence. Commit to a view."""


def _build_prompt(ticker: str, m: Dict, regime: Dict, sent: Dict,
                  rec: Dict, sig: Optional[Dict]) -> str:

    # Strategy context block
    strat_ctx = ""
    if sig:
        strat = sig.get("strategy","")
        kb = _STRATEGY_KB.get(strat,{})
        if kb:
            strat_ctx = f"""
╔══ STRATEGY CONTEXT: {strat} ══╗
Historical Edge  : Win Rate {kb['win_rate']}% | Profit Factor {kb['pf']} | Expectancy +{kb['expectancy']}%/trade
Hold Period      : {kb['hold']}
Edge Description : {kb['edge']}
Regime Fit       : Ideal={kb['ideal']} | Blocked={kb['blocked']}
Current Regime   : {sig.get('regime','?')} → Fit = {sig.get('regime_fit','?')}
╚══════════════════════════════════╝"""

    scanner_ctx = ""
    if sig:
        scanner_ctx = f"""
╔══ NSE SCANNER PRO SIGNAL ══════════════════════════════╗
Strategy    : {sig.get('strategy','N/A')}     Signal: {sig.get('signal','N/A')}
Entry       : ₹{sig.get('entry','?')}         SL: ₹{sig.get('sl','?')}
Target 1    : ₹{sig.get('t1','?')}            Target 2: ₹{sig.get('t2','?')}
R:R Ratio   : {sig.get('rr','?')}             Hold: {_STRATEGY_KB.get(sig.get('strategy',''),{}).get('hold','?')}
RS Rating   : {sig.get('rs',sig.get('rs_rating','?'))}
SQI Score   : {sig.get('sqi','?')}/100 Grade: {sig.get('sqi_grade','?')}
SQI Breakdown: {sig.get('sqi_breakdown', 'Edge:? RS:? Regime:? Vol:? Confirm:?')}
Sector      : {sig.get('sector','?')}
Regime Fit  : {sig.get('regime_fit','?')}
Scanner Reasons:
{chr(10).join('  → ' + r for r in (sig.get('reasons') or [sig.get('reason','Signal generated by scanner')]))}
╚════════════════════════════════════════════════════════╝
{strat_ctx}"""

    # Multi-timeframe trend summary
    gc = m.get("golden_cross")
    gc_str = ("✅ Golden Cross active" if gc else "❌ No Golden Cross") if gc is not None else "Insufficient data"

    bb_squeeze_str = "🔒 BB SQUEEZE — volatility compression, breakout imminent" if m.get("bb_squeeze") else f"BB Width {m.get('bb_width',0):.1f}% — {'tight' if m.get('bb_width',10)<6 else 'normal'}"

    return f"""Analyse {ticker} for NSE India. Today: {__import__('datetime').date.today()}
{scanner_ctx}
╔══ LIVE MARKET REGIME ══════════════════════════════════════════╗
Regime      : {regime.get('regime')} [{regime.get('code')}] — {regime.get('confidence',0)}% confidence
Description : {regime.get('description')}
VIX         : {regime.get('vix','N/A')}  (>22=fear, <13=complacency, 14-19=normal)
Nifty 50    : ₹{regime.get('nifty_level','N/A')} | 1D: {regime.get('nifty_change',0):+.2f}% | 5D: {regime.get('nifty_change_5d',0):+.2f}% | 20D: {regime.get('nifty_change_20d',0):+.2f}%
Breadth     : {regime.get('breadth','N/A')} ({regime.get('advances',0)} adv / {regime.get('declines',0)} dec of 15 sample)
Regime Strat: {regime.get('strategy')}
Agent Wts   : ATLAS {regime.get('atlas_weight')}% | ORACLE {regime.get('oracle_weight')}% | SENTINEL {regime.get('sentinel_weight')}%
╚════════════════════════════════════════════════════════════════╝

╔══ TECHNICAL ANALYSIS — {ticker} ══════════════════════════════════╗
PRICE ACTION:
  CMP         : ₹{m.get('current_price',0):,.2f}
  Performance : 1D {m.get('price_change_1d',0):+.2f}% | 5D {m.get('price_change_5d',0):+.2f}% | 1M {m.get('price_change_1m',0):+.2f}% | 3M {m.get('price_change_3m',0):+.2f}%
  52W Range   : Low ₹{m.get('low_52w',0):,.2f} (+{m.get('pct_from_52w_low',0):.1f}%) ← CMP → High ₹{m.get('high_52w',0):,.2f} ({m.get('pct_from_52w_high',0):+.1f}%)

MOVING AVERAGES:
  vs SMA20    : {m.get('price_vs_sma20_pct',0):+.2f}%  |  SMA20={_fp(m.get('sma_20',0))}
  vs EMA21    : CMP {'above' if m.get('current_price',0)>m.get('ema_21',0) else 'below'} EMA21={_fp(m.get('ema_21',0))}
  vs SMA50    : {m.get('price_vs_sma50_pct',0):+.2f}%  |  SMA50={_fp(m.get('sma_50',0))}
  vs SMA200   : {f"{m.get('price_vs_sma200_pct',0):+.2f}%" if m.get('price_vs_sma200_pct') is not None else 'N/A'}
  MA Alignment: {m.get('trend_alignment_str','?')}  |  {gc_str}
  SMA20/50    : {m.get('sma20_vs_sma50','?')}

MOMENTUM:
  RSI(14)     : {m.get('rsi_14',0):.1f} [{m.get('rsi_status')}]  RSI(9)={m.get('rsi_9',0):.1f}  Divergence={m.get('rsi_divergence','')}
  MACD        : {m.get('macd_status','')}  Line={m.get('macd_line',0):.3f}  Hist={m.get('macd_histogram',0):.3f} (prev={m.get('macd_prev_histogram',0):.3f})
  Stochastic  : K={m.get('stoch_k',0):.1f} D={m.get('stoch_d',0):.1f} [{m.get('stoch_status','')}]
  ADX(14)     : {m.get('adx_14',0):.1f} [{m.get('trend_strength','')}]  (>25=trending, <20=choppy)

VOLATILITY:
  Bollinger   : Upper={_fp(m.get('bb_upper',0))} | Lower={_fp(m.get('bb_lower',0))} | Position={m.get('bb_position','')} ({m.get('bb_pct',0)*100:.0f}th percentile)
  {bb_squeeze_str}
  ATR%        : {m.get('atr_pct',0):.2f}%/day
  RealVol 20D : {m.get('realized_volatility_20d',0):.1f}% annualized | 5D: {m.get('realized_volatility_5d',0):.1f}%

VOLUME & INSTITUTIONAL FLOW:
  Vol Ratio   : {m.get('volume_ratio',0):.2f}× (20D avg) | {m.get('volume_ratio_50d',0):.2f}× (50D avg)
  Vol Trend   : {m.get('volume_trend','')}
  A/D Ratio   : {m.get('accumulation_ratio',0):.2f} → {m.get('accumulation','')}

TREND STRUCTURE:
  Short-term  : {m.get('trend_short_term','')}  |  Medium-term: {m.get('trend_medium_term','')}  |  Long-term: {m.get('trend_long_term','')}
  RS Proxy    : {m.get('rs_proxy',0):.0f}/100 (3M+1M momentum-based)
╚════════════════════════════════════════════════════════════════╝

╔══ PRICE ACTION SENTIMENT ══════════════════════════════════════╗
Overall     : {sent.get('overall')} (Score: {sent.get('score')}/8)
Key Factors :
{chr(10).join('  ' + f for f in sent.get('factors',[]))}
╚════════════════════════════════════════════════════════════════╝

╔══ PRE-COMPUTED AGENT SIGNALS ══════════════════════════════════╗
ATLAS    : {rec.get('atlas_signal')}  (raw score {rec.get('atlas_score',0):+d}, weight {regime.get('atlas_weight')}%)
ORACLE   : {rec.get('oracle_signal')}  (score {rec.get('oracle_score',0):+d}, conf {rec.get('oracle_confidence')}%, weight {regime.get('oracle_weight')}%)
SENTINEL : Risk={rec.get('sentinel_risk')}  (raw score {rec.get('sentinel_score',0)}, weight {regime.get('sentinel_weight')}%)
Weighted : {rec.get('weighted_score'):+.3f}  →  FINAL DECISION: {rec.get('final_recommendation')}
╚════════════════════════════════════════════════════════════════╝

Provide a COMPLETE, PROFESSIONAL analysis in this EXACT format.
Be specific with price levels. Be direct — no wishy-washy language.
If bullish, say why with conviction. If bearish, say so clearly.

═══════════════════════════════════════════════════════════════════
## 🏛️ 1. MARKET REGIME & MACRO CONTEXT
[3-4 sentences: Current regime implications. How VIX + breadth + Nifty trend affect THIS specific trade.
What the regime says about position sizing and risk appetite right now.]

═══════════════════════════════════════════════════════════════════
## 🔬 2. AGENT ANALYSIS

### 🔵 AGENT A — ATLAS (Technical Eye) | {rec.get('atlas_signal')} | {regime.get('atlas_weight')}% weight

**Signal Conviction:** [High/Medium/Low] — [1 sentence why]

**Price Structure:**
[Describe the dominant chart pattern visible — base, breakout, breakdown, consolidation, trend.
Reference specific price levels: support at ₹X, resistance at ₹X, pivot at ₹X]

**Multi-Timeframe Alignment:**
[Comment on all 4 trend readings: short/medium/long-term + MA alignment score.
{m.get('trend_alignment',0)}/5 MAs bullish — what does this mean?]

**Momentum Analysis:**
[RSI {m.get('rsi_14',0):.1f} — is this healthy momentum or extended/exhausted?
MACD {m.get('macd_status','')} — is histogram expanding or fading? 
ADX {m.get('adx_14',0):.1f} — is this a trending or choppy move?]

**Bollinger Band Context:**
[Price at {m.get('bb_position','')} ({m.get('bb_pct',0)*100:.0f}th percentile).
{bb_squeeze_str}. What does this mean for the next move?]

**Key Levels:**
- Support: ₹[level] (reason)
- Resistance: ₹[level] (reason)  
- Stop Loss: ₹[your recommended level]
- Targets: T1=₹[level], T2=₹[level]

**Risk Flags:** [Technical concerns — divergences, extended moves, weak volume, etc.]

---

### 🟡 AGENT B — ORACLE (Sentiment & Flow) | {rec.get('oracle_signal')} | {regime.get('oracle_weight')}% weight

**Signal Conviction:** [High/Medium/Low] — [1 sentence why]

**Institutional Flow Reading:**
[Analyse volume: {m.get('volume_ratio',0):.2f}× ratio, A/D={m.get('accumulation_ratio',0):.2f} ({m.get('accumulation','')}).
Are institutions buying or selling? What does the 10-day accumulation pattern suggest?]

**Relative Strength Assessment:**
[Stock is {m.get('pct_from_52w_high',0):+.1f}% from 52W high.
RS Proxy score {m.get('rs_proxy',0):.0f}/100.
Is this a market leader, market follower, or laggard? Should you be buying it?]

**Price Action Sentiment Summary:**
[Reference the {len(sent.get('factors',[]))} sentiment factors. What's the dominant theme?
Accumulation? Distribution? Breakout attempt? Exhaustion?]

{f'**Scanner Signal Quality:**' + chr(10) + '[SQI ' + str(sig.get("sqi","?")) + '/100 (' + str(sig.get("sqi_grade","?")) + '). Validate each SQI component. Does the ' + sig.get("strategy","strategy") + " deserve execution at this SQI grade?]" if sig else ""}

**Risk Flags:** [Sentiment risks — distribution, weak RS, low volume breakouts, etc.]

---

### 🔴 AGENT C — SENTINEL (Risk Manager) | {rec.get('sentinel_risk')} Risk | {regime.get('sentinel_weight')}% weight

**Risk Level Justification:**
[Why {rec.get('sentinel_risk')} risk? Reference volatility {m.get('realized_volatility_20d',0):.1f}%, ATR% {m.get('atr_pct',0):.2f}%, and regime {regime.get('code','')}]

**Position Sizing:**
[Given {rec.get('sentinel_risk')} risk and {regime.get('code','')} regime, recommend position size as % of portfolio.
Use regime heat caps: EXPANSION=6%, ACCUMULATION=4%, DISTRIBUTION=2%, PANIC=1%]

**Risk:Reward Analysis:**
{f"[Scanner R:R is {sig.get('rr','?')}. Is this acceptable? Minimum 2:1 required. Adjust levels if needed.]" if sig else "[Calculate R:R based on recommended entry/stop/target levels above]"}

**Volatility Context:**
[20D realized vol {m.get('realized_volatility_20d',0):.1f}% — daily expected move ±₹{m.get('current_price',0)*m.get('atr_pct',0)/100:.2f}.
What does this mean for stop placement? Avoid stops inside daily noise.]

**Risk Flags:** [Portfolio-level risks — sector concentration, correlation with market, black swan scenarios]

═══════════════════════════════════════════════════════════════════
## ⚖️ 3. SWARM CONSENSUS

**Final Decision:** {rec.get('final_recommendation')} | **Overall Conviction:** [High/Medium/Low]
**Weighted Score:** {rec.get('weighted_score'):+.3f} (threshold ±0.40 — distance from threshold matters)

**Agent Agreement:** [Do all 3 agents agree? If split, who wins and why based on regime weights?]

{f'**Scanner Signal Validation:** [Specifically validate the ' + sig.get("strategy","") + ' signal. ' + 'Does the AI confirm or override the scanner? Address Entry ₹' + str(sig.get("entry","?")) + ', SL ₹' + str(sig.get("sl","?")) + ', T1 ₹' + str(sig.get("t1","?")) + '. Is the R:R of ' + str(sig.get("rr","?")) + ' worth taking given current conditions?]' if sig else ''}

**Bull Case:** [3 things that need to go right for this to work]
**Bear Case:** [3 things that could go wrong]

═══════════════════════════════════════════════════════════════════
## 📋 4. EXECUTION PLAN

**Action:** [BUY/SELL/HOLD/WAIT] — [precise trigger condition]

**Entry Strategy:**
- Entry: ₹[level] — [condition: e.g., "above 3pm close", "on volume confirmation", "at open"]
- Type: [Breakout buy / Pullback buy / Breakdown short / etc.]

**Risk Management:**
- Stop Loss: ₹[level] — [logic: below support/EMA/ATR-based]
- Position Size: [X% of capital] based on [regime + volatility reasoning]
- Risk per trade: ₹[amount] per ₹1L capital

**Targets:**
- T1: ₹[level] — [exit 40-50% here] — [R multiple]R
- T2: ₹[level] — [exit 30-40% here] — [R multiple]R
- T3 / Trail: ₹[level] — [exit rest, use trailing stop]

**Trade Management:**
- If price does X: [action]
- If price does Y: [action]

═══════════════════════════════════════════════════════════════════
## 🚫 5. INVALIDATION TRIGGERS

[List exactly 4-5 specific price levels or events that kill this thesis immediately:
1. ₹[level] — [what it breaks]
2. [specific event or candle pattern]
3. [volume or breadth condition]
4. [regime change trigger]
5. [time-based rule — e.g., "if not at T1 within 10 days, exit"]]

═══════════════════════════════════════════════════════════════════
## 📊 6. QUICK SCORECARD

| Factor | Score | Comment |
|--------|-------|---------|
| Trend Alignment | {m.get('trend_alignment',0)}/5 | {m.get('trend_alignment_str','')} |
| Momentum | [1-5] | RSI + MACD summary |
| Volume/Flow | [1-5] | Accumulation analysis |
| Risk:Reward | [1-5] | Entry/Exit quality |
| Regime Fit | [1-5] | Strategy vs current regime |
| **TOTAL** | **/25** | **[Verdict: Strong Buy / Buy / Neutral / Avoid]** |
"""


# ══════════════════════════════════════════════════════════════════════════════
# AI CALLS
# ══════════════════════════════════════════════════════════════════════════════

def _call_groq(key, prompt):
    """LEGACY — kept for signature compatibility, no longer used.
    v2 routes everything to Claude via ai_engine.py.
    """
    raise NotImplementedError("Use _call_claude_via_engine instead")

def _call_gemini(key, model, prompt):
    """LEGACY — kept for signature compatibility, no longer used."""
    raise NotImplementedError("Use _call_claude_via_engine instead")

def _run_ai(ticker, m, regime, sent, rec, sig, groq_key, gemini_key, gemini_model):
    """v2: All AI calls routed to Claude via ai_engine.py.
    
    If a scanner signal context is provided → uses analyze_signal (Haiku, cheap).
    Otherwise → uses deep_dive (Sonnet, premium).
    Returns rendered markdown.
    """
    from ai_engine import (
        analyze_signal, deep_dive,
        render_signal_verdict, render_deep_dive,
    )
    
    # Check if user wants premium (Sonnet) — read from session state, default False
    use_premium = st.session_state.get("ai_use_premium", False)
    
    try:
        if sig:
            with st.spinner(f"🧠 Claude {'Sonnet 4.6' if use_premium else 'Haiku 4.5'} validating signal…"):
                verdict = analyze_signal(
                    symbol=ticker,
                    strategy=sig.get("strategy", ""),
                    signal=sig.get("signal", "BUY"),
                    cmp=m.get("current_price", 0),
                    entry=float(sig.get("entry") or m.get("current_price", 0)),
                    stop_loss=float(sig.get("sl") or 0),
                    target_1=float(sig.get("t1") or 0),
                    target_2=float(sig.get("t2") or 0),
                    rr=float(sig.get("rr") or 0),
                    rsi=m.get("rsi_14", 50),
                    volume_ratio=m.get("volume_ratio", 1.0),
                    sqi=float(sig.get("sqi") or 0),
                    sqi_grade=sig.get("sqi_grade", "?"),
                    rs_rating=float(sig.get("rs") or 50),
                    sector=sig.get("sector", ""),
                    regime=regime.get("regime", "UNKNOWN"),
                    regime_score=regime.get("score", 0),
                    regime_fit=sig.get("regime_fit", "OK"),
                    reasons=(sig.get("reasons") or [sig.get("reason", "Scanner signal")]),
                    use_premium=use_premium,
                    sector_quadrant=regime.get("sector_quadrant"),
                )
            return render_signal_verdict(verdict) if verdict else None
        else:
            with st.spinner("🧠 Claude Sonnet 4.6 deep-diving…"):
                dd_result = deep_dive(
                    symbol=ticker, metrics=m, regime=regime,
                    sentiment=sent, recommendation=rec,
                    signal_context=None,
                )
            return render_deep_dive(dd_result) if dd_result else None
    except Exception as e:
        st.error(f"❌ Claude API call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE + HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_open_signals():
    try:
        from signal_tracker import _get_supabase
        sb = _get_supabase()
        if not sb: return []
        res = (sb.table("signals")
               .select("symbol,strategy,signal,cmp,entry,sl,t1,t2,rr,sqi,sqi_grade,sqi_breakdown,sector,regime,regime_fit,rs,reasons,created_at")
               .eq("status","OPEN")
               .order("sqi",desc=True)  # sort by quality
               .limit(200)
               .execute())
        return res.data or []
    except Exception as e:
        print(f"[AI Deep Dive] Supabase load: {e}")
        return []

def _fp(v):
    try:
        v=float(v)
        return f"₹{v:,.0f}" if v>=10000 else f"₹{v:,.1f}" if v>=100 else f"₹{v:,.2f}"
    except: return str(v)

def _sig_col(s): return {"BUY":"#26a69a","SELL":"#ef5350","HOLD":"#ff9800"}.get(s,"#9e9e9e")
def _risk_col(r): return {"HIGH":"#ef5350","MEDIUM":"#ff9800","LOW":"#26a69a"}.get(r,"#9e9e9e")
def _regime_bg(code):
    return {"stress":"#922b21","accumulate":"#6c3483","trend":"#0e6655",
            "goldilocks":"#7d3c98","uncertainty":"#424949","unknown":"#2c3e50"}.get(code.lower(),"#2c3e50")


# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════

def _chart(df, sym, sig=None):
    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.04,
                        row_heights=[0.6,0.2,0.2],
                        subplot_titles=(f"{sym} — Price Action","RSI (14)","Volume"))
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name="Price",
        increasing_line_color="#26a69a",decreasing_line_color="#ef5350"),row=1,col=1)
    for col,name,color in [("_sma20","SMA 20","#ff9800"),("_ema21","EMA 21","#ffd700"),("_sma50","SMA 50","#42a5f5")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index,y=df[col],name=name,
                line=dict(color=color,width=1.5)),row=1,col=1)
    if sig:
        for price,label,color,dash in [
            (sig.get("entry"),"Entry","#42a5f5","dash"),
            (sig.get("sl"),"SL","#ef5350","dot"),
            (sig.get("t1"),"T1","#26a69a","dash"),
            (sig.get("t2"),"T2","#66bb6a","dash")]:
            if price:
                try: fig.add_hline(y=float(price),line_dash=dash,line_color=color,
                                   annotation_text=f"{label} {_fp(price)}",row=1,col=1)
                except: pass
    if "_rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["_rsi"],name="RSI",
            line=dict(color="#ab47bc",width=1.5)),row=2,col=1)
        fig.add_hline(y=70,line_dash="dash",line_color="#ef5350",row=2,col=1)
        fig.add_hline(y=30,line_dash="dash",line_color="#26a69a",row=2,col=1)
    colors=["#ef5350" if r["Close"]<r["Open"] else "#26a69a" for _,r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume",marker_color=colors),row=3,col=1)
    fig.update_layout(height=780,xaxis_rangeslider_visible=False,template="plotly_dark",
                      showlegend=True,hovermode="x unified",
                      margin=dict(l=50,r=60,t=55,b=30),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="monospace",size=11))
    return fig


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
      border-radius:10px;padding:12px 18px;margin:10px 0;font-size:.86rem;}
    .sqi-bar{height:6px;border-radius:3px;background:linear-gradient(90deg,#ef5350,#ff9800,#26a69a);}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dd-title">🧠 AI DEEP DIVE</div>', unsafe_allow_html=True)
    st.caption("Exhaustive Multi-Agent Swarm Analysis · ATLAS (Technical) · ORACLE (Flow/Sentiment) · SENTINEL (Risk)")

    # API keys — v2 uses Claude only
    try:
        anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        anthropic_key = ""
    import os as _os
    anthropic_key = anthropic_key or _os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        st.error("❌ ANTHROPIC_API_KEY missing. Add it to Streamlit Secrets to enable AI Deep Dive.")
        st.info("Get a key at https://console.anthropic.com/settings/keys")
        return
    # Legacy keys kept for backward-compat references later in the function
    groq_key = ""
    gemini_key = ""

    # ── Premium model toggle ──
    st.sidebar.markdown("### 🧠 AI Settings")
    st.session_state["ai_use_premium"] = st.sidebar.checkbox(
        "Use Sonnet 4.6 (premium, ~5x cost)",
        value=st.session_state.get("ai_use_premium", False),
        help="Sonnet 4.6 gives deeper reasoning. Default Haiku 4.5 is fast and cheap (~₹0.35/call)."
    )
    from ai_engine import get_cost_summary, reset_cost_tracker as _reset_cost
    _cost = get_cost_summary()
    st.sidebar.metric("Claude calls (session)", _cost["calls"])
    st.sidebar.metric("Cost", f"${_cost['total_usd']:.4f}", delta=f"~₹{_cost['total_inr']:.2f}")
    if st.sidebar.button("Reset cost tracker", key="dd_cost_reset"):
        _reset_cost(); st.rerun()

    # Load Supabase signals
    with st.spinner("Loading open signals from Supabase…"):
        open_sigs = _load_open_signals()

    sig_lookup: Dict[str, Dict] = {}
    for s in open_sigs:
        grade_icon = {"ELITE":"🏆","STRONG":"⭐","MODERATE":"✅","WEAK":"⚠️"}.get(s.get("sqi_grade",""),"")
        label = (f"{s['symbol']}  ·  {s['strategy']}  ·  {s['signal']}"
                 f"  ·  {grade_icon}SQI {s.get('sqi','?')} ({s.get('sqi_grade','?')})"
                 f"  ·  RR {s.get('rr','?')}  ·  RS {s.get('rs','?')}")
        sig_lookup[label] = s

    # ── Input ──────────────────────────────────────────────────────────────
    st.divider()
    tab_signal, tab_manual = st.tabs(["📡 From Scanner Signals", "🔍 Any Stock (1000+)"])

    sig_ctx: Optional[Dict] = None
    ticker = ""

    with tab_signal:
        if sig_lookup:
            picked = st.selectbox("Pick a live OPEN signal (sorted by SQI quality)",
                                  ["— Choose a signal —"] + list(sig_lookup.keys()), key="dd_pick")
            if picked != "— Choose a signal —":
                sig_ctx = sig_lookup[picked]
                ticker = sig_ctx["symbol"]
        else:
            st.info("No OPEN signals. Run a scan first, or use the 'Any Stock' tab.")
        run_signal = st.button("🚀 Analyse Signal", type="primary", key="dd_run_sig",
                               disabled=(not ticker))

    with tab_manual:
        st.info("Type any NSE symbol. Covers Nifty 500 + midcap + smallcap (~1000 stocks via live NSE feed).")
        universe = _get_nse_universe()
        col1, col2 = st.columns([3,1])
        with col1:
            manual_input = st.text_input("Enter NSE symbol",
                placeholder="e.g. RELIANCE  or  HDFCBANK  or  KAYNES", key="dd_manual")
            # Search helper
            if manual_input and len(manual_input) >= 2:
                matches = [s for s in universe if manual_input.upper() in s.upper()][:8]
                if matches and manual_input.upper() not in universe:
                    st.caption(f"Suggestions: {', '.join(matches)}")
        with col2:
            gemini_model = st.selectbox("AI Model fallback",["gemini-2.0-flash","gemini-2.5-flash"],key="dd_gem2")
        run_manual = st.button("🚀 Analyse Stock", type="primary", key="dd_run_manual",
                               disabled=(not manual_input.strip()))
        if manual_input.strip():
            ticker = manual_input.strip().upper().replace(".NS","")

    run_btn = (run_signal and sig_ctx) or (run_manual and manual_input.strip())

    # ── If Analyse clicked, wipe any previous cached result ─────────────────
    if run_btn:
        st.session_state.pop("dd_result", None)

    # ── Idle state: show open signals table only if no saved result either ──
    if not run_btn and "dd_result" not in st.session_state:
        if open_sigs:
            st.divider()
            st.markdown("#### 📋 Open Signals — sorted by SQI quality")
            rows = []
            for s in open_sigs:
                sqi = s.get("sqi") or 0
                grade = s.get("sqi_grade","")
                grade_icon = {"ELITE":"🏆","STRONG":"⭐","MODERATE":"✅","WEAK":"⚠️"}.get(grade,"")
                rows.append({"Symbol":s["symbol"],"Strategy":s["strategy"],"Signal":s["signal"],
                    "CMP":_fp(s.get("cmp",0)),"Entry":_fp(s.get("entry",0)),
                    "SL":_fp(s.get("sl",0)),"T1":_fp(s.get("t1",0)),
                    "R:R":s.get("rr","—"),f"{grade_icon} SQI": f"{sqi}" if sqi else "—",
                    "Grade":grade,
                    "RS":s.get("rs","—"),"Sector":s.get("sector","—"),"Fit":s.get("regime_fit","—")})
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,height=320)
        st.info("👆 Pick a signal above or type any stock in the 'Any Stock' tab, then click Analyse.")
        return

    # ── If no new run_btn but we have a saved result, restore from state ────
    if not run_btn and "dd_result" in st.session_state:
        _r = st.session_state["dd_result"]
        ticker   = _r["ticker"]
        m        = _r["m"]
        df       = _r["df"]
        regime   = _r["regime"]
        sent     = _r["sent"]
        rec      = _r["rec"]
        sig_ctx  = _r["sig_ctx"]
        analysis = _r["analysis"]
        gemini_model = _r.get("gemini_model", "gemini-2.0-flash")
        # Skip data fetch / computation — jump straight to rendering
        _render_deep_dive_results(
            ticker, m, df, regime, sent, rec, sig_ctx, analysis,
            gemini_model, groq_key, gemini_key, open_sigs
        )
        return

    if not ticker:
        st.error("Please pick a signal or enter a ticker.")
        return

    ns_ticker = ticker + ".NS"

    # Scanner context banner
    if sig_ctx:
        sqi_val = sig_ctx.get("sqi",0) or 0
        grade = sig_ctx.get("sqi_grade","")
        grade_colors = {"ELITE":"#ffd700","STRONG":"#5dade2","MODERATE":"#f0b429","WEAK":"#e74c3c"}
        gc = grade_colors.get(grade,"#888")
        st.markdown(f"""<div class="ctx-box">
            <b>📡 {sig_ctx['strategy']}</b> &nbsp;·&nbsp; <b>{sig_ctx['symbol']}</b>
            &nbsp;·&nbsp; Signal: <b>{sig_ctx['signal']}</b>
            &nbsp;·&nbsp; <span style="color:{gc}">SQI <b>{sqi_val}/100 ({grade})</b></span>
            &nbsp;·&nbsp; RS: <b>{sig_ctx.get('rs','?')}</b><br>
            Entry <b>{_fp(sig_ctx.get('entry','?'))}</b> &nbsp;|&nbsp;
            SL <b>{_fp(sig_ctx.get('sl','?'))}</b> &nbsp;|&nbsp;
            T1 <b>{_fp(sig_ctx.get('t1','?'))}</b> &nbsp;|&nbsp;
            T2 <b>{_fp(sig_ctx.get('t2','?'))}</b> &nbsp;|&nbsp;
            R:R <b>{sig_ctx.get('rr','?')}</b> &nbsp;|&nbsp;
            Regime Fit: <b>{sig_ctx.get('regime_fit','?')}</b> &nbsp;|&nbsp;
            Sector: <b>{sig_ctx.get('sector','?')}</b>
            <div style="margin-top:6px;font-size:.8rem;color:#aaa">
            SQI Breakdown: {sig_ctx.get('sqi_breakdown','N/A')}</div>
        </div>""", unsafe_allow_html=True)

    # Fetch data
    with st.spinner(f"Fetching 9 months data for {ns_ticker}…"):
        try:
            df = yf.Ticker(ns_ticker).history(period="9mo")
        except Exception as e:
            st.error(f"Data fetch failed: {e}"); return

    if df is None or df.empty or len(df)<30:
        st.error(f"Insufficient data for {ns_ticker}. Try without the .NS suffix.")
        return

    with st.spinner("Computing 20+ technical indicators…"):
        regime = _fetch_regime()
        m = _compute_metrics(df)
    if not m: st.error("Metrics computation failed."); return

    sent = _sentiment(m)
    rec  = _decision(m,regime,sent)

    # ── Metrics dashboard ──────────────────────────────────────────────────
    st.divider()
    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    c1.metric("Price",    _fp(m["current_price"]),     f"{m['price_change_1d']:+.2f}%")
    c2.metric("1M Perf",  f"{m['price_change_1m']:+.1f}%", f"3M {m['price_change_3m']:+.1f}%")
    c3.metric("RSI",      f"{m['rsi_14']:.1f}",        m["rsi_status"])
    c4.metric("ADX",      f"{m['adx_14']:.1f}",        m["trend_strength"])
    c5.metric("Vol Ratio",f"{m['volume_ratio']:.2f}×", m["accumulation"])
    c6.metric("MAs Align",f"{m['trend_alignment']}/5", m["trend_long_term"])
    c7.metric("Sentiment",sent["overall"])
    c8.metric("AI Signal",rec["final_recommendation"], f"Score {rec['weighted_score']:+.3f}")

    # ── Regime banner ──────────────────────────────────────────────────────
    bg = _regime_bg(regime.get("code","unknown"))
    st.markdown(f"""<div style="background:{bg};padding:10px 18px;border-radius:10px;
        color:#fff;margin:10px 0;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div><b style="font-size:1.05rem">{regime['regime']}</b>
        <span style="opacity:.8;font-size:.82rem;margin-left:8px">{regime['description']}</span></div>
        <div style="margin-left:auto;font-size:.8rem;text-align:right">
        VIX <b>{regime['vix']}</b> &nbsp;·&nbsp;
        Nifty <b>₹{regime.get('nifty_level','?')}</b> 1D{regime.get('nifty_change',0):+.2f}% 5D{regime.get('nifty_change_5d',0):+.2f}%
        &nbsp;·&nbsp; Breadth <b>{regime.get('breadth','?')}</b> ({regime.get('advances',0)}A/{regime.get('declines',0)}D)
        </div></div>
    """, unsafe_allow_html=True)

    # ── Agent cards ────────────────────────────────────────────────────────
    a1,a2,a3 = st.columns(3)
    with a1:
        cls="bull" if rec["atlas_signal"]=="BUY" else ("bear" if rec["atlas_signal"]=="SELL" else "neut")
        st.markdown(f"""<div class="agent-box {cls}">
            <div style="font-size:.7rem;color:#888;letter-spacing:1px">⚡ ATLAS · TECHNICAL · {regime['atlas_weight']}% weight</div>
            <div style="margin-top:6px"><span class="sbadge" style="background:{_sig_col(rec['atlas_signal'])}">{rec['atlas_signal']}</span></div>
            <div style="font-size:.78rem;margin-top:6px;color:#ccc">
            {m.get('trend_short_term')} trend · {m.get('trend_alignment',0)}/5 MAs<br>
            MACD: {m.get('macd_status','')} · ADX {m['adx_14']:.0f} · BB {m.get('bb_position','')}</div>
        </div>""",unsafe_allow_html=True)
    with a2:
        cls="bull" if rec["oracle_signal"]=="BUY" else ("bear" if rec["oracle_signal"]=="SELL" else "neut")
        st.markdown(f"""<div class="agent-box {cls}">
            <div style="font-size:.7rem;color:#888;letter-spacing:1px">🔮 ORACLE · FLOW+SENTIMENT · {regime['oracle_weight']}% weight</div>
            <div style="margin-top:6px">
            <span class="sbadge" style="background:{_sig_col(rec['oracle_signal'])}">{rec['oracle_signal']}</span>
            <span style="font-size:.75rem;color:#888;margin-left:8px">{rec['oracle_confidence']}% conf</span></div>
            <div style="font-size:.78rem;margin-top:6px;color:#ccc">
            {sent['overall']} (score {sent['score']})<br>
            Vol {m['volume_ratio']:.2f}× · A/D {m.get('accumulation_ratio',0):.2f} · RS {m.get('rs_proxy',0):.0f}/100</div>
        </div>""",unsafe_allow_html=True)
    with a3:
        cls="bear" if rec["sentinel_risk"]=="HIGH" else ("bull" if rec["sentinel_risk"]=="LOW" else "neut")
        st.markdown(f"""<div class="agent-box {cls}">
            <div style="font-size:.7rem;color:#888;letter-spacing:1px">🛡️ SENTINEL · RISK · {regime['sentinel_weight']}% weight</div>
            <div style="margin-top:6px"><span class="sbadge" style="background:{_risk_col(rec['sentinel_risk'])}">RISK: {rec['sentinel_risk']}</span></div>
            <div style="font-size:.78rem;margin-top:6px;color:#ccc">
            Vol {m['realized_volatility_20d']:.0f}% · ATR% {m['atr_pct']:.2f}%<br>
            {'🔒 BB Squeeze' if m.get('bb_squeeze') else f"BB at {m.get('bb_position','')}"}</div>
        </div>""",unsafe_allow_html=True)

    # ── Chart ──────────────────────────────────────────────────────────────
    st.plotly_chart(_chart(df,ticker,sig_ctx),use_container_width=True)

    # ── Expandable technical details ───────────────────────────────────────
    with st.expander("📊 Full Technical Snapshot", expanded=False):
        col1,col2,col3 = st.columns(3)
        with col1:
            st.markdown("**Price & MAs**")
            st.markdown(f"SMA20: {_fp(m['sma_20'])} | EMA21: {_fp(m['ema_21'])}")
            st.markdown(f"SMA50: {_fp(m['sma_50'])} | SMA200: {_fp(m['sma_200']) if m['sma_200'] else 'N/A'}")
            st.markdown(f"vs SMA20: {m['price_vs_sma20_pct']:+.2f}% | vs SMA50: {m['price_vs_sma50_pct']:+.2f}%")
            st.markdown(f"Golden Cross: {'✅' if m.get('golden_cross') else '❌'}")
        with col2:
            st.markdown("**Momentum**")
            st.markdown(f"RSI(14): {m['rsi_14']:.1f} RSI(9): {m['rsi_9']:.1f}")
            st.markdown(f"MACD Line: {m['macd_line']:.3f} Hist: {m['macd_histogram']:.3f}")
            st.markdown(f"Stoch K/D: {m['stoch_k']:.1f}/{m['stoch_d']:.1f} [{m['stoch_status']}]")
            st.markdown(f"ADX: {m['adx_14']:.1f} [{m['trend_strength']}]")
        with col3:
            st.markdown("**Volatility & Volume**")
            st.markdown(f"BB: {_fp(m['bb_lower'])} — {_fp(m['bb_upper'])}")
            st.markdown(f"BB Width: {m['bb_width']:.1f}% {'🔒 SQUEEZE' if m.get('bb_squeeze') else ''}")
            st.markdown(f"ATR%: {m['atr_pct']:.2f}% | RealVol: {m['realized_volatility_20d']:.1f}%")
            st.markdown(f"Vol 20D: {m['volume_ratio']:.2f}× | A/D: {m.get('accumulation_ratio',0):.2f}")

    # ── AI Analysis ────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"🤖 Swarm Analysis — {ticker}")
    _gm = "claude-haiku-4-5" if not st.session_state.get("ai_use_premium") else "claude-sonnet-4-6"
    st.caption(f"🧠 {_gm} · {'Premium reasoning' if 'sonnet' in _gm else 'Cost-efficient screening'} · Cached system prompt")

    # ── Rate limit: 30s between analysis calls per session ─────────────────
    import time as _time
    _last = st.session_state.get("dd_last_analysis_ts", 0)
    _now  = _time.time()
    _wait = 30 - (_now - _last)
    if _wait > 0 and st.session_state.get("dd_result"):
        st.warning(f"⏳ Please wait {int(_wait)}s before running another analysis (rate limit).")
        _render_deep_dive_results(
            st.session_state["dd_result"]["ticker"],
            st.session_state["dd_result"]["m"],
            st.session_state["dd_result"]["df"],
            st.session_state["dd_result"]["regime"],
            st.session_state["dd_result"]["sent"],
            st.session_state["dd_result"]["rec"],
            st.session_state["dd_result"]["sig_ctx"],
            st.session_state["dd_result"]["analysis"],
            _gm, groq_key, gemini_key, open_sigs
        )
        return
    st.session_state["dd_last_analysis_ts"] = _now

    analysis = _run_ai(ticker,m,regime,sent,rec,sig_ctx,groq_key,gemini_key, _gm)
    if analysis:
        st.markdown(analysis)

    if sent.get("factors"):
        with st.expander("📊 Sentiment Breakdown", expanded=False):
            for f in sent["factors"]: st.markdown(f"- {f}")

    # ── Save all computed data to session_state so buttons survive re-runs ──
    st.session_state["dd_result"] = {
        "ticker": ticker, "m": m, "df": df, "regime": regime,
        "sent": sent, "rec": rec, "sig_ctx": sig_ctx,
        "analysis": analysis, "gemini_model": _gm,
    }
    # ── Render remaining UI (buttons, gauge, news) via shared helper ────────
    _render_deep_dive_results(
        ticker, m, df, regime, sent, rec, sig_ctx, analysis,
        _gm, groq_key, gemini_key, open_sigs
    )


def _render_deep_dive_results(ticker, m, df, regime, sent, rec, sig_ctx, analysis,
                               gemini_model, groq_key, gemini_key, open_sigs):
    """
    Renders everything after the AI analysis text:
    Conviction Meter, Trade Calculator, Paper Trade, News, Share Buttons, Sector RRG.
    Extracted into a standalone function so that button clicks (which cause Streamlit
    re-runs) can restore from session_state and re-render without re-running analysis.
    """
    def _fp(v):
        try:
            v=float(v)
            return f"₹{v:,.0f}" if v>=10000 else f"₹{v:,.1f}" if v>=100 else f"₹{v:,.2f}"
        except: return str(v)

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE 1: CONVICTION GAUGE
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 🎯 Conviction Meter")
    ws = rec["weighted_score"]
    gauge_val = int((ws + 1) / 2 * 100)
    gauge_color = "#26a69a" if ws >= 0.4 else ("#ef5350" if ws <= -0.4 else "#ff9800")
    conviction_label = (
        "STRONG BUY" if ws >= 0.7 else "BUY" if ws >= 0.4 else
        "STRONG SELL" if ws <= -0.7 else "SELL" if ws <= -0.4 else "HOLD / WAIT"
    )
    _cg1, _cg2, _cg3 = st.columns([1,3,1])
    with _cg2:
        _atlas_w  = regime["atlas_weight"]
        _oracle_w = regime["oracle_weight"]
        _sent_w   = regime["sentinel_weight"]
        _at_sig   = rec["atlas_signal"]
        _or_sig   = rec["oracle_signal"]
        _se_risk  = rec["sentinel_risk"]
        st.markdown(
            '<div style="background:#1a1d23;border:1px solid #333;border-radius:12px;'
            'padding:18px 24px;text-align:center">'
            '<div style="font-size:.75rem;color:#888;letter-spacing:2px;margin-bottom:8px">'
            'SWARM CONVICTION</div>'
            f'<div style="height:12px;background:#e2e8f0;border-radius:6px;'
            f'margin-bottom:10px;overflow:hidden">'
            f'<div style="height:100%;width:{gauge_val}%;'
            f'background:linear-gradient(90deg,#ef5350,#ff9800,#26a69a);'
            f'border-radius:6px"></div></div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:{gauge_color};'
            f'letter-spacing:2px">{conviction_label}</div>'
            f'<div style="font-size:.88rem;color:#888;margin-top:4px">'
            f'Score <b style="color:{gauge_color}">{ws:+.3f}</b>'
            f' &nbsp;(threshold ±0.40)</div>'
            f'<div style="margin-top:10px;font-size:.78rem;color:#aaa">'
            f'ATLAS {_at_sig} ({_atlas_w}%) · ORACLE {_or_sig} ({_oracle_w}%) · '
            f'SENTINEL {_se_risk} risk ({_sent_w}%)'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE 2: TRADE CALCULATOR
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    with st.expander("🧮 Trade Calculator — Position Size & Risk", expanded=(sig_ctx is not None)):
        st.caption("Calculate exact shares, ₹ risk, and position size before placing the trade.")
        default_entry = float(sig_ctx.get("entry", m["current_price"])) if sig_ctx else m["current_price"]
        default_sl    = float(sig_ctx.get("sl",    m["current_price"] * 0.95)) if sig_ctx else m["current_price"] * 0.95
        default_t1    = float(sig_ctx.get("t1",    m["current_price"] * 1.10)) if sig_ctx else m["current_price"] * 1.10
        default_t2    = float(sig_ctx.get("t2",    m["current_price"] * 1.20)) if sig_ctx else m["current_price"] * 1.20
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            tc_capital  = st.number_input("Capital (₹)", value=500000, step=50000, min_value=10000, key="tc_cap")
            tc_risk_pct = st.slider("Risk per trade (%)", 0.5, 3.0, 1.5, 0.25, key="tc_risk",
                help="% of capital you risk if SL is hit. Professional range: 1-2%.")
        with tc2:
            tc_entry = st.number_input("Entry ₹", value=round(default_entry, 2), step=1.0, key="tc_entry")
            tc_sl    = st.number_input("Stop Loss ₹", value=round(default_sl, 2), step=1.0, key="tc_sl")
        with tc3:
            tc_t1 = st.number_input("Target 1 ₹", value=round(default_t1, 2), step=1.0, key="tc_t1")
            tc_t2 = st.number_input("Target 2 ₹", value=round(default_t2, 2), step=1.0, key="tc_t2")
        if tc_entry > 0 and tc_sl > 0 and abs(tc_entry - tc_sl) > 0.01:
            risk_ps   = abs(tc_entry - tc_sl)
            max_risk  = tc_capital * tc_risk_pct / 100
            shares    = max(1, int(max_risk / risk_ps))
            pos_val   = shares * tc_entry
            pos_pct   = pos_val / tc_capital * 100
            rr1       = abs(tc_t1 - tc_entry) / risk_ps if tc_entry != tc_t1 else 0
            rr2       = abs(tc_t2 - tc_entry) / risk_ps if tc_entry != tc_t2 else 0
            tr1, tr2, tr3, tr4, tr5 = st.columns(5)
            tr1.metric("Shares",       f"{shares:,}")
            tr2.metric("Position",     f"₹{pos_val:,.0f}", f"{pos_pct:.1f}% of capital")
            tr3.metric("₹ Risk (SL)",  f"₹{max_risk:,.0f}", f"{tc_risk_pct}%")
            tr4.metric("R:R at T1",    f"1:{rr1:.1f}")
            tr5.metric("R:R at T2",    f"1:{rr2:.1f}")
            pnl_t1 = shares * (tc_t1 - tc_entry)
            pnl_t2 = shares * (tc_t2 - tc_entry)
            pnl_sl = shares * (tc_sl - tc_entry)
            st.markdown(
                f"**P&L:** &nbsp; 🟢 T1: **+₹{pnl_t1:,.0f}** "
                f"({pnl_t1/tc_capital*100:.1f}%) &nbsp;·&nbsp; "
                f"🟢 T2: **+₹{pnl_t2:,.0f}** ({pnl_t2/tc_capital*100:.1f}%) "
                f"&nbsp;·&nbsp; 🔴 SL: **₹{pnl_sl:,.0f}** ({pnl_sl/tc_capital*100:.1f}%)"
            )
            if rr1 < 1.5: st.warning("⚠️ R:R below 1.5 — consider adjusting targets or skipping.")
            if pos_pct > 20: st.warning(f"⚠️ Position {pos_pct:.0f}% of capital — reduce risk % for safety.")

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE 3: ONE-CLICK PAPER TRADE
    # ══════════════════════════════════════════════════════════════════════
    if sig_ctx:
        st.divider()
        st.markdown("### 🎮 Paper Trade")
        _c_pt1, _c_pt2 = st.columns([3, 1])
        with _c_pt1:
            _vc = {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#ff9800"}.get(rec["final_recommendation"], "#888")
            st.markdown(
                f"AI verdict: **<span style='color:{_vc}'>{rec['final_recommendation']}</span>** "
                f"(score {rec['weighted_score']:+.3f}). Virtual ₹50,000 — no real money.",
                unsafe_allow_html=True
            )
        with _c_pt2:
            if st.button("🎮 Enter Paper Trade", type="primary", key="dd_paper_trade", use_container_width=True):
                try:
                    from paper_trading import enter_paper_trade
                    result = enter_paper_trade(sig_ctx)
                    if result:
                        st.success("✅ Paper trade entered!")
                        st.balloons()
                    else:
                        st.warning("Could not open paper trade. Check Virtual Game page.")
                except Exception as _pe:
                    st.error(f"Paper trade error: {_pe}")

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE 4: NEWS & CATALYST CONTEXT
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    with st.expander("📰 News & Catalyst Context (Gemini + Google Search)", expanded=False):
        st.caption("Live AI: why is this stock moving? Gemini with Google Search grounding.")
        if st.button("🔍 Fetch News Context", key="dd_news_btn"):
            try:
                from perplexity_enrichment import get_signal_context
                _sig_dir  = rec["final_recommendation"] if rec["final_recommendation"] != "HOLD" else "BUY"
                _sig_strat = sig_ctx.get("strategy", "Technical") if sig_ctx else "Technical"
                with st.spinner(f"Analysing {ticker} via Gemini + Google Search..."):
                    ctx = get_signal_context(ticker, _sig_strat, _sig_dir)
                if ctx.get("error"):
                    st.warning(f"News fetch: {ctx['error']}")
                    st.info("Add GEMINI_API_KEY to Streamlit Secrets for live news.")
                else:
                    _cat   = ctx.get("catalyst", "No specific catalyst identified")
                    _ctype = ctx.get("catalyst_type", "Unknown")
                    _csent = ctx.get("sentiment", "NEUTRAL")
                    _conf  = ctx.get("confidence", 0)
                    _live  = ctx.get("is_live", False)
                    _facts = ctx.get("factors", [])
                    _sc = {"BULLISH": "#26a69a", "BEARISH": "#ef5350", "NEUTRAL": "#ff9800"}.get(_csent, "#888")
                    st.markdown(
                        f'<div style="background:#1a1d23;border:1px solid #333;border-radius:10px;padding:16px">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
                        f'<b>{_ctype}</b>'
                        f'<span style="color:{_sc};font-weight:700">{_csent} · {_conf}% conf</span></div>'
                        f'<div style="color:#ccc">{_cat}</div>'
                        f'{"<div style=color:#26a69a;font-size:.75rem;margin-top:6px>🔴 LIVE — Google Search</div>" if _live else "<div style=color:#888;font-size:.75rem;margin-top:6px>📚 Training data</div>"}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    for _f in _facts: st.markdown(f"- {_f}")
            except ImportError:
                st.info("Add GEMINI_API_KEY to Streamlit Secrets for news analysis.")
            except Exception as _ne:
                st.error(f"News context error: {_ne}")

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE 5: SHARE BUTTONS
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    _sh1, _sh2, _sh3 = st.columns(3)

    with _sh1:
        if st.button("📱 Send to My Telegram", key="dd_tg_share", use_container_width=True):
            try:
                from auth_manager import get_current_profile
                _prof = get_current_profile() or {}
                _tgt  = _prof.get("telegram_bot_token", "")
                _tgc  = _prof.get("telegram_chat_id", "")
                if not _tgt:
                    try: _tgt = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
                    except: pass
                if not _tgc:
                    try: _tgc = st.secrets.get("TELEGRAM_CHAT_ID", "")
                    except: pass
                if not _tgt or not _tgc:
                    st.warning("Configure Telegram in Settings → Alert Preferences.")
                else:
                    _ve = "BUY" if rec["final_recommendation"]=="BUY" else ("SELL" if rec["final_recommendation"]=="SELL" else "HOLD")
                    _vmoji = {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(_ve,"⚪")
                    _tg_lines = [
                        "<b>AI Deep Dive: " + ticker + "</b>",
                        _vmoji + " " + _ve + " — Score: " + f"{rec['weighted_score']:+.3f}",
                        "Price: " + _fp(m["current_price"]) + " | 1D: " + f"{m['price_change_1d']:+.2f}%",
                        "RSI: " + f"{m['rsi_14']:.1f}" + " | ADX: " + f"{m['adx_14']:.1f}" + " | Vol: " + f"{m['volume_ratio']:.2f}x",
                        "Regime: " + regime["regime"] + " | VIX: " + str(regime["vix"]),
                        "Sentiment: " + sent["overall"],
                    ]
                    if sig_ctx:
                        _tg_lines += [
                            "Strategy: " + sig_ctx.get("strategy","") + " " + sig_ctx.get("signal",""),
                            "Entry " + _fp(sig_ctx.get("entry","?")) + " | SL " + _fp(sig_ctx.get("sl","?")) + " | T1 " + _fp(sig_ctx.get("t1","?")),
                            "SQI " + str(sig_ctx.get("sqi","?")) + " (" + sig_ctx.get("sqi_grade","") + ")",
                        ]
                    _tg_lines.append("<i>NSE Scanner Pro</i>")
                    _tg_msg = "\n".join(_tg_lines)
                    import requests as _r
                    _rr = _r.post(f"https://api.telegram.org/bot{_tgt}/sendMessage",
                                  json={"chat_id": _tgc, "text": _tg_msg, "parse_mode": "HTML"}, timeout=10)
                    if _rr.status_code == 200: st.success("✅ Sent to Telegram!")
                    else: st.error(f"Failed: {_rr.json().get('description','Unknown')}")
            except Exception as _te: st.error(f"Telegram error: {_te}")

    with _sh2:
        if st.button("📋 Copy Summary", key="dd_copy", use_container_width=True):
            _sumlines = [
                f"NSE Scanner Pro — AI Deep Dive: {ticker}",
                f"Signal: {rec['final_recommendation']} (Score: {rec['weighted_score']:+.3f})",
                f"Price: {_fp(m['current_price'])} | RSI: {m['rsi_14']:.1f} | ADX: {m['adx_14']:.1f}",
                f"Regime: {regime['regime']} | VIX: {regime['vix']}",
                f"Sentiment: {sent['overall']} (score: {sent['score']})",
            ]
            if sig_ctx:
                _sumlines += [
                    f"Strategy: {sig_ctx.get('strategy','')} | Entry: {_fp(sig_ctx.get('entry','?'))} | SL: {_fp(sig_ctx.get('sl','?'))} | T1: {_fp(sig_ctx.get('t1','?'))}",
                    f"R:R: {sig_ctx.get('rr','?')} | SQI: {sig_ctx.get('sqi','?')} ({sig_ctx.get('sqi_grade','')})",
                ]
            st.text_area("Copy this:", "\n".join(_sumlines), height=160, key="dd_summary_text")

    with _sh3:
        if st.button("⭐ Add to Watchlist", key="dd_watchlist", use_container_width=True):
            try:
                from auth_manager import get_current_user
                from signal_tracker import _get_supabase
                _wu = get_current_user()
                if _wu:
                    _wsb = _get_supabase()
                    if _wsb:
                        _wsb.table("user_watchlists").upsert({
                            "user_id": _wu["id"], "symbol": ticker,
                            "strategy": sig_ctx.get("strategy","Manual") if sig_ctx else "Manual",
                            "notes": f"AI Deep Dive — {rec['final_recommendation']} @ {_fp(m['current_price'])}",
                        }, on_conflict="user_id,symbol,strategy").execute()
                        st.success(f"⭐ {ticker} added to your watchlist!")
                    else:
                        st.session_state.setdefault("watchlist",[]).append({"symbol":ticker,"strategy":"AI Deep Dive","cmp":m["current_price"],"entry":m["current_price"],"stop":m["current_price"]*0.95,"target1":m["current_price"]*1.10,"target2":m["current_price"]*1.20,"confidence":80,"date":str(__import__("datetime").date.today()),"entry_type":"AT CMP","regime":regime.get("code","?"),"regime_fit":"OK"})
                        st.success(f"⭐ {ticker} added!")
                else:
                    st.session_state.setdefault("watchlist",[]).append({"symbol":ticker,"strategy":"AI Deep Dive","cmp":m["current_price"],"entry":m["current_price"],"stop":m["current_price"]*0.95,"target1":m["current_price"]*1.10,"target2":m["current_price"]*1.20,"confidence":80,"date":str(__import__("datetime").date.today()),"entry_type":"AT CMP","regime":regime.get("code","?"),"regime_fit":"OK"})
                    st.success(f"⭐ {ticker} added (sign in to persist)!")
            except Exception as _we: st.error(f"Watchlist error: {_we}")

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE 6: SECTOR RRG CONTEXT
    # ══════════════════════════════════════════════════════════════════════
    _rrg = st.session_state.get("rrg_data", {})
    _sector = (sig_ctx.get("sector","") if sig_ctx else "") or ""
    if not _sector:
        try:
            from stock_universe import get_sector
            _sector = get_sector(ticker)
        except Exception: pass
    if _sector and _rrg:
        _rrg_s = _rrg.get(_sector, {})
        if _rrg_s:
            _quad = _rrg_s.get("quadrant","UNKNOWN")
            _qmap = {
                "LEADING":   ("🟢","Strong + Improving — best sector for longs (+8 confidence)","#26a69a"),
                "WEAKENING": ("🟡","Strong but slowing — trade carefully","#ffd700"),
                "IMPROVING": ("🔵","Weak but accelerating — early rotation opportunity","#42a5f5"),
                "LAGGING":   ("🔴","Weak + Declining — avoid longs, consider shorts (-15 confidence)","#ef5350"),
            }
            _qe, _qdesc, _qcolor = _qmap.get(_quad, ("⚪","Unknown","#888"))
            st.info(f"**Sector: {_sector}** — {_qe} **{_quad}**: {_qdesc}")
    elif _sector:
        st.caption(f"Sector: **{_sector}** | Load data from Dashboard for RRG sector rotation context.")


