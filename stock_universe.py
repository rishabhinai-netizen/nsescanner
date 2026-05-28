"""
NSE Stock Universe — Dynamic Nifty 500 Fetcher with Hardcoded Fallback
"""

import requests
import csv
import io
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
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DYNAMIC FETCH FROM NSE (tries first)
# ============================================================================

# Module-level cache — HTTP call happens at most once per process lifetime
_nse_cache = {"symbols": None, "sector_map": None, "fetched": False}

def fetch_nifty500_from_nse():
    """Fetch latest Nifty 500 constituents from NSE Indices website.
    Results are cached in _nse_cache so HTTP is called at most once per session.
    """
    global _nse_cache
    if _nse_cache["fetched"]:
        return _nse_cache["symbols"], _nse_cache["sector_map"]
    try:
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            reader = csv.DictReader(io.StringIO(r.text))
            symbols = []
            sector_map = {}
            for row in reader:
                sym = row.get("Symbol", "").strip()
                industry = row.get("Industry", "").strip()
                if sym:
                    symbols.append(sym)
                    sector_map[sym] = industry
            if len(symbols) > 400:
                _nse_cache["symbols"] = symbols
                _nse_cache["sector_map"] = sector_map
                _nse_cache["fetched"] = True
                return symbols, sector_map
    except Exception as e:
        logger.warning(f"Failed to fetch Nifty 500 from NSE: {e}")
    _nse_cache["fetched"] = True  # Don't retry on failure
    return None, None


# ============================================================================
# HARDCODED NIFTY 200 FALLBACK (always available)
# ============================================================================

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NTPC",
    "WIPRO", "NESTLEIND", "BAJAJFINSV", "POWERGRID", "M&M",
    "ONGC", "JSWSTEEL", "ADANIPORTS", "TATASTEEL", "COALINDIA",
    "HDFCLIFE", "SBILIFE", "TECHM", "GRASIM", "DRREDDY",
    "CIPLA", "BPCL", "APOLLOHOSP", "DIVISLAB", "EICHERMOT",
    "HEROMOTOCO", "INDUSINDBK", "TATACONSUM", "BRITANNIA",
    "BAJAJ-AUTO", "HINDALCO", "ADANIENT", "SHRIRAMFIN", "LTIM", "TRENT",
]

NIFTY_NEXT_50 = [
    "ABB", "ACC", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM",
    "ATGL", "AUROPHARMA", "BANKBARODA", "BEL", "BERGEPAINT",
    "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "CONCOR",
    "DABUR", "DLF", "DMART", "GAIL", "GODREJCP",
    "HAL", "HAVELLS", "ICICIPRULI", "IDEA", "INDIGO",
    "IOC", "IRCTC", "IRFC", "JIOFIN", "JINDALSTEL",
    "LICI", "LODHA", "LUPIN", "MARICO", "MOTHERSON",
    "NAUKRI", "NHPC", "OBEROIRLTY", "PIRAMALPHARM", "PERSISTENT",
    "PIDILITIND", "PIIND", "PNB", "POLYCAB", "RECLTD",
    "SBICARD", "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM",
]

NIFTY_MIDCAP_100_PARTIAL = [
    "ABCAPITAL", "ALKEM", "ASHOKLEY", "ASTRAL", "ATUL",
    "AUBANK", "BALKRISIND", "BANDHANBNK", "BATAINDIA", "BHEL",
    "BIOCON", "BSE", "CANFINHOME", "CGPOWER", "CHAMBLFERT",
    "CUMMINSIND", "DEEPAKNTR", "DIXON", "ESCORTS", "EXIDEIND",
    "FEDERALBNK", "FORTIS", "GLENMARK", "GMRAIRPORT", "GNFC",
    "GODREJPROP", "GSPL", "HDFCAMC", "HINDPETRO", "IDFCFIRSTB",
    "IEX", "INDHOTEL", "INDUSTOWER", "IREDA", "JKCEMENT",
    "JSL", "JUBLFOOD", "KANSAINER", "KEI", "LAURUSLABS",
    "LICHSGFIN", "LINDEINDIA", "LTTS", "M&MFIN", "MANAPPURAM",
    "UNITDSPR", "METROPOLIS", "MFSL", "MSUMI", "MUTHOOTFIN",
    "NAM-INDIA", "NATIONALUM", "NAVINFLUOR", "NIACL", "NMDC",
    "OFSS", "PAGEIND", "PATANJALI", "PETRONET", "PHOENIXLTD",
    "PFC", "PRESTIGE", "PVRINOX", "RAJESHEXPO", "RAMCOCEM",
    "RVNL", "SAIL", "SCHAEFFLER", "SJVN", "SONACOMS",
    "STARHEALTH", "SUNDARMFIN", "SUPREMEIND", "SYNGENE", "TATACHEM",
    "TATACOMM", "TATAELXSI", "TORNTPOWER", "TVSMOTOR", "UBL",
    "UNIONBANK", "UNITDSPR", "UPL", "VEDL", "VOLTAS",
    "YESBANK", "ZOMATO", "ZYDUSLIFE",
]

# Combined fallback: ~200 stocks
HARDCODED_NIFTY_200 = list(set(NIFTY_50 + NIFTY_NEXT_50 + NIFTY_MIDCAP_100_PARTIAL))

# Sector map for hardcoded stocks
HARDCODED_SECTOR_MAP = {
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LTIM": "IT", "PERSISTENT": "IT", "NAUKRI": "IT", "TATAELXSI": "IT",
    "LTTS": "IT", "OFSS": "IT", "MPHASIS": "IT", "COFORGE": "IT",
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "INDUSINDBK": "Banking", "BANKBARODA": "Banking", "CANBK": "Banking",
    "PNB": "Banking", "UNIONBANK": "Banking", "IDFCFIRSTB": "Banking", "FEDERALBNK": "Banking",
    "AUBANK": "Banking", "BANDHANBNK": "Banking", "YESBANK": "Banking", "IDEA": "Telecom",
    # Finance
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "HDFCLIFE": "Finance", "SBILIFE": "Finance",
    "CHOLAFIN": "Finance", "ICICIPRULI": "Finance", "SHRIRAMFIN": "Finance", "SBICARD": "Finance",
    "JIOFIN": "Finance", "RECLTD": "Finance", "PIRAMALPHARM": "Finance", "LICI": "Finance",
    "HDFCAMC": "Finance", "MFSL": "Finance", "MUTHOOTFIN": "Finance", "MANAPPURAM": "Finance",
    "PFC": "Finance", "IRFC": "Finance", "ABCAPITAL": "Finance", "M&MFIN": "Finance",
    "LICHSGFIN": "Finance", "CANFINHOME": "Finance", "STARHEALTH": "Finance", "IREDA": "Finance",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "LUPIN": "Pharma", "APOLLOHOSP": "Pharma", "TORNTPHARM": "Pharma", "ZYDUSLIFE": "Pharma",
    "AUROPHARMA": "Pharma", "BIOCON": "Pharma", "ALKEM": "Pharma", "GLENMARK": "Pharma",
    "LAURUSLABS": "Pharma", "SYNGENE": "Pharma", "FORTIS": "Pharma", "METROPOLIS": "Pharma",
    # Auto
    "MARUTI": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto", "MOTHERSON": "Auto", "TVSMOTOR": "Auto",
    "ASHOKLEY": "Auto", "ESCORTS": "Auto", "EXIDEIND": "Auto", "BALKRISIND": "Auto",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "TATACONSUM": "FMCG",
    "BRITANNIA": "FMCG", "COLPAL": "FMCG", "DABUR": "FMCG", "GODREJCP": "FMCG",
    "MARICO": "FMCG", "UNITDSPR": "FMCG", "UBL": "FMCG", "UNITDSPR": "FMCG",
    "PATANJALI": "FMCG",
    # Metals
    "JSWSTEEL": "Metals", "TATASTEEL": "Metals", "HINDALCO": "Metals", "VEDL": "Metals",
    "JINDALSTEL": "Metals", "SAIL": "Metals", "NATIONALUM": "Metals", "NMDC": "Metals",
    "JSL": "Metals",
    # Energy/Power
    "RELIANCE": "Energy", "NTPC": "Energy", "POWERGRID": "Energy", "ONGC": "Energy",
    "BPCL": "Energy", "COALINDIA": "Energy", "GAIL": "Energy", "IOC": "Energy",
    "TATAPOWER": "Energy", "ADANIGREEN": "Energy", "NHPC": "Energy", "ATGL": "Energy",
    "ADANIPOWER": "Energy", "PETRONET": "Energy", "HINDPETRO": "Energy", "SJVN": "Energy",
    "TORNTPOWER": "Energy",
    # Infra/Capital Goods
    "LT": "Infra", "ADANIPORTS": "Infra", "ADANIENT": "Infra", "ULTRACEMCO": "Cement",
    "GRASIM": "Cement", "AMBUJACEM": "Cement", "ACC": "Cement", "RAMCOCEM": "Cement",
    "JKCEMENT": "Cement", "DLF": "Realty", "LODHA": "Realty", "OBEROIRLTY": "Realty",
    "GODREJPROP": "Realty", "PRESTIGE": "Realty", "PHOENIXLTD": "Realty",
    "IRCTC": "Infra", "INDIGO": "Aviation", "SIEMENS": "Capital Goods", "ABB": "Capital Goods",
    "HAL": "Defence", "BEL": "Defence", "BHEL": "Capital Goods", "CGPOWER": "Capital Goods",
    "CUMMINSIND": "Capital Goods", "RVNL": "Infra", "GMRAIRPORT": "Infra", "CONCOR": "Logistics",
    # Consumer/Retail
    "TITAN": "Consumer", "ASIANPAINT": "Consumer", "BERGEPAINT": "Consumer", "HAVELLS": "Consumer",
    "PIDILITIND": "Consumer", "VOLTAS": "Consumer", "POLYCAB": "Consumer", "SRF": "Chemicals",
    "TRENT": "Retail", "DMART": "Retail", "ZOMATO": "Consumer Tech", "BOSCHLTD": "Auto Ancillary",
    "BATAINDIA": "Consumer", "PAGEIND": "Consumer", "JUBLFOOD": "Consumer",
    "PVRINOX": "Consumer", "INDHOTEL": "Hotels",
    # Chemicals
    "PIIND": "Chemicals", "DEEPAKNTR": "Chemicals", "ATUL": "Chemicals",
    "CHAMBLFERT": "Chemicals", "GNFC": "Chemicals", "NAVINFLUOR": "Chemicals",
    # Others
    "DIXON": "Electronics", "ASTRAL": "Building Materials", "KEI": "Cables",
    "KANSAINER": "Packaging", "SUNDARMFIN": "Finance", "SUPREMEIND": "Building Materials",
    "TATACOMM": "Telecom", "TATACHEM": "Chemicals", "SCHAEFFLER": "Auto Ancillary",
    "LINDEINDIA": "Industrial Gas", "SONACOMS": "Auto Ancillary", "IEX": "Exchange",
    "BSE": "Exchange", "INDUSTOWER": "Telecom", "BHARTIARTL": "Telecom",
    "UPL": "Agrochemicals", "GSPL": "Energy", "RAJESHEXPO": "FMCG",
    "NIACL": "Insurance", "NAM-INDIA": "Finance",
}


def get_stock_universe(size: str = "nifty200") -> list:
    """
    Get stock universe. Tries dynamic NSE fetch first, falls back to hardcoded.
    size: 'nifty50', 'nifty100', 'nifty200', 'nifty500'
    """
    if size == "nifty500":
        symbols, _ = fetch_nifty500_from_nse()
        if symbols:
            return symbols
        return HARDCODED_NIFTY_200  # fallback
    elif size == "nifty200":
        return HARDCODED_NIFTY_200
    elif size == "nifty100":
        return list(set(NIFTY_50 + NIFTY_NEXT_50))
    else:
        return NIFTY_50


def get_sector(symbol: str) -> str:
    """Get sector for a symbol. Uses cached NSE data — no repeated HTTP calls."""
    _, dynamic_sectors = fetch_nifty500_from_nse()  # returns cached after first call
    if dynamic_sectors and symbol in dynamic_sectors:
        return dynamic_sectors[symbol]
    return HARDCODED_SECTOR_MAP.get(symbol, "Other")
