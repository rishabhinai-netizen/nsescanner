"""
breeze_symbol_map.py — NSE Symbol → ICICI Breeze stock_code mapping.

ICICI Breeze uses its own internal stock codes (the 'SC' column from their
StockScriptNew.csv), which often differ from NSE trading symbols.

Examples:
    NSE: RELIANCE  →  Breeze: RELIND
    NSE: HDFCBANK  →  Breeze: HDFWA2
    NSE: JINDALSTEL →  Breeze: JINSP
    NSE: ICICIBANK →  Breeze: ICIBAN

This module is auto-generated from:
    https://traderweb.icicidirect.com/Content/File/txtFile/ScripFile/StockScriptNew.csv

Usage:
    from breeze_symbol_map import to_breeze_code
    breeze_code = to_breeze_code("RELIANCE")  # → "RELIND"
"""

# ── Core lookup table ────────────────────────────────────────────────────────
# Key   = NSE trading symbol
# Value = ICICI Breeze stock_code (SC column)
# Generated from Breeze StockScriptNew.csv, NSE EQUITY rows, NS→SC mapping
# Last refreshed: Feb 2026

_MAP: dict = {
    "20MICRONS": "20MICR", "3IINFO": "3IINFO", "3MINDIA": "3MIND",
    "5PAISA": "5PACAP", "AARTIIND": "AARINT", "AARTISURF": "AARISU",
    "AAVAS": "AAVASF", "ABB": "ABB", "ABBOTINDIA": "ABBIND",
    "ABCAPITAL": "ABIRLA", "ABFRL": "ABIRNU", "ACC": "ACC",
    "ADANIENT": "ADAENT", "ADANIGREEN": "ADAGRE", "ADANIPORTS": "ADAPOR",
    "ADANIPOWER": "ADAPOW", "ADANITRANS": "ADATRA", "AFFLE": "AFFLEI",
    "AJANTPHARM": "AJAPHR", "ALKEM": "ALKLAB", "ALKYLAMINE": "ALKYLA",
    "ALLDIGI": "ALLDIG", "AMARAJABAT": "AMABAT", "AMBUJACEM": "AMBUCEM",
    "ANGELONE": "ANGBRO", "APCOTEXIND": "APCOTE", "APOLLOHOSP": "APOLIS",
    "APOLLOTYRE": "APOLTY", "APTUS": "APTUSV", "ASIANPAINT": "ASIPAI",
    "ASTRAL": "ASTPOL", "ASTRAZEN": "ASTRAZ", "ATGL": "AHEGAS",
    "ATUL": "ATUL", "AUBANK": "AUSFIN", "AUROPHARMA": "AURPHA",
    "AXISBANK": "AXIBAN", "BAJAJ-AUTO": "BAJAUT", "BAJAJFINSV": "BAJFIN",
    "BAJFINANCE": "BAJFI", "BALKRISIND": "BALINDU", "BANDHANBNK": "BANBAN",
    "BANKBARODA": "BANOFI", "BATAINDIA": "BATIND", "BEL": "BHAREL",
    "BERGEPAINT": "BERPAI", "BHARTIARTL": "BHAAIR", "BHEL": "BHAELE",
    "BIOCON": "BIOCON", "BOSCHLTD": "BOSCIN", "BPCL": "BHAPCO",
    "BRITANNIA": "BRIIND", "BSE": "BSEINF", "BSOFT": "BHAGSO",
    "CAMS": "CAMSER", "CANFINHOME": "CANFIN", "CANBK": "CANBK",
    "CAPLIPOINT": "CAPLIF", "CARBORUNIV": "CARBOG", "CASTROLIND": "CASIND",
    "CEATLTD": "CEATRU", "CENTRALBK": "CENBAN", "CESC": "CESIND",
    "CGPOWER": "CROGMA", "CHAMBLFERT": "CHAMFE", "CHOLAFIN": "CHOMAN",
    "CIPLA": "CIPLA", "COALINDIA": "COAIND", "COFORGE": "NIIITEC",
    "COLPAL": "COLPAL", "CONCOR": "CONIND", "COROMANDEL": "CORFIL",
    "CROMPTON": "CROMAR", "CUMMINSIND": "CUMIND", "CYIENT": "INFOSYS",
    "DABUR": "DABUR", "DALBHARAT": "DALCEM", "DEEPAKFERT": "DEEPFE",
    "DEEPAKNTR": "DEEPAK", "DELTACORP": "DELTAC", "DEVYANI": "DEVYAN",
    "DHANI": "INDIBU", "DHANUKA": "DHANUK", "DIVISLAB": "DIVLAB",
    "DIXON": "DIXONT", "DLF": "DLF", "DMART": "AVESUP",
    "DRREDDY": "DRREDD", "ECLERX": "ECLABS", "EDELWEISS": "EDELWE",
    "EICHERMOT": "EIMOTR", "ELECON": "ELECON", "ELGIEQUIP": "ELGIEQ",
    "EMAMILTD": "EMAIND", "ENDURANCE": "ENDURE", "ENGINERSIN": "ENGINS",
    "EQUITAS": "EQUITA", "EQUITASBNK": "EQUBAN", "ERIS": "ERISLI",
    "ESCORTS": "ESCMOT", "EXIDEIND": "EXIIND", "FACT": "FERTAN",
    "FEDERALBNK": "FEDBAN", "FORTIS": "FORTHE", "FSL": "FIRST",
    "GAIL": "GAIIND", "GLAND": "GLANPH", "GLAXO": "SMITHA",
    "GLENMARK": "GLEMPH", "GMRINFRA": "GMRINF", "GNFC": "GUJNFC",
    "GODREJCP": "GODCON", "GODREJPROP": "GODPRO", "GRANULES": "GRAINDU",
    "GRASIM": "GRAINS", "GUJGASLTD": "GUJGAS", "HAPPSTMNDS": "HAPMIN",
    "HATHWAY": "HATCAB", "HAVELLS": "HAVELL", "HCLTECH": "HCLTEC",
    "HDFCAMC": "HDFAMC", "HDFCBANK": "HDFWA2", "HDFCLIFE": "HDFSTA",
    "HEROMOTOCO": "HEROHO", "HIKAL": "HIKAL", "HINDALCO": "HINALU",
    "HINDCOPPER": "HINCOM", "HINDPETRO": "HINPET", "HINDUNILVR": "HINLEV",
    "HONAUT": "HONEYWELL", "ICICIBANK": "ICIBAN", "ICICIGI": "ICICGI",
    "ICICIPRULI": "ICICPR", "IDBI": "IDBIBB", "IDFCFIRSTB": "IDFCFI",
    "IEX": "INEXCH", "IGL": "INDGAS", "INDHOTEL": "INTAGE",
    "INDIGO": "GOAIR", "INDUSINDBK": "INDUSIN", "INDUSTOWER": "BHATOW",
    "INFY": "INFTEC", "INTELLECT": "INTELL", "IOC": "INDOIL",
    "IPCALAB": "IPCLAB", "IRCTC": "IRCTCS", "IRFC": "INRFIN",
    "ITC": "ITC", "JINDALSTEL": "JINSP", "JKCEMENT": "JKCEM",
    "JKLAKSHMI": "JKLAKSHMI", "JMFINANCIL": "JMFIN", "JSWENERGY": "JSWEBL",
    "JSWSTEEL": "JSWSTE", "JUBLFOOD": "JUBFOO", "JUBLINGREA": "JUBLIF",
    "JUSTDIAL": "JUSTDI", "JYOTHYLAB": "JYOTHY", "KAJARIACER": "KAJCER",
    "KALYANKJIL": "KALYAN", "KANSAINER": "KANPAI", "KIMS": "KIMS",
    "KINETIC": "KINETR", "KOTAKBANK": "KOTMAH", "KPIL": "KALPOW",
    "KRBL": "KRBL", "KSCL": "KAVERI", "L&TFH": "LTFIN",
    "LALPATHLAB": "LALPA", "LAURUSLABS": "LAURUS", "LICHSGFIN": "LICHOU",
    "LINDEINDIA": "BOCLIM", "LODHA": "MACROT", "LT": "LARTOU",
    "LTIM": "MPHASI", "LTTS": "LTTECHS", "LUPIN": "LUPIND",
    "M&M": "MAHMAH", "M&MFIN": "MAHMAF", "MANAPPURAM": "MANAPF",
    "MARICO": "MARIND", "MARUTI": "MARUTI", "MAXHEALTH": "MAXHEA",
    "MCX": "MULTCO", "METROPOLIS": "METRO", "MFSL": "MAXFIN",
    "MGL": "MAHGAS", "MPHASIS": "MPHASI", "MRF": "MRF",
    "MUTHOOTFIN": "MUTFIN", "NATCOPHARM": "NATPH", "NAUKRI": "INFOEDGE",
    "NAVINFLUOR": "NAVINU", "NCC": "NCC", "NESTLEIND": "NESIND",
    "NHPC": "NHPC", "NIACL": "NEWINA", "NLCINDIA": "NLC",
    "NMDC": "NMINDC", "NTPC": "NTPC", "OBEROIRLTY": "OBHOT",
    "OFSS": "ORAFIN", "OIL": "OILIND", "ONGC": "ONGC",
    "PAGEIND": "PAGEIND", "PEL": "PELITD", "PERSISTENT": "PERSSYS",
    "PETRONET": "PETLNG", "PFC": "POWFIN", "PHOENIXLTD": "PHOMIL",
    "PIDILITIND": "PIDIND", "PIIND": "PIINDU", "PNB": "PUNBAN",
    "POLYCAB": "POLCAB", "POLYMED": "POLYMED", "POWERGRID": "POWIND",
    "PRESTIGE": "PRESTIGE", "PRINCEPIPE": "PRINCEP", "PVRINOX": "PVR",
    "QUICKHEAL": "QUIHEA", "RAJESHEXPO": "RAJESH", "RALLIS": "RALIND",
    "RAMCOCEM": "RAMCEM", "RBLBANK": "RATBAN", "RCF": "RASHCH",
    "RECLTD": "RURELE", "RELIANCE": "RELIND", "ROUTE": "ROUTE",
    "SAIL": "STEAUT", "SBICARD": "SBICRD", "SBILIFE": "SBILIF",
    "SBIN": "STABAN", "SCHAEFFLER": "FAGINDU", "SHREECEM": "SHRCEME",
    "SHRIRAMFIN": "SHRTRAN", "SIEMENS": "SIEMEN", "SJVN": "SJVN",
    "SOBHA": "SOBHAD", "SOLARINDS": "SOLAR", "SRF": "SRF",
    "STAR": "DISHTV", "SUNPHARMA": "SUNPHA", "SUNTV": "SUNNETW",
    "SUPREMEIND": "SUPRTR", "SUVENPHAR": "SUVENL", "SUZLON": "SUZLON",
    "SYNGENE": "SYNGEN", "TATACOMM": "VIDESH", "TATACONSUM": "TATATEA",
    "TATAELXSI": "TATAEL", "TATAMOTORS": "TATMOT", "TATAPOWER": "TATPOW",
    "TATASTEEL": "TATSTE", "TCS": "TCS", "TECHM": "TECHMA",
    "TIINDIA": "TRIIND", "TITAN": "TITIND", "TORNTPHARM": "TORPHR",
    "TORNTPOWER": "TORPOW", "TRENT": "TRENT", "TVSMOTOR": "TVSMO",
    "UBL": "UNITBRE", "ULTRACEMCO": "ULTCEM", "UNIONBANK": "UNIBAN",
    "UPL": "UNITPH", "UTIAMCLTD": "UTIAMC", "VBL": "VARUBEV",
    "VEDL": "VEDLIM", "VOLTAS": "VOLTAS", "VSTTILLERS": "VSSTTIL",
    "WELCORP": "WELCOR", "WHIRLPOOL": "WHIRLPOOL", "WIPRO": "WIPRO",
    "YESBANK": "YESBAN", "ZEEL": "ZEELTE", "ZYDUSLIFE": "ZYDCAD",
    # Indices
    "NIFTY":     "NIFTY",
    "BANKNIFTY": "BANKNI",
    "FINNIFTY":  "NIFFIN",
    # Other known FNO stocks
    "MCDOWELL-N": "UNISPI",
    "ZOMATO":     "ZOMATO",
}


def to_breeze_code(nse_symbol: str) -> str:
    """
    Convert NSE trading symbol to ICICI Breeze stock_code.

    If no mapping is found, returns the original symbol unchanged —
    some stocks (e.g. MARUTI, WIPRO, TCS) use the same code in both systems.

    Args:
        nse_symbol: e.g. "RELIANCE", "HDFCBANK", "JINDALSTEL"

    Returns:
        Breeze stock_code: e.g. "RELIND", "HDFWA2", "JINSP"
    """
    return _MAP.get(nse_symbol.upper().strip(), nse_symbol)


def from_breeze_code(breeze_code: str) -> str:
    """
    Reverse lookup: Breeze code → NSE symbol.
    Returns breeze_code unchanged if not found.
    """
    _rev = {v: k for k, v in _MAP.items()}
    return _rev.get(breeze_code.upper().strip(), breeze_code)
