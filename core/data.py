"""Data engine — yfinance (daily) + Breeze (intraday, optional).
Keeps the two hard-won fixes from v15: yfinance MultiIndex handling and
the 20s thread-timeout around every Breeze SDK call."""
import logging, threading
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytz
import yfinance as yf

from core.config import IST, BREEZE_API_KEY, BREEZE_API_SECRET

logger = logging.getLogger("nx.data")
BREEZE_TIMEOUT = 20  # hard ceiling — SDK has no native timeout (v11 spinner bug)


def now_ist() -> datetime:
    return datetime.now(pytz.timezone(IST))


def is_market_hours() -> bool:
    n = now_ist()
    return n.weekday() < 5 and (9, 15) <= (n.hour, n.minute) <= (15, 30)


# ---------------------------------------------------------------- yfinance --
def fetch_batch_daily(symbols: List[str], period: str = "1y",
                      batch_size: int = 40) -> Dict[str, pd.DataFrame]:
    """Batch daily OHLCV. Handles yfinance>=0.2.x MultiIndex columns."""
    out: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        try:
            raw = yf.download([f"{s}.NS" for s in chunk], period=period,
                              group_by="ticker", auto_adjust=True,
                              progress=False, threads=True)
        except Exception as e:
            logger.warning(f"yf batch failed ({chunk[:3]}...): {e}")
            continue
        for s in chunk:
            try:
                df = raw[f"{s}.NS"] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(how="all")
                if len(df) >= 60:
                    out[s] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            except Exception:
                continue
    return out


def fetch_nifty(period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker("^NSEI").history(period=period, auto_adjust=True)
        return df if len(df) else None
    except Exception as e:
        logger.warning(f"nifty fetch failed: {e}")
        return None


# ------------------------------------------------------------------ Breeze --
class Breeze:
    """Thin Breeze wrapper. Token lives in nx_app_config (refreshed daily
    from ICICI Direct portal via Settings page). Every call timeout-wrapped."""

    def __init__(self, session_token: str):
        from breeze_connect import BreezeConnect
        self.api = BreezeConnect(api_key=BREEZE_API_KEY())
        self.api.generate_session(api_secret=BREEZE_API_SECRET(),
                                  session_token=session_token)

    def _call(self, fn, *args, timeout: int = BREEZE_TIMEOUT, **kwargs):
        result, err = {}, {}
        def run():
            try:
                result["v"] = fn(*args, **kwargs)
            except Exception as e:
                err["e"] = e
        t = threading.Thread(target=run, daemon=True)
        t.start(); t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"Breeze call exceeded {timeout}s — refresh session token")
        if "e" in err:
            raise err["e"]
        return result.get("v")

    def quotes(self, stock_code: str):
        return self._call(self.api.get_quotes, stock_code=stock_code,
                          exchange_code="NSE", product_type="cash")
