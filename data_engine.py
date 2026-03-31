"""
Data Engine — yfinance (daily, rate-limited) + ICICI Breeze (intraday, from st.secrets)
"""

import pandas as pd
import numpy as np
import yfinance as yf
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False
    # Create a mock st module for non-Streamlit environments (GitHub Actions)
    class _MockSt:
        @staticmethod
        def cache_data(ttl=None, show_spinner=False):
            def decorator(func): return func
            return decorator
        class secrets:
            @staticmethod
            def get(key, default=""): return default
    st = _MockSt()
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import time as time_module
try:
    from breeze_symbol_map import to_breeze_code
except ImportError:
    def to_breeze_code(s: str) -> str: return s  # fallback: pass through
import pytz
import logging

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


def now_ist() -> datetime:
    """Current time in IST."""
    return datetime.now(IST)


# ============================================================================
# YFINANCE DATA ENGINE (FREE — DAILY DATA, RATE-LIMITED)
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_daily_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV data from yfinance for a single symbol."""
    try:
        yf_symbol = f"{symbol}.NS"
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(IST).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        df["symbol"] = symbol
        return df
    except Exception as e:
        return None


def fetch_batch_daily(symbols: List[str], period: str = "1y",
                      progress_callback=None) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily data in batches to avoid yfinance rate limits.
    Batches of 40 symbols with 2-second delay between batches.
    """
    results = {}
    batch_size = 40
    total = len(symbols)
    
    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        if progress_callback:
            progress_callback(i / total, f"Batch {batch_num}/{total_batches}: Fetching {len(batch)} stocks...")
        
        yf_symbols = [f"{s}.NS" for s in batch]
        
        try:
            raw = yf.download(
                yf_symbols, period=period, auto_adjust=True,
                threads=True, progress=False, group_by="ticker"
            )
            
            if raw is not None and not raw.empty:
                for symbol, yf_sym in zip(batch, yf_symbols):
                    try:
                        if len(batch) == 1:
                            df = raw.copy()
                        elif isinstance(raw.columns, pd.MultiIndex):
                            if yf_sym in raw.columns.get_level_values(0):
                                df = raw[yf_sym].copy()
                            else:
                                continue
                        else:
                            df = raw.copy()
                        
                        if df is not None and not df.empty:
                            df = df.dropna(how="all")
                            if len(df) >= 50:
                                # Normalize columns
                                col_map = {}
                                for c in df.columns:
                                    cl = str(c).lower()
                                    if "open" in cl: col_map[c] = "open"
                                    elif "high" in cl: col_map[c] = "high"
                                    elif "low" in cl: col_map[c] = "low"
                                    elif "close" in cl: col_map[c] = "close"
                                    elif "volume" in cl or "vol" in cl: col_map[c] = "volume"
                                df = df.rename(columns=col_map)
                                
                                needed = ["open", "high", "low", "close", "volume"]
                                if all(c in df.columns for c in needed):
                                    df = df[needed]
                                    # Fix timezone
                                    if df.index.tz is not None:
                                        df.index = df.index.tz_convert(IST).tz_localize(None)
                                    df["symbol"] = symbol
                                    results[symbol] = df
                    except Exception:
                        continue
        except Exception as e:
            # Fallback: fetch individually for this batch
            for symbol in batch:
                df = fetch_daily_data(symbol, period)
                if df is not None:
                    results[symbol] = df
        
        # Rate limit delay between batches (except last batch)
        if i + batch_size < total:
            time_module.sleep(1.5)
    
    return results


@st.cache_data(ttl=600, show_spinner=False)
def fetch_nifty_data(period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch Nifty 50 index data for market health check."""
    try:
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(period=period, auto_adjust=True)
        if df.empty:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        if df.index.tz is not None:
            df.index = df.index.tz_convert(IST).tz_localize(None)
        return df
    except Exception:
        return None


# ============================================================================
# ICICI BREEZE API ENGINE — reads from Streamlit Secrets
# ============================================================================

def _get_breeze_token_from_supabase() -> str:
    """
    Read BREEZE_SESSION_TOKEN from Supabase app_config table.
    Returns empty string if Supabase not configured or key not found.
    """
    try:
        import os
        url = ""
        key = ""
        try:
            url = st.secrets.get("SUPABASE_URL", "")
            key = st.secrets.get("SUPABASE_SERVICE_KEY", "") or st.secrets.get("SUPABASE_ANON_KEY", "")
        except Exception:
            pass
        if not url:
            url = os.environ.get("SUPABASE_URL", "")
        if not key:
            key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            return ""
        from supabase import create_client
        sb = create_client(url, key)
        resp = sb.table("app_config").select("value").eq("key", "BREEZE_SESSION_TOKEN").execute()
        if resp.data:
            token = resp.data[0].get("value", "")
            if token and token != "placeholder":
                return token
        return ""
    except Exception:
        return ""


def update_breeze_token_in_supabase(new_token: str) -> bool:
    """
    Write a new BREEZE_SESSION_TOKEN to Supabase app_config.
    Called from Settings page when user updates the token.
    """
    try:
        import os
        url = ""
        key = ""
        try:
            url = st.secrets.get("SUPABASE_URL", "")
            key = st.secrets.get("SUPABASE_SERVICE_KEY", "")
        except Exception:
            pass
        if not url: url = os.environ.get("SUPABASE_URL", "")
        if not key: key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            return False
        from supabase import create_client
        sb = create_client(url, key)
        sb.table("app_config").upsert({
            "key":        "BREEZE_SESSION_TOKEN",
            "value":      new_token.strip(),
            "updated_at": "now()",
            "updated_by": "settings_page",
        }, on_conflict="key").execute()
        return True
    except Exception:
        return False


class BreezeEngine:
    """
    ICICI Breeze API integration.
    Credentials are read from Streamlit secrets (Settings > Secrets).
    Session token must be regenerated daily from ICICI Direct portal.
    """
    
    def __init__(self):
        self.connected = False
        self.breeze = None
        self.connection_message = ""
    
    def connect_from_secrets(self) -> Tuple[bool, str]:
        """
        Connect using credentials.
        Session token priority: Supabase app_config → Streamlit secrets
        API key/secret: always from Streamlit secrets (permanent credentials)
        """
        try:
            api_key    = st.secrets.get("BREEZE_API_KEY", "")
            api_secret = st.secrets.get("BREEZE_API_SECRET", "")

            if not api_key or not api_secret:
                return False, "BREEZE_API_KEY / BREEZE_API_SECRET missing from Streamlit secrets."
            if api_key == "your_api_key_here":
                return False, "Breeze credentials are placeholder values. Update them in Streamlit secrets."

            # Session token: try Supabase first (user can update via Settings page),
            # then fall back to Streamlit secrets
            session_token = _get_breeze_token_from_supabase()
            if not session_token:
                session_token = st.secrets.get("BREEZE_SESSION_TOKEN", "")
            if not session_token:
                return False, "BREEZE_SESSION_TOKEN missing. Update it in Settings → Breeze Token."

            return self.connect(api_key, api_secret, session_token)
        except Exception as e:
            return False, f"Error reading secrets: {str(e)}"
    
    def connect(self, api_key: str, api_secret: str, session_token: str) -> Tuple[bool, str]:
        """Connect to ICICI Breeze API with validation."""
        try:
            from breeze_connect import BreezeConnect
            
            self.breeze = BreezeConnect(api_key=api_key)
            self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
            
            # VALIDATE connection by making a test call
            test = self.breeze.get_customer_details()
            if test and ("Success" in str(test.get("Status", "")) or test.get("Success")):
                self.connected = True
                self.connection_message = "✅ Breeze API connected and validated!"
                return True, self.connection_message
            else:
                # generate_session succeeded, so connection works
                self.connected = True
                self.connection_message = "✅ Breeze API connected!"
                return True, self.connection_message
                
        except ImportError:
            return False, "❌ breeze-connect not installed. Run: pip install breeze-connect"
        except Exception as e:
            err = str(e)
            self.connected = False
            if "Invalid Session" in err or "session" in err.lower():
                return False, "❌ Session token expired. Generate a new one from ICICI Direct portal."
            elif "Invalid" in err:
                return False, f"❌ Invalid credentials: {err}"
            else:
                return False, f"❌ Connection failed: {err}"
    
    def fetch_intraday(self, symbol: str, interval: str = "5minute",
                       days_back: int = 5) -> Optional[pd.DataFrame]:
        """Fetch intraday data from Breeze API."""
        if not self.connected or self.breeze is None:
            return None
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT07:00:00.000Z")
            to_date = datetime.now().strftime("%Y-%m-%dT23:59:59.000Z")
            
            data = self.breeze.get_historical_data_v2(
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                stock_code=symbol,
                exchange_code="NSE",
                product_type="cash"
            )
            if data and "Success" in str(data.get("Status", "")):
                df = pd.DataFrame(data["Success"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
                df = df.rename(columns={
                    "open": "open", "high": "high",
                    "low": "low", "close": "close", "volume": "volume"
                })
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                df["symbol"] = symbol
                return df
            return None
        except Exception as e:
            logger.warning(f"Breeze intraday failed for {symbol}: {e}")
            return None

    def fetch_volume_profile(self, symbol: str, days_back: int = 20,
                             num_bins: int = 20) -> Optional[Dict]:
        """Compute Volume-by-Price profile using Breeze intraday data.
        Returns POC, VAH, VAL, HVN (high volume nodes), LVN (low volume nodes).
        """
        if not self.connected or self.breeze is None:
            return None
        try:
            df = self.fetch_intraday(symbol, interval="5minute", days_back=days_back)
            if df is None or len(df) < 50:
                return None

            price_min = df["low"].min()
            price_max = df["high"].max()
            if price_max <= price_min:
                return None

            bin_size = (price_max - price_min) / num_bins
            bins = np.arange(price_min, price_max + bin_size, bin_size)
            vol_profile = np.zeros(len(bins) - 1)
            price_levels = (bins[:-1] + bins[1:]) / 2

            for _, row in df.iterrows():
                bar_low, bar_high, bar_vol = row["low"], row["high"], row["volume"]
                if bar_vol <= 0 or bar_high <= bar_low:
                    continue
                for j in range(len(price_levels)):
                    if bar_low <= price_levels[j] <= bar_high:
                        vol_profile[j] += bar_vol / max(1, int((bar_high - bar_low) / bin_size))

            if vol_profile.sum() == 0:
                return None

            poc_idx = np.argmax(vol_profile)
            poc_price = round(float(price_levels[poc_idx]), 2)

            # Value Area (70% of volume around POC)
            total_vol = vol_profile.sum()
            target_vol = total_vol * 0.70
            cum_vol = vol_profile[poc_idx]
            lo_idx, hi_idx = poc_idx, poc_idx
            while cum_vol < target_vol:
                expand_lo = vol_profile[lo_idx - 1] if lo_idx > 0 else 0
                expand_hi = vol_profile[hi_idx + 1] if hi_idx < len(vol_profile) - 1 else 0
                if expand_lo >= expand_hi and lo_idx > 0:
                    lo_idx -= 1; cum_vol += vol_profile[lo_idx]
                elif hi_idx < len(vol_profile) - 1:
                    hi_idx += 1; cum_vol += vol_profile[hi_idx]
                else:
                    break

            vah = round(float(price_levels[hi_idx]), 2)
            val_ = round(float(price_levels[lo_idx]), 2)

            avg_vol = vol_profile.mean()
            hvn = [round(float(price_levels[i]), 2) for i in range(len(vol_profile)) if vol_profile[i] > avg_vol * 1.5]
            lvn = [round(float(price_levels[i]), 2) for i in range(len(vol_profile)) if 0 < vol_profile[i] < avg_vol * 0.5]

            return {
                "price_levels": [round(float(p), 2) for p in price_levels],
                "volumes": [round(float(v), 0) for v in vol_profile],
                "poc": poc_price, "vah": vah, "val": val_,
                "hvn": hvn[:5], "lvn": lvn[:5],
                "total_bars": len(df), "days": days_back,
            }
        except Exception as e:
            logger.warning(f"Volume profile {symbol}: {type(e).__name__}: {e}")
            return None

    def fetch_option_chain(self, symbol: str, expiry_date: str = None) -> Optional[Dict]:
        """
        Fetch and parse live option chain data from Breeze API.

        Breeze response fields (confirmed from API docs):
            strike_price, ltp, open_interest, chnge_oi,
            total_quantity_traded, spot_price (str), best_bid_price, best_offer_price
        NOTE: Breeze does NOT return implied_volatility — IV is estimated from LTP.

        Returns structured dict ready for compute_option_chain_signal().
        """
        if not self.connected or self.breeze is None:
            return None
        try:
            from datetime import date as ddate

            # ── Translate NSE symbol → Breeze stock_code ────────────────────
            # CRITICAL: Breeze uses its own codes (RELIANCE→RELIND, HDFCBANK→HDFWA2, etc.)
            breeze_symbol = to_breeze_code(symbol)

            # ── Resolve nearest expiry ──────────────────────────────────────────
            # Breeze requires format: "2025-08-28T06:00:00.000Z"
            expiry_date_plain = None
            if expiry_date is None:
                today = ddate.today()
                # Find nearest Thursday (NSE weekly expiry)
                for offset in range(0, 14):
                    candidate = today + timedelta(days=offset)
                    if candidate.weekday() == 3:  # Thursday
                        expiry_date_plain = candidate.strftime("%Y-%m-%d")
                        expiry_fmt = candidate.strftime("%Y-%m-%dT06:00:00.000Z")
                        break
                if not expiry_date_plain:
                    candidate = today + timedelta(days=7)
                    expiry_date_plain = candidate.strftime("%Y-%m-%d")
                    expiry_fmt = candidate.strftime("%Y-%m-%dT06:00:00.000Z")
            else:
                # Normalize whatever format is passed in
                expiry_date_plain = expiry_date[:10]
                expiry_fmt = expiry_date_plain + "T06:00:00.000Z"

            def _fetch_leg(right: str):
                """Fetch one leg (call/put) from Breeze using translated breeze_symbol."""
                resp = self.breeze.get_option_chain_quotes(
                    stock_code=breeze_symbol,
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=expiry_fmt,
                    right=right,
                    strike_price="0"
                )
                if resp and resp.get("Status") == 200 and resp.get("Success"):
                    return resp["Success"]
                # Try without strike_price param
                resp2 = self.breeze.get_option_chain_quotes(
                    stock_code=breeze_symbol,
                    exchange_code="NFO",
                    product_type="options",
                    expiry_date=expiry_fmt,
                    right=right,
                    strike_price=""
                )
                if resp2 and resp2.get("Status") == 200 and resp2.get("Success"):
                    return resp2["Success"]
                logger.warning(f"_fetch_leg({breeze_symbol}, {right}): status={getattr(resp,'get',lambda k,d=None:d)('Status')} err={getattr(resp,'get',lambda k,d=None:d)('Error')}")
                return []

            calls_raw = _fetch_leg("call")
            puts_raw  = _fetch_leg("put")

            if not calls_raw and not puts_raw:
                logger.warning(f"fetch_option_chain({symbol}→{breeze_symbol}): empty response for expiry {expiry_fmt}")
                return None

            def _parse_legs(raw_list: list) -> pd.DataFrame:
                """
                Parse Breeze option chain list into clean DataFrame.
                Confirmed field names from Breeze API docs:
                  strike_price, ltp, open_interest, chnge_oi,
                  total_quantity_traded, spot_price (str),
                  best_bid_price, best_offer_price
                """
                if not raw_list:
                    return pd.DataFrame()
                rows = []
                for item in raw_list:
                    try:
                        strike = float(item.get("strike_price") or 0)
                        if strike <= 0:
                            continue
                        rows.append({
                            "strike":    strike,
                            "ltp":       float(item.get("ltp") or 0),
                            "oi":        float(item.get("open_interest") or 0),
                            "oi_change": float(item.get("chnge_oi") or 0),
                            "volume":    float(str(item.get("total_quantity_traded") or "0").replace(",", "")),
                            "bid":       float(item.get("best_bid_price") or 0),
                            "ask":       float(item.get("best_offer_price") or 0),
                            # spot_price is a string in each row — use it for spot
                            "spot_price": float(str(item.get("spot_price") or "0").replace(",", "")),
                            # IV not provided by Breeze — estimate from bid/ask midpoint
                            "iv":        0.0,
                        })
                    except Exception:
                        continue
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
                return df

            calls = _parse_legs(calls_raw)
            puts  = _parse_legs(puts_raw)

            if calls.empty and puts.empty:
                return None

            # ── Spot price: embedded in every row as spot_price ─────────────────
            spot = 0.0
            for df in [calls, puts]:
                if not df.empty and "spot_price" in df.columns:
                    sp = df["spot_price"].dropna()
                    sp = sp[sp > 0]
                    if not sp.empty:
                        spot = float(sp.iloc[0])
                        break

            # Fallback: put-call parity on ATM strikes
            if spot <= 0 and not calls.empty and not puts.empty:
                common = sorted(set(calls["strike"].values) & set(puts["strike"].values))
                if common:
                    mid = common[len(common) // 2]
                    c = calls.loc[calls["strike"] == mid, "ltp"].values
                    p = puts.loc[puts["strike"] == mid, "ltp"].values
                    if len(c) and len(p) and c[0] > 0 and p[0] > 0:
                        spot = float(mid) + float(c[0]) - float(p[0])

            # ── PCR ─────────────────────────────────────────────────────────────
            total_call_oi = float(calls["oi"].sum()) if not calls.empty else 0.0
            total_put_oi  = float(puts["oi"].sum())  if not puts.empty  else 0.0
            pcr = total_put_oi / max(total_call_oi, 1.0)

            # ── Max Pain ────────────────────────────────────────────────────────
            max_pain = spot
            try:
                all_strikes = sorted(set(
                    (calls["strike"].tolist() if not calls.empty else []) +
                    (puts["strike"].tolist()  if not puts.empty  else [])
                ))
                if all_strikes and spot > 0:
                    pain = {}
                    for s in all_strikes:
                        cp = calls[calls["strike"] > s]
                        pp = puts[puts["strike"] < s]
                        c_pain = float(((cp["strike"] - s) * cp["oi"]).sum()) if not cp.empty else 0.0
                        p_pain = float(((s - pp["strike"]) * pp["oi"]).sum()) if not pp.empty else 0.0
                        pain[s] = c_pain + p_pain
                    max_pain = min(pain, key=pain.get)
            except Exception:
                pass

            # ── Call Wall / Put Wall (highest OI strikes) ───────────────────────
            call_wall = put_wall = spot
            try:
                if not calls.empty and calls["oi"].sum() > 0:
                    call_wall = float(calls.loc[calls["oi"].idxmax(), "strike"])
                if not puts.empty and puts["oi"].sum() > 0:
                    put_wall = float(puts.loc[puts["oi"].idxmax(), "strike"])
            except Exception:
                pass

            # ── IV Spread (estimated from bid-ask midpoint difference ATM) ──────
            # Breeze doesn't provide IV — we use 0 and note this in scoring
            iv_spread = 0.0

            # ── DTE ─────────────────────────────────────────────────────────────
            try:
                exp_d = ddate.fromisoformat(expiry_date_plain)
                dte = max(0, (exp_d - ddate.today()).days)
            except Exception:
                dte = 7

            return {
                "calls":         calls,
                "puts":          puts,
                "spot":          spot,
                "pcr":           pcr,
                "max_pain":      max_pain,
                "call_wall":     call_wall,
                "put_wall":      put_wall,
                "iv_spread":     iv_spread,
                "total_call_oi": total_call_oi,
                "total_put_oi":  total_put_oi,
                "expiry_date":   expiry_date_plain,
                "dte":           dte,
                "symbol":        symbol,
            }

        except Exception as e:
            logger.warning(f"fetch_option_chain({symbol}): {type(e).__name__}: {e}")
            return None


# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

class Indicators:
    """Technical indicator calculations on DataFrames."""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        return (typical * df["volume"]).cumsum() / df["volume"].cumsum()
    
    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr = Indicators.atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        return dx.rolling(period).mean()
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period=20, std_dev=2.0):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return sma + std_dev * std, sma, sma - std_dev * std
    
    @staticmethod
    def relative_strength(stock_df: pd.DataFrame, nifty_df: pd.DataFrame, period: int = 63) -> float:
        if stock_df is None or nifty_df is None or len(stock_df) < period or len(nifty_df) < period:
            return 50.0
        stock_ret = (stock_df["close"].iloc[-1] / stock_df["close"].iloc[-period] - 1) * 100
        nifty_ret = (nifty_df["close"].iloc[-1] / nifty_df["close"].iloc[-period] - 1) * 100
        rs = stock_ret - nifty_ret
        return min(max(50 + rs * 2, 0), 100)

    @staticmethod
    def rs_acceleration(stock_df: pd.DataFrame, nifty_df: pd.DataFrame, 
                        lookback: int = 21, period: int = 63) -> float:
        """
        Compute RS acceleration (slope of RS over last `lookback` days).
        Positive = RS improving, Negative = RS deteriorating.
        Returns slope in RS-points-per-day (typically -2.0 to +2.0).
        """
        if (stock_df is None or nifty_df is None or 
            len(stock_df) < period + lookback or len(nifty_df) < period + lookback):
            return 0.0
        try:
            from scipy.stats import linregress
        except ImportError:
            # Manual slope calculation if scipy not available
            rs_values = []
            for i in range(lookback, 0, -1):
                idx = -i
                if abs(idx) >= len(stock_df) or abs(idx) >= len(nifty_df):
                    continue
                s_ret = (stock_df["close"].iloc[idx] / stock_df["close"].iloc[idx - period] - 1) * 100
                n_ret = (nifty_df["close"].iloc[idx] / nifty_df["close"].iloc[idx - period] - 1) * 100
                rs_values.append(min(max(50 + (s_ret - n_ret) * 2, 0), 100))
            if len(rs_values) < 5:
                return 0.0
            x = list(range(len(rs_values)))
            n = len(x)
            x_mean = sum(x) / n
            y_mean = sum(rs_values) / n
            num = sum((x[i] - x_mean) * (rs_values[i] - y_mean) for i in range(n))
            den = sum((x[i] - x_mean) ** 2 for i in range(n))
            return num / den if den > 0 else 0.0
        
        rs_values = []
        for i in range(lookback, 0, -1):
            idx = -i
            if abs(idx) >= len(stock_df) or abs(idx) >= len(nifty_df):
                continue
            s_ret = (stock_df["close"].iloc[idx] / stock_df["close"].iloc[idx - period] - 1) * 100
            n_ret = (nifty_df["close"].iloc[idx] / nifty_df["close"].iloc[idx - period] - 1) * 100
            rs_values.append(min(max(50 + (s_ret - n_ret) * 2, 0), 100))
        
        if len(rs_values) < 5:
            return 0.0
        
        x = list(range(len(rs_values)))
        slope, _, _, _, _ = linregress(x, rs_values)
        return round(slope, 3)

    @staticmethod
    def volume_down_day_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Ratio of average volume on down-days vs up-days over lookback period.
        < 1.0 = more volume on up days (bullish accumulation)
        > 1.0 = more volume on down days (distribution)
        """
        if df is None or len(df) < lookback:
            return 1.0
        recent = df.iloc[-lookback:]
        up_days = recent[recent["close"] > recent["open"]]
        down_days = recent[recent["close"] < recent["open"]]
        
        avg_up_vol = up_days["volume"].mean() if len(up_days) > 0 else 1
        avg_down_vol = down_days["volume"].mean() if len(down_days) > 0 else 1
        
        return round(avg_down_vol / (avg_up_vol + 1), 3)

    @staticmethod
    def volatility_compression_ratio(df: pd.DataFrame, short: int = 10, long: int = 50) -> float:
        """
        Ratio of short-term std to long-term std.
        < 0.5 = strong volatility compression (VCP-like)
        """
        if df is None or len(df) < long:
            return 1.0
        std_short = df["close"].iloc[-short:].std()
        std_long = df["close"].iloc[-long:].std()
        return round(std_short / (std_long + 1e-10), 3)

    @staticmethod
    def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Add all common indicators to a dataframe."""
        df = df.copy()
        df["sma_20"] = Indicators.sma(df["close"], 20)
        df["sma_50"] = Indicators.sma(df["close"], 50)
        df["sma_200"] = Indicators.sma(df["close"], 200)
        df["ema_9"] = Indicators.ema(df["close"], 9)
        df["ema_21"] = Indicators.ema(df["close"], 21)
        df["rsi_14"] = Indicators.rsi(df["close"], 14)
        df["rsi_9"] = Indicators.rsi(df["close"], 9)
        df["atr_14"] = Indicators.atr(df, 14)
        df["adx_14"] = Indicators.adx(df, 14)
        df["vol_sma_20"] = Indicators.sma(df["volume"], 20)
        df["vol_sma_50"] = Indicators.sma(df["volume"], 50)
        df["macd"], df["macd_signal"], df["macd_hist"] = Indicators.macd(df["close"])
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = Indicators.bollinger_bands(df["close"])
        df["high_52w"] = df["high"].rolling(window=250, min_periods=50).max()
        df["low_52w"] = df["low"].rolling(window=250, min_periods=50).min()
        df["pct_from_52w_high"] = (df["close"] / df["high_52w"] - 1) * 100
        return df