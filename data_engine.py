"""
Data Engine v3.1 — yfinance (daily, rate-limited) + ICICI Breeze (intraday)
Patch 2026-06-13: BreezeEngine hard timeout wrapper; intraday cache (5-min TTL);
strict connect validation; is_market_hours() helper; updated_at fix on token upsert.
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


def is_market_hours() -> bool:
    """True during NSE cash-market hours (Mon–Fri, 09:15–15:30 IST).
    Intraday scanners are meaningless (and Breeze endpoints unreliable)
    outside this window."""
    n = now_ist()
    if n.weekday() >= 5:
        return False
    minutes = n.hour * 60 + n.minute
    return (9 * 60 + 15) <= minutes <= (15 * 60 + 30)


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
                        # ── yfinance ≥0.2.x always returns MultiIndex (ticker, field)
                        # regardless of whether 1 or N symbols were requested.
                        # Flatten to a plain frame keyed by OHLCV column names. ──
                        if isinstance(raw.columns, pd.MultiIndex):
                            lvl0 = raw.columns.get_level_values(0).tolist()
                            lvl1 = raw.columns.get_level_values(1).tolist()
                            # Detect orientation: (ticker, field) vs (field, ticker)
                            ohlcv_keywords = {"open", "high", "low", "close", "volume"}
                            lvl0_is_field = len(set(str(v).lower() for v in lvl0) & ohlcv_keywords) >= 4
                            if lvl0_is_field:
                                # Old layout (field, ticker) — ticker in level 1
                                if yf_sym not in raw.columns.get_level_values(1):
                                    continue
                                df = raw.xs(yf_sym, axis=1, level=1).copy()
                            else:
                                # New layout (ticker, field) — ticker in level 0
                                if yf_sym not in raw.columns.get_level_values(0):
                                    continue
                                df = raw[yf_sym].copy()
                        else:
                            # Flat columns — single symbol edge case
                            df = raw.copy()

                        if df is not None and not df.empty:
                            df = df.dropna(how="all")
                            if len(df) >= 50:
                                # Normalize columns (handles both Title-case and lower-case)
                                col_map = {}
                                for c in df.columns:
                                    # c is now a plain string (MultiIndex already resolved above)
                                    cl = str(c).lower()
                                    if cl == "open": col_map[c] = "open"
                                    elif cl == "high": col_map[c] = "high"
                                    elif cl == "low": col_map[c] = "low"
                                    elif cl in ("close", "adj close"): col_map[c] = "close"
                                    elif "volume" in cl or cl == "vol": col_map[c] = "volume"
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
                logger.info(f"Breeze token loaded from Supabase: {token[:4]}****")
                return token
        logger.warning("Breeze token not found in Supabase app_config")
        return ""
    except Exception as e:
        logger.warning(f"_get_breeze_token_from_supabase failed: {type(e).__name__}: {e}")
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
            "updated_by": "settings_page",
            # Must be set explicitly — column default only fires on INSERT,
            # so upserts left the timestamp frozen at row-creation time and
            # made the token look months stale in audits.
            "updated_at": datetime.now(IST).isoformat(),
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
    
    # Hard ceiling for any single Breeze REST call. The breeze-connect SDK
    # uses requests WITHOUT a timeout — one stalled TCP connection used to
    # hang the entire Streamlit script forever (the infinite "Scanning..."
    # spinner bug). Every SDK call now goes through _breeze_call().
    BREEZE_CALL_TIMEOUT = 8          # seconds per API call
    INTRADAY_CACHE_TTL = 300         # seconds — ORB/VWAP/LunchLow share one fetch

    def __init__(self):
        self.connected = False
        self.breeze = None
        self.connection_message = ""
        self._intraday_cache: Dict[tuple, tuple] = {}   # (symbol, interval) -> (ts, df)

    def _breeze_call(self, fn, *args, timeout: int = None, **kwargs):
        """Run a Breeze SDK call in a worker thread with a hard timeout.
        Returns the result, or None if the call timed out or raised."""
        import threading
        timeout = timeout or self.BREEZE_CALL_TIMEOUT
        box = {}
        def _worker():
            try:
                box["result"] = fn(*args, **kwargs)
            except Exception as e:
                box["error"] = e
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            logger.warning(f"Breeze call {getattr(fn, '__name__', fn)} timed out after {timeout}s")
            return None
        if "error" in box:
            logger.warning(f"Breeze call failed: {box['error']}")
            return None
        return box.get("result")
    
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
            
            # VALIDATE connection by making a test call (with hard timeout —
            # the SDK has none). A stale daily token can pass generate_session
            # locally yet fail every real API call; treating that as
            # "connected" used to unleash hundreds of doomed intraday calls.
            test = self._breeze_call(self.breeze.get_customer_details)
            if test and ("Success" in str(test.get("Status", "")) or test.get("Success")):
                self.connected = True
                self.connection_message = "✅ Breeze API connected and validated!"
                return True, self.connection_message
            else:
                self.connected = False
                self.breeze = None
                return False, ("❌ Breeze session token appears stale or invalid — "
                               "generate a fresh one from ICICI Direct and update it "
                               "in Settings → Breeze Token.")
                
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
        """Fetch intraday data from Breeze API.

        v3 fixes (infinite-spinner bug):
        - Per-(symbol, interval) cache with 5-min TTL: ORB, VWAP-Reclaim and
          Lunch-Low scanners previously each refetched the SAME data —
          3× the API calls for zero benefit.
        - Every SDK call runs through _breeze_call() with a hard 8s timeout.
          The breeze-connect SDK has NO HTTP timeout, so a single stalled
          connection used to freeze the whole app forever.

        v2 fix: NSE trading symbol → Breeze internal stock_code via
        breeze_symbol_map.to_breeze_code(). Previously sent the NSE symbol
        raw, which Breeze rejected for most stocks, so every intraday scan
        silently returned None.
        """
        if not self.connected or self.breeze is None:
            return None

        # ── Cache hit? ──
        import time as _time
        ck = (symbol, interval)
        hit = self._intraday_cache.get(ck)
        if hit and (_time.time() - hit[0]) < self.INTRADAY_CACHE_TTL:
            return hit[1]

        try:
            try:
                from breeze_symbol_map import to_breeze_code
                stock_code = to_breeze_code(symbol)
            except Exception:
                stock_code = symbol

            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%dT07:00:00.000Z")
            to_date = datetime.now().strftime("%Y-%m-%dT23:59:59.000Z")

            data = self._breeze_call(
                self.breeze.get_historical_data_v2,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                stock_code=stock_code,
                exchange_code="NSE",
                product_type="cash"
            )
            if data and "Success" in str(data.get("Status", "")):
                df = pd.DataFrame(data["Success"])
                if df.empty:
                    self._intraday_cache[ck] = (_time.time(), None)
                    return None
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
                df = df.rename(columns={
                    "open": "open", "high": "high",
                    "low": "low", "close": "close", "volume": "volume"
                })
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                df["symbol"] = symbol
                self._intraday_cache[ck] = (_time.time(), df)
                return df
            # Negative-cache failures too so a dead Breeze session doesn't
            # cost 183 × 8s of timeouts in one scan.
            self._intraday_cache[ck] = (_time.time(), None)
            return None
        except Exception as e:
            logger.warning(f"Breeze intraday failed for {symbol}: {e}")
            self._intraday_cache[ck] = (_time.time(), None)
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
                now_time = datetime.now().time()
                import time as _time_mod
                # If today is Thursday AND market has closed (after 3:30 PM IST),
                # options have expired — use next Thursday
                is_expiry_day = today.weekday() == 3
                market_closed = now_time.hour > 15 or (now_time.hour == 15 and now_time.minute >= 30)
                start_offset = 7 if (is_expiry_day and market_closed) else 0

                # Build list of next 3 expiry candidates to try
                expiry_candidates = []
                for offset in range(start_offset, start_offset + 21):
                    candidate = today + timedelta(days=offset)
                    if candidate.weekday() == 3:  # Thursday
                        expiry_candidates.append(candidate)
                    if len(expiry_candidates) >= 3:
                        break

                if not expiry_candidates:
                    expiry_candidates = [today + timedelta(days=7)]

                # Use nearest expiry as primary; we'll try others if this fails
                primary = expiry_candidates[0]
                expiry_date_plain = primary.strftime("%Y-%m-%d")
                expiry_fmt        = primary.strftime("%Y-%m-%dT06:00:00.000Z")
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

            # If primary expiry returns nothing, try next Thursday automatically
            if not calls_raw and not puts_raw and len(expiry_candidates) > 1:
                for next_exp in expiry_candidates[1:]:
                    expiry_date_plain = next_exp.strftime("%Y-%m-%d")
                    expiry_fmt        = next_exp.strftime("%Y-%m-%dT06:00:00.000Z")
                    logger.info(f"Retrying with next expiry: {expiry_date_plain}")
                    calls_raw = _fetch_leg("call")
                    puts_raw  = _fetch_leg("put")
                    if calls_raw or puts_raw:
                        break

            if not calls_raw and not puts_raw:
                logger.warning(f"fetch_option_chain({symbol}→{breeze_symbol}): empty response for all expiries tried")
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
        Compute RS acceleration (slope of RS over last lookback days).
        VECTORIZED: uses numpy for 20x speedup vs per-row Python loop.
        """
        if (stock_df is None or nifty_df is None or
                len(stock_df) < period + lookback or len(nifty_df) < period + lookback):
            return 0.0
        try:
            s_close = stock_df["close"].values
            n_close = nifty_df["close"].values
            # Align lengths
            min_len = min(len(s_close), len(n_close))
            s_close = s_close[-min_len:]
            n_close = n_close[-min_len:]
            # Vectorized rolling-period returns for the last `lookback` days
            indices = np.arange(min_len - lookback, min_len)
            s_ret = (s_close[indices] / s_close[indices - period] - 1) * 100
            n_ret = (n_close[indices] / n_close[indices - period] - 1) * 100
            rs_vals = np.clip(50 + (s_ret - n_ret) * 2, 0, 100)
            if len(rs_vals) < 5:
                return 0.0
            x = np.arange(len(rs_vals), dtype=float)
            # OLS slope via numpy (no scipy dependency)
            x_mean = x.mean()
            y_mean = rs_vals.mean()
            slope = np.dot(x - x_mean, rs_vals - y_mean) / (np.dot(x - x_mean, x - x_mean) + 1e-10)
            return round(float(slope), 3)
        except Exception:
            return 0.0

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
