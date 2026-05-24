"""
data_quality.py — Data integrity & calendar utilities for NSE Scanner Pro
==========================================================================
Centralises:
  - NSE trading calendar (2026 + 2027 holidays)
  - yfinance retry-with-backoff wrapper
  - Stale data detection
  - Liquidity-adjusted slippage
"""

import time
import logging
from datetime import date, timedelta, datetime
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# NSE HOLIDAY CALENDAR
# Source: https://www.nseindia.com/resources/exchange-communication-holidays
# Update yearly.
# ════════════════════════════════════════════════════════════════════════

NSE_HOLIDAYS_2026 = {
    "2026-01-26",  # Republic Day
    "2026-03-06",  # Holi
    "2026-03-26",  # Ram Navami
    "2026-04-03",  # Mahavir Jayanti
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-04-15",  # Good Friday
    "2026-05-01",  # Maharashtra Day
    "2026-06-19",  # Eid ul-Adha
    "2026-08-15",  # Independence Day
    "2026-08-27",  # Ganesh Chaturthi
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-21",  # Diwali Laxmi Pujan (Muhurat trading evening)
    "2026-11-09",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
}

NSE_HOLIDAYS_2027 = {
    # Placeholder — update with actual 2027 calendar when published by NSE
    "2027-01-26", "2027-03-25", "2027-04-02", "2027-05-01",
    "2027-08-15", "2027-08-26", "2027-10-02", "2027-11-09", "2027-12-25",
}

NSE_HOLIDAYS_ALL = NSE_HOLIDAYS_2026 | NSE_HOLIDAYS_2027


def is_market_day(d: Optional[date] = None) -> bool:
    """True if NSE is/was open on date d (default: today)."""
    d = d or date.today()
    if d.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    if d.isoformat() in NSE_HOLIDAYS_ALL:
        return False
    return True


def previous_market_day(d: Optional[date] = None) -> date:
    """Most recent NSE trading day on or before d."""
    d = d or date.today()
    while not is_market_day(d):
        d -= timedelta(days=1)
    return d


def market_days_between(start: date, end: date) -> int:
    """Count NSE trading days between two dates inclusive."""
    if start > end:
        return 0
    count = 0
    cur = start
    while cur <= end:
        if is_market_day(cur):
            count += 1
        cur += timedelta(days=1)
    return count


def is_market_open_now() -> bool:
    """True if NSE is currently in regular trading hours."""
    from datetime import time as dtime
    import pytz
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if not is_market_day(now.date()):
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)


# ════════════════════════════════════════════════════════════════════════
# RETRY WITH BACKOFF — for yfinance and other flaky external APIs
# ════════════════════════════════════════════════════════════════════════

def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure_return: Any = None,
) -> Any:
    """
    Call func() up to max_attempts times with exponential backoff.
    Returns func's result on success, or on_failure_return on total failure.
    """
    delay = initial_delay
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            last_exc = e
            if attempt < max_attempts:
                logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
    return on_failure_return


# ════════════════════════════════════════════════════════════════════════
# STALENESS DETECTION
# ════════════════════════════════════════════════════════════════════════

def assess_data_quality(df, today: Optional[date] = None) -> str:
    """
    Inspect a price DataFrame and return one of:
      "OK"       — last bar is the most recent market day
      "STALE_1D" — last bar is 1 market day old (yesterday's bar today is fine post-EOD)
      "STALE"    — last bar is 2+ market days old (potential broken data feed)
      "INCOMPLETE" — bars exist but fewer than expected (50 minimum)
      "EMPTY"    — no usable data
    """
    if df is None or df.empty:
        return "EMPTY"
    if len(df) < 50:
        return "INCOMPLETE"

    today = today or date.today()
    last_bar_date = df.index[-1]
    if hasattr(last_bar_date, "date"):
        last_bar_date = last_bar_date.date()

    prev_md = previous_market_day(today)

    if last_bar_date == prev_md or last_bar_date == today:
        return "OK"

    diff = market_days_between(last_bar_date, today) - 1  # exclude bar's own date
    if diff <= 1:
        return "STALE_1D"
    return "STALE"


# ════════════════════════════════════════════════════════════════════════
# LIQUIDITY-ADJUSTED SLIPPAGE
# ════════════════════════════════════════════════════════════════════════

def estimate_slippage_pct(avg_daily_value_inr: float, position_value_inr: float) -> float:
    """
    Estimate one-sided slippage % as a function of:
      - Average daily traded value (proxy for liquidity)
      - Position size relative to ADV

    Calibration heuristic for NSE retail trading:
      - ADV > ₹100 cr, position < 0.1% of ADV  → 0.05%
      - ADV ₹20-100 cr                          → 0.10-0.15%
      - ADV ₹5-20 cr                            → 0.20-0.35%
      - ADV < ₹5 cr                             → 0.50%+ (avoid)
      - Position > 1% of ADV → add penalty
    """
    if avg_daily_value_inr <= 0:
        return 0.5  # Unknown liquidity → assume worst case

    # Base slippage by ADV bucket
    if avg_daily_value_inr >= 100_00_00_000:    # ≥ ₹100 cr
        base = 0.05
    elif avg_daily_value_inr >= 20_00_00_000:   # ₹20-100 cr
        base = 0.10
    elif avg_daily_value_inr >= 5_00_00_000:    # ₹5-20 cr
        base = 0.25
    else:
        base = 0.50

    # Position-size penalty
    if position_value_inr > 0:
        pct_of_adv = position_value_inr / avg_daily_value_inr
        if pct_of_adv > 0.01:    # > 1% of ADV
            base += 0.20
        elif pct_of_adv > 0.005:  # > 0.5% of ADV
            base += 0.10

    return round(base, 3)


# ════════════════════════════════════════════════════════════════════════
# CONVENIENCE — wrap a yfinance fetch with retries
# ════════════════════════════════════════════════════════════════════════

def fetch_with_retry(symbol: str, period: str = "1y", interval: str = "1d"):
    """
    yfinance fetch with 3 retries, exponential backoff.
    Returns DataFrame or None.
    """
    def _do():
        import yfinance as yf
        tk = yf.Ticker(symbol)
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            raise ValueError(f"Empty response from yfinance for {symbol}")
        return df

    return retry_with_backoff(_do, max_attempts=3, initial_delay=1.5,
                              on_failure_return=None)
