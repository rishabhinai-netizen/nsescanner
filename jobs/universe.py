"""Universe — dynamic Nifty 500 from NSE Indices CSV with hardcoded fallback.
Ported from v15 (the dynamic fetch + fallback pattern was a keeper)."""
from jobs._universe_source import (
    get_stock_universe, get_sector, fetch_nifty500_from_nse)


def get_universe(name: str = "nifty200"):
    return get_stock_universe(name)


def get_sectors() -> dict:
    _, sector_map = fetch_nifty500_from_nse()
    return sector_map or {}
