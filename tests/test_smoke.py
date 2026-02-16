"""
Smoke Tests for NSE Scanner Pro
=================================
Run with: python -m pytest tests/test_smoke.py -v
Or standalone: python tests/test_smoke.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_dummy_df(n=250, base_price=100.0, trend_up=True):
    """Create a realistic-looking OHLCV DataFrame for testing."""
    dates = pd.bdate_range(end=datetime.now(), periods=n)
    n = len(dates)  # Ensure consistent length
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n) * 1.5) + base_price
    if trend_up:
        close += np.linspace(0, 30, n)
    else:
        close -= np.linspace(0, 15, n)
    close = np.maximum(close, 10)  # Floor at 10
    high = close + np.abs(np.random.randn(n) * 0.8)
    low = close - np.abs(np.random.randn(n) * 0.8)
    low = np.maximum(low, 1)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(100000, 5000000, n).astype(float)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }, index=dates)
    return df


# ============================================================================
# TEST 1: Indicators enrich_dataframe produces expected columns
# ============================================================================

def test_indicators_enrich():
    from data_engine import Indicators
    df = make_dummy_df(250)
    enriched = Indicators.enrich_dataframe(df)
    expected_cols = [
        "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
        "rsi_14", "rsi_9", "atr_14", "adx_14",
        "vol_sma_20", "vol_sma_50",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower",
        "high_52w", "low_52w", "pct_from_52w_high"
    ]
    for col in expected_cols:
        assert col in enriched.columns, f"Missing indicator column: {col}"
    # RSI should be between 0 and 100
    valid_rsi = enriched["rsi_14"].dropna()
    assert valid_rsi.min() >= 0, "RSI below 0"
    assert valid_rsi.max() <= 100, "RSI above 100"
    print("✅ test_indicators_enrich PASSED")


# ============================================================================
# TEST 2: Data quality gate
# ============================================================================

def test_data_quality_gate():
    from data_engine import check_data_quality

    # Good data
    df = make_dummy_df(250)
    ok, reason = check_data_quality(df, "TEST")
    assert ok, f"Good data rejected: {reason}"

    # Too few bars
    ok2, reason2 = check_data_quality(df.iloc[:10], "TEST")
    assert not ok2, "Should reject < 50 bars"

    # Empty
    ok3, reason3 = check_data_quality(pd.DataFrame(), "TEST")
    assert not ok3, "Should reject empty"

    # Zero volume
    df_zv = df.copy()
    df_zv["volume"] = 0
    ok4, reason4 = check_data_quality(df_zv, "TEST")
    assert not ok4, f"Should reject zero-volume: {reason4}"

    print("✅ test_data_quality_gate PASSED")


# ============================================================================
# TEST 3: Cost model calculations
# ============================================================================

def test_cost_model():
    from backtester import CostModel

    cm = CostModel()
    cost = cm.total_cost_pct(100.0, 110.0, shares=100)
    assert cost > 0, "Cost should be positive"
    assert cost < 2.0, f"Cost {cost}% seems too high for a round trip"

    # Higher brokerage for small positions
    cost_small = cm.total_cost_pct(100.0, 110.0, shares=1)
    cost_large = cm.total_cost_pct(100.0, 110.0, shares=1000)
    assert cost_small > cost_large, "Small positions should have higher % brokerage"

    print("✅ test_cost_model PASSED")


# ============================================================================
# TEST 4: ScanResult has expected fields
# ============================================================================

def test_scan_result_schema():
    from scanners import ScanResult
    r = ScanResult(
        symbol="TEST", strategy="VCP", signal="BUY",
        cmp=100, entry=101, stop_loss=95,
        target_1=110, target_2=120, target_3=130,
        risk_reward=2.5, confidence=75,
        reasons=["Test reason"]
    )
    assert r.risk_pct > 0, "Risk % should be positive"
    assert r.entry_gap_pct >= 0, "Entry gap should be >= 0"
    assert r.signal in ("BUY", "SHORT"), "Invalid signal type"
    print("✅ test_scan_result_schema PASSED")


# ============================================================================
# TEST 5: BacktestResult has net fields (costs)
# ============================================================================

def test_backtest_has_costs():
    from backtester import BacktestResult
    import inspect
    fields = [f.name for f in BacktestResult.__dataclass_fields__.values()]
    assert "net_pnl_pct" in fields, "Missing net_pnl_pct field"
    assert "net_profit_factor" in fields, "Missing net_profit_factor field"
    assert "total_costs_pct" in fields, "Missing total_costs_pct field"
    assert "net_expectancy_pct" in fields, "Missing net_expectancy_pct field"
    print("✅ test_backtest_has_costs PASSED")


# ============================================================================
# TEST 6: RRG sector classification
# ============================================================================

def test_rrg_sector_classification():
    from scanners import compute_sector_rrg
    nifty = make_dummy_df(250, base_price=18000, trend_up=True)

    # Create some stock data with known sectors
    data_dict = {}
    for i, sym in enumerate(["INFY", "TCS", "RELIANCE", "HDFCBANK"]):
        df = make_dummy_df(250, base_price=500 + i * 100, trend_up=(i % 2 == 0))
        data_dict[sym] = df

    sector_map = {"INFY": "IT", "TCS": "IT", "RELIANCE": "Energy", "HDFCBANK": "Banking"}
    rrg = compute_sector_rrg(data_dict, nifty, sector_map)

    for sector, info in rrg.items():
        assert "quadrant" in info, f"Missing quadrant for {sector}"
        assert info["quadrant"] in ("LEADING", "WEAKENING", "LAGGING", "IMPROVING"), \
            f"Invalid quadrant: {info['quadrant']}"
        assert 0 <= info["score"] <= 100, f"Score out of range for {sector}"

    print("✅ test_rrg_sector_classification PASSED")


# ============================================================================
# TEST 7: Market regime detection
# ============================================================================

def test_regime_detection():
    from scanners import detect_market_regime
    nifty = make_dummy_df(250, base_price=18000, trend_up=True)
    regime = detect_market_regime(nifty)
    assert regime["regime"] in ("EXPANSION", "ACCUMULATION", "DISTRIBUTION", "PANIC", "UNKNOWN"), \
        f"Invalid regime: {regime['regime']}"
    assert "allowed_strategies" in regime
    assert "blocked_strategies" in regime
    assert 0 <= regime["position_multiplier"] <= 1
    print("✅ test_regime_detection PASSED")


# ============================================================================
# RUN ALL
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_indicators_enrich,
        test_data_quality_gate,
        test_cost_model,
        test_scan_result_schema,
        test_backtest_has_costs,
        test_rrg_sector_classification,
        test_regime_detection,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("🎉 All tests passed!")
    sys.exit(0 if failed == 0 else 1)
