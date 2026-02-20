# NSE Scanner Pro v5.2 — CHANGELOG
## 10-Item Upgrade from Deep Research Backtests

### 🔴 CRITICAL FIX

#### 1. VCP Scanner Complete Rewrite (`scanners.py`)
**The old VCP was actively destroying capital** (Profit Factor 0.24-0.29).

**Root causes fixed:**
- ❌ Fixed 40d/25d/10d windows → ✅ `std_10/std_50 < 0.50` volatility compression ratio
- ❌ Entered BEFORE breakout → ✅ Entry ONLY when close ≥ pivot on 1.3x+ volume  
- ❌ Stop at 10-day low minus ATR (8-12% risk) → ✅ Contraction low × 0.995 (max 8%)
- ❌ No volume quality check → ✅ Down-day volume < up-day volume required
- ❌ 0.65 contraction threshold → ✅ 0.50 (much stricter)
- ❌ No range depth check → ✅ 25-day range must be < 15% of price

**Expected improvement:** PF 0.24 → PF 1.5-2.0 (conservative estimate)

---

### 🟡 HIGH PRIORITY

#### 2. Signal Quality Index — NEW (`signal_quality.py`)
Replaces raw confidence with evidence-based multi-factor ranking:

```
SQI = (0.30 × Backtest_Edge) + (0.25 × RS_Acceleration) + 
      (0.20 × Regime_Fit) + (0.15 × Vol_Contraction) + (0.10 × Volume_Confirm)
```

- **ELITE (80+):** Top priority, full position
- **STRONG (65-79):** Take it, standard size
- **MODERATE (50-64):** Smaller size
- **WEAK (<50):** Skip or paper-trade

#### 3. Fundamental Quality Gate — NEW (`fundamental_gate.py`)
Prevents buying breakouts on fundamentally broken stocks:
- EPS growth QoQ > 0%
- Revenue growth > 0%
- Market cap > ₹500 Cr
- PE ratio < 100
- Debt/Equity < 2.0

Adds **SQI penalty** (not hard block) for failing stocks. Grade A-F system.

#### 4. Strategy × Regime Matrix (`scanners.py`, `signal_quality.py`)
Pre-computed profit factors from real backtests:

| Strategy | EXPANSION | ACCUMULATION | DISTRIBUTION | PANIC |
|----------|-----------|--------------|--------------|-------|
| VCP | 1.8 | 1.3 | 0.6 | 0.3 |
| EMA21_Bounce | 1.96 | 1.4 | 0.8 | 0.5 |
| 52WH_Breakout | 1.86 | 1.5 | 0.7 | 0.3 |
| Failed_Short | 1.0 | 1.3 | 1.66 | 1.55 |

Strategies with PF < 0.7 in current regime are **automatically blocked**.

#### 7. Approaching Setup Watchlist — NEW (`scanners.py`)
Solves the "zero signals" problem. Shows stocks 50-95% through a setup:
- **VCP approaching:** Vol compressing (0.50-0.75x) but not tight enough yet
- **EMA21 approaching:** Pulling back toward 21 EMA (1.5-5% above)
- **52WH approaching:** Within 1-5% of 52-week high

---

### 🟢 MEDIUM PRIORITY

#### 5. RS Acceleration Slope (`data_engine.py`)
Uses `scipy.stats.linregress` on 21-day RS values to compute slope:
- Positive slope = RS improving (momentum accelerating)
- Negative slope = RS deteriorating (losing relative strength)
- Used as 25% weight in SQI calculation
- Falls back to manual slope if scipy unavailable

#### 6. Regime-Adaptive Heat Caps (`risk_manager.py`)
Portfolio heat limits that tighten as market deteriorates:
- EXPANSION: 6.0% max heat
- ACCUMULATION: 4.0% max heat
- DISTRIBUTION: 2.0% max heat
- PANIC: 1.0% max heat

Also includes regime-adaptive risk-per-trade limits.

#### 8. Broker Basket Export — NEW (`basket_export.py`)
One-click CSV export compatible with:
- **Zerodha Kite Basket Orders** (direct import)
- **Generic broker CSV** (full trade plan with position sizing)

Includes position sizing, risk calculation, and portfolio summary.

#### 9. Auto-Dim Weak Strategies (`scanners.py`)
`StrategyHealthTracker` monitors rolling profit factor from signal tracker:
- PF ≥ 1.5: STRONG 💪 (full SQI weight)
- PF ≥ 1.0: OK ✅ (full weight)
- PF ≥ 0.8: WEAK ⚠️ (70% SQI weight + warning)
- PF < 0.8: FAILING 🔴 (50% SQI weight + strong warning)

#### 10. Volume Dry-Up on Down-Days (`data_engine.py`)
New indicator: `volume_down_day_ratio(df, lookback=20)`
- Ratio < 1.0 = More volume on up-days → bullish accumulation
- Ratio > 1.0 = More volume on down-days → distribution
- Used in VCP validation AND as SQI component

---

### Updated Strategy Profiles (from real backtests)
- VCP win_rate: 67.2% → 45.0% (realistic), PF: 3.1 → 1.8 (rewritten)
- EMA21 Bounce: PF 2.2 → 1.96 (actual backtest)
- 52WH Breakout: PF 2.8 → 1.86 (actual backtest)
- Failed Breakout Short: PF 2.5 → 1.60 (actual backtest, wins by R:R not win rate)

---

### New Files
- `signal_quality.py` — Signal Quality Index (SQI)
- `fundamental_gate.py` — Fundamental quality gate
- `basket_export.py` — Broker basket export

### Modified Files
- `scanners.py` — VCP rewrite, approaching setups, strategy health, SQI integration
- `data_engine.py` — RS acceleration, vol compression ratio, vol down-day ratio
- `risk_manager.py` — Regime-adaptive heat caps
- `requirements.txt` — Added scipy

### NOT Changed (intentionally)
- `app.py` — UI integration deferred to next session (all backend ready)
- `enhancements.py` — No changes needed
- `auto_scanner.py` — Compatible with new scanner signatures
- `backtester.py` — Compatible, can benefit from new indicators later

---

### Integration Notes for UI (next step)
1. Import `signal_quality`, `fundamental_gate`, `basket_export` in app.py
2. Display SQI grade next to each signal (replace or augment confidence)
3. Add "Watchlist" tab showing approaching setups
4. Add fundamental gate toggle in sidebar
5. Add basket export button in results section
6. Show strategy×regime matrix in regime panel
7. Show strategy health status in sidebar
8. Update portfolio heat display with regime-adaptive cap
