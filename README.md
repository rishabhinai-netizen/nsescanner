# 🎯 NSE Scanner Pro v15

**The most comprehensive NSE trading scanner for serious Indian retail traders.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## What's New in v15

### Merged from Both Branches
- ✅ **Option Chain Analysis** — 7-factor composite scoring via Breeze API
  - OI Change Pattern (25%), Unusual Options Activity (20%), IV Percentile (15%), PCR Contrarian (15%)
  - Max Pain (10%), IV Skew (7.5%), IV Spread (7.5%)
  - Academic basis: Cremers & Weinbaum (2010), Pan & Poteshman (2006)
- ✅ **IPO Base Scanner** — Auto-scrapes NSE for listing data, tracks lock-up calendar
  - Day 30/90/180/540 lock-up alerts with 5-day buffer
  - O'Neil: 57% hit rate, +2.79% alpha on ≥150% volume breakouts
- ✅ **RRG Sector Rotation** — 4-quadrant sector classification (+8 LEADING / -15 LAGGING confidence)
- ✅ **Net P&L / Cost Model** — Realistic backtesting with slippage, STT, brokerage, GST
- ✅ **Volume Profile** — POC, VAH, VAL, HVN, LVN via Breeze intraday data
- ✅ **Stock Lookup Page** — Deep-dive per stock with RRG sector status
- ✅ **Signal History Page** — Entry drift detection, first-seen performance table
- ✅ **Walk-Forward Validation** — Optional toggle, detects overfitting (70/30 train/test)
- ✅ **Breeze Threading Fix** — 20-second timeout using `threading.Thread` (cross-platform)
- ✅ **Breeze Retry Button** — Reconnect without page reload
- ✅ **VCP Rewrite** — std_10/std_50 < 0.50 compression ratio (PF improved from 0.24 → 1.5+)
- ✅ **SQI + Fundamental Gate + Strategy×Regime Matrix** (from v5.2)

---

## Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | Market regime, breadth, RRG, sector heatmap, quick scan |
| 🔍 Scanner Hub | Individual and batch strategy scans |
| 📈 Charts & RS | Candlestick charts, RS rankings, sector performance, volume profile |
| 🔗 Option Chain | 7-factor F&O option chain analysis via Breeze |
| 🚀 IPO Scanner | IPO base detection, breakout signals, lock-up alerts |
| 🔎 Stock Lookup | Deep-dive per stock — indicators, verdict, signal history |
| 📜 Signal History | All flagged stocks, first-seen dates, entry drift detection |
| 🧪 Backtest | Single stock and portfolio backtesting with net P&L + walk-forward |
| 📋 Signal Log | Auto-saved daily signal records |
| 📊 Tracker | Forward-test outcomes — actual live performance |
| 📐 Trade Planner | Position sizing, risk calculation, targets |
| ⭐ Watchlist | Manual watchlist + approaching setups |
| 📓 Journal | Trade journal with P&L analytics and equity curve |
| ⚙️ Settings | Breeze setup, universe, Telegram config |

---

## Quick Start

### 1. Deploy to Streamlit Cloud (Free)

```bash
git clone https://github.com/YOUR_USERNAME/nse-scanner-pro
cd nse-scanner-pro
```

Push to GitHub, then deploy at [share.streamlit.io](https://share.streamlit.io)

### 2. Configure Secrets (Streamlit Settings → Secrets)

```toml
# Breeze API (required for intraday + option chain + volume profile)
BREEZE_API_KEY = "your_api_key"
BREEZE_API_SECRET = "your_api_secret"
BREEZE_SESSION_TOKEN = "your_daily_token"  # Expires daily

# Optional
APP_PASSWORD = "your_password"
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

### 3. Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Architecture

```
app.py                  ← Main UI (14 pages)
├── scanners.py         ← 8 strategies + RRG + regime engine
├── data_engine.py      ← yfinance + Breeze + volume profile
├── backtester.py       ← Cost model + walk-forward validation
├── signal_quality.py   ← SQI + Strategy×Regime matrix
├── fundamental_gate.py ← CANSLIM screening
├── option_chain.py     ← 7-factor OC analysis (NEW v15)
├── ipo_scanner.py      ← IPO base detection (NEW v15)
├── enhancements.py     ← Charts, RS rankings, breadth, journal
├── basket_export.py    ← Zerodha + generic CSV export
├── risk_manager.py     ← Position sizing + heat caps
├── signal_tracker.py   ← Forward testing engine
├── stock_universe.py   ← Nifty 50/200/500 + sectors
├── fno_list.py         ← F&O eligible stocks
└── tooltips.py         ← Help text
```

---

## Data Sources

| Source | Used For | Reliability |
|--------|----------|-------------|
| yfinance | Daily OHLCV, fundamentals, RS | High (free) |
| Breeze API | Intraday, option chain, volume profile | High (authenticated) |
| NSE website | IPO listing data, subscription figures | Medium (scraping) |

---

## For Full Documentation

See **[TOOL_REFERENCE.md](TOOL_REFERENCE.md)** — comprehensive explanation of every feature, every metric, and every decision with research citations.

---

*Not financial advice. Past performance does not guarantee future results.*
