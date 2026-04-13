"""
Tooltips v2.0 — Comprehensive hover/help text for every metric, indicator,
strategy, and AI concept in NSE Scanner Pro.
"""

TIPS = {
    # ══════════════════════════════════════════════════════════════
    # MARKET REGIME
    # ══════════════════════════════════════════════════════════════
    "regime": (
        "Market Regime classifies the overall market into 4 states based on "
        "Nifty trend, momentum, volatility, and breadth. "
        "🟢 EXPANSION = aggressive breakout buying. "
        "🟡 ACCUMULATION = selective, base-building phase. "
        "🟠 DISTRIBUTION = defensive, prefer shorts or sit out. "
        "🔴 PANIC = cash is king, only short setups."
    ),
    "regime_score": (
        "Composite score from -5 to +12 based on: "
        "Golden/Death Cross (±2), Price vs 50 DMA (±1), Price vs 21 EMA (±1), "
        "RSI momentum (±2), MACD (±1), 52W position (±2), Volatility (±3), "
        "ADX (+1), Breadth (±2). Higher = more bullish."
    ),
    "position_size": (
        "Regime-adjusted position sizing. "
        "EXPANSION=100%, ACCUMULATION=70%, DISTRIBUTION=40%, PANIC=15%. "
        "Automatically reduces size as market deteriorates — "
        "your single most important risk control."
    ),
    "india_vix": (
        "India VIX = 'Fear Index' — measures expected volatility of Nifty over next 30 days. "
        "< 13 = extreme complacency (breakout coiling). "
        "13-19 = normal (optimal trading). "
        "20-25 = elevated fear (reduce size). "
        "> 25 = panic (cash only). "
        "Based on NSE F&O option prices."
    ),
    "market_breadth": (
        "Advance/Decline Ratio = advancing stocks ÷ declining stocks. "
        "> 1.5 = broad participation (healthy rally). "
        "< 0.7 = narrow market (only a few stocks holding up — dangerous). "
        "Also measured as % stocks above 200 SMA — > 60% = bull market."
    ),

    # ══════════════════════════════════════════════════════════════
    # SIGNAL QUALITY & SCORING
    # ══════════════════════════════════════════════════════════════
    "sqi": (
        "Signal Quality Index (SQI) — composite score 0-100 measuring how strong "
        "a signal is across 5 dimensions: "
        "📊 Backtest Edge 30% (historical profit factor for this strategy+regime combo), "
        "💪 RS Acceleration 25% (stock outperforming market + accelerating), "
        "🎯 Regime Fit 20% (does this strategy work in current market?), "
        "🔒 Volatility Contraction 15% (tight base = energy coiling), "
        "📈 Volume Confirm 10% (institutional participation). "
        "ELITE ≥80 | STRONG 65-79 | MODERATE 50-64 | WEAK <50"
    ),
    "sqi_grade": (
        "SQI Grade: 🏆 ELITE (80+) = take full size. "
        "⭐ STRONG (65-79) = trade with standard size. "
        "✅ MODERATE (50-64) = reduce size by 25-30%. "
        "⚠️ WEAK (<50) = skip or paper trade only. "
        "Grade adjusts based on regime — same setup scores higher in EXPANSION than DISTRIBUTION."
    ),
    "sqi_breakdown": (
        "Detailed breakdown of the 5 SQI components: "
        "Edge = backtest profit factor score, "
        "RS = relative strength + acceleration score, "
        "Regime = strategy-regime compatibility score, "
        "Vol = volatility contraction quality, "
        "Confirm = volume confirmation score. "
        "Each component out of 100. Weighted sum = final SQI."
    ),
    "rs_rating": (
        "Relative Strength (RS) Rating — how this stock performed vs Nifty 50 "
        "over the past 3 months (weighted: 40% most recent month). "
        "RS 90 = outperformed 90% of all NSE stocks. "
        "RS 70 = minimum bar for long signals. "
        "RS < 30 = laggard — avoid longs, consider shorts. "
        "Mark Minervini: '95% of big winners had RS > 80 before their major move.'"
    ),
    "confidence": (
        "Signal confidence score — how many strategy conditions are satisfied "
        "and how strongly. Factors: price pattern quality, volume confirmation, "
        "trend alignment, sector strength, regime fit. "
        "≥80% = high confidence. 60-79% = moderate. <60% = weak, use caution."
    ),
    "regime_fit": (
        "Strategy-Regime compatibility: "
        "✅ IDEAL = this strategy has highest win rate in current regime. "
        "🟡 OK = acceptable, slight headwind. "
        "⚠️ CAUTION = historically underperforms here, reduce size 50%. "
        "🚫 BLOCKED = strategy fails in this regime — skip entirely. "
        "Based on backtest profit factors per strategy × regime combination."
    ),

    # ══════════════════════════════════════════════════════════════
    # TRADE PARAMETERS
    # ══════════════════════════════════════════════════════════════
    "risk_reward": (
        "Risk-to-Reward Ratio. R:R 1:2.5 means for every ₹1 risked "
        "(Entry price minus Stop Loss), you target ₹2.5 profit (T1). "
        "Professional minimum: 1:2. "
        "Never take trades below 1:1.5 — even a 60% win rate can't save bad R:R."
    ),
    "weekly_aligned": (
        "Multi-timeframe confirmation (weekly chart check). "
        "4 conditions: (1) Close > 10 EMA weekly, (2) 10 EMA > 30 EMA weekly, "
        "(3) RSI > 50 weekly, (4) Bullish candle weekly. "
        "Score 3-4/4 = ✅ aligned = higher probability. "
        "Trading with weekly trend increases win rate by ~15%."
    ),
    "CMP": "Current Market Price — last traded price on NSE.",
    "Entry": (
        "Recommended entry price. 'AT CMP' = buy at current price. "
        "'ABOVE ₹X' = wait for price to break above pivot level with volume. "
        "Never chase — if entry is missed, skip the trade."
    ),
    "SL": (
        "Stop Loss — mandatory exit price if trade goes against you. "
        "Always place a stop. Calculated as: below key support / EMA / "
        "recent low with buffer for daily noise. "
        "Risk = (Entry - SL) × Quantity. Never risk more than 2% of capital per trade."
    ),
    "T1": (
        "Target 1 — first profit booking level (1.5R to 2R from entry). "
        "Book 40-50% of position here. Move stop to breakeven. "
        "Locking profits at T1 converts uncertainty into a 'no-lose' trade."
    ),
    "T2": (
        "Target 2 — extended target (2.5R to 3R from entry). "
        "Hold 30-40% of position after T1 with trailing stop. "
        "Let winners run — most of the profit comes from the 20% of trades that reach T2+."
    ),
    "sector": (
        "Stock's industry sector. Sector alignment amplifies probability: "
        "buying a strong stock in a LEADING sector = best setup. "
        "Buying strong stock in LAGGING sector = unnecessary headwind. "
        "Check RRG sector rotation for quadrant (Leading/Weakening/Improving/Lagging)."
    ),

    # ══════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS
    # ══════════════════════════════════════════════════════════════
    "rsi": (
        "RSI (Relative Strength Index) — momentum oscillator 0-100. "
        "> 70 = overbought (potential reversal or consolidation). "
        "< 30 = oversold (potential bounce). "
        "50-70 = healthy uptrend. "
        "Key: RSI trend > RSI level. Rising RSI in 40-70 range = strong uptrend."
    ),
    "macd": (
        "MACD = Moving Average Convergence Divergence. "
        "Line = 12 EMA minus 26 EMA. Signal = 9 EMA of MACD line. "
        "Histogram = MACD minus Signal. "
        "Bullish crossover = MACD crosses above Signal = buy signal. "
        "Expanding histogram = momentum increasing. "
        "Fading histogram = momentum weakening."
    ),
    "adx": (
        "ADX (Average Directional Index) — measures TREND STRENGTH, not direction. "
        "> 35 = very strong trend (ride it). "
        "25-35 = strong trend (follow momentum). "
        "20-25 = moderate trend. "
        "< 20 = weak/choppy market (avoid breakouts, use mean-reversion). "
        "Key: ADX > 25 + rising = best conditions for breakout strategies."
    ),
    "bollinger_bands": (
        "Bollinger Bands = SMA20 ± 2 standard deviations. "
        "Price at upper band = stretched, not necessarily a sell. "
        "Price at lower band = oversold bounce zone. "
        "BB Squeeze = bands contracting = volatility compression = big move coming. "
        "%B = where price sits within the band (0=lower, 1=upper)."
    ),
    "atr": (
        "ATR (Average True Range) — measures daily price volatility. "
        "ATR% = ATR ÷ Price. 2% ATR means daily expected move is ±2%. "
        "Use for stop placement: stops should be 1.5-2× ATR below entry "
        "to avoid getting stopped out by normal noise."
    ),
    "stochastic": (
        "Stochastic Oscillator — momentum indicator comparing closing price "
        "to recent high-low range. "
        "K > 80 = overbought. K < 20 = oversold. "
        "Best signal: K crosses above D from oversold = buy. "
        "Works best in ranging markets, less reliable in strong trends."
    ),
    "volume_ratio": (
        "Volume Ratio = today's volume ÷ 20-day average volume. "
        "> 2.0x = major institutional activity. "
        "> 1.5x = above-average participation. "
        "< 0.7x = below-average = weak conviction. "
        "Breakouts REQUIRE volume ≥ 1.5x to be valid. "
        "Volume is the 'fuel' — price is the 'car'."
    ),
    "accumulation_distribution": (
        "A/D Ratio over 10 days = (volume on up days) ÷ (volume on down days). "
        "> 1.2 = accumulation (institutions buying). "
        "< 0.8 = distribution (institutions selling). "
        "Smart money moves first — stock price follows. "
        "Divergence between price and A/D = early warning."
    ),

    # ══════════════════════════════════════════════════════════════
    # STRATEGIES
    # ══════════════════════════════════════════════════════════════
    "EMA21_Bounce": (
        "21 EMA Bounce — Best long strategy by backtest (PF 1.96). "
        "Setup: Stock in Stage 2 uptrend, pulls back to rising 21 EMA, "
        "bullish candle reversal with reduced volume on dip. "
        "Entry: Above bounce candle high. "
        "Logic: Strong stocks use 21 EMA as dynamic support — institutional buy zone. "
        "Hold: 5-15 days. Ideal: EXPANSION regime."
    ),
    "VCP": (
        "VCP (Volatility Contraction Pattern) — Mark Minervini's flagship setup. "
        "Stock makes a series of contractions: each pullback smaller than previous "
        "(e.g., 20% → 13% → 8% → 5%). Volume dries up. "
        "Entry: Breakout above pivot (tight area high) with 1.5x volume. "
        "Edge: Stock is 'coiling' — energy building for explosive move. "
        "Hold: 15-40 days. PF 1.80. Win rate 45%."
    ),
    "52WH_Breakout": (
        "52-Week High Breakout — stocks breaking to new 52W highs have no overhead resistance. "
        "Entry: Close above 52W high with volume > 1.5x average. "
        "Edge: Institutions adding at new highs = conviction buying. "
        "Counterintuitive but works — 'new high leads to higher high'. "
        "Hold: 20-60 days. Highest expectancy +5.82%/trade. Win rate 35%."
    ),
    "Failed_Breakout_Short": (
        "Failed Breakout Short — most consistent strategy across all regimes (PF 1.60). "
        "Setup: Stock attempts breakout above resistance, reverses sharply below it. "
        "Entry: Below the failed breakout candle low. "
        "Logic: Trapped longs (bought the breakout) create forced selling. "
        "Works in ALL regimes. Best in DISTRIBUTION/PANIC. Hold: 3-10 days."
    ),
    "Last30Min_ATH": (
        "Last 30 Min ATH (BTST) — highest win rate at 68.4%! "
        "Setup: Stock closing at or near all-time high in last 30 min of trading. "
        "Entry: 3:25 PM. Hold overnight, exit at 9:30 AM next day. "
        "Logic: Strong close = institutions accumulating. Gap-up likely next morning. "
        "Best in EXPANSION. Avoid in PANIC/DISTRIBUTION."
    ),
    "ORB": (
        "Opening Range Breakout (INTRADAY) — price breaks above/below first 15-min candle. "
        "Entry: Break of ORB high/low with volume + VWAP confirmation. "
        "⚠️ REQUIRES Breeze API live intraday data. Cannot use daily candles. "
        "Hold: 2-6 hours. PF 1.72. Best: 9:30-10:30 AM window."
    ),
    "VWAP_Reclaim": (
        "VWAP Reclaim (INTRADAY) — price dips below VWAP then reclaims it with volume. "
        "VWAP = Volume Weighted Average Price = institutional 'fair value' for the day. "
        "Reclaim = institutions stepping in at day's value area. "
        "⚠️ REQUIRES Breeze API. Hold: 2-4 hours. PF 1.84."
    ),
    "Lunch_Low": (
        "Lunch Low Buy (INTRADAY) — reversal at 12:30-1:30 PM low. "
        "Logic: Lunch hour = low volume = weak sellers = mean reversion. "
        "Hammer candle reversal with RSI oversold. "
        "⚠️ REQUIRES Breeze API. Hold: 2-3 hours. PF 1.52."
    ),

    # ══════════════════════════════════════════════════════════════
    # AI DEEP DIVE SPECIFIC
    # ══════════════════════════════════════════════════════════════
    "atlas_agent": (
        "ATLAS — The Technical Eye. "
        "Analyses pure price action: trend structure, momentum (RSI/MACD), "
        "moving average alignment, volume pattern, Bollinger Band position. "
        "Weight varies by regime: highest in TREND regime (50%), lowest in STRESS (10%). "
        "Signal: BUY/HOLD/SELL based on technical score -5 to +5."
    ),
    "oracle_agent": (
        "ORACLE — The Sentiment & Flow Eye. "
        "Analyses institutional activity: volume spikes, A/D ratio (accumulation vs distribution), "
        "52W relative strength, Bollinger Band position, multi-timeframe momentum consistency. "
        "Proxy for 'smart money' positioning. "
        "Weight highest in GOLDILOCKS regime (40%). "
        "Signal: BUY/HOLD/SELL based on sentiment score -8 to +8."
    ),
    "sentinel_agent": (
        "SENTINEL — The Risk Manager. "
        "Assesses position-level risk: realized volatility, ATR%, "
        "distance from 52W high (vulnerable if extended), regime risk multiplier. "
        "Output: LOW/MEDIUM/HIGH risk. "
        "HIGH risk reduces weighted score by 50% × sentinel weight. "
        "Weight highest in STRESS regime (70%) — risk management dominates in fear."
    ),
    "weighted_score": (
        "Swarm Weighted Score = (ATLAS × regime_weight + ORACLE × regime_weight) × "
        "(1 - risk_penalty × SENTINEL_weight). "
        "Range: -1.0 to +1.0. "
        "≥ +0.40 = BUY. ≤ -0.40 = SELL. Between = HOLD. "
        "Distance from threshold matters: +0.80 = strong conviction, +0.41 = marginal."
    ),
    "swarm_consensus": (
        "Swarm Consensus = final decision after all 3 agents vote with regime-adjusted weights. "
        "Unlike a simple majority vote, weights shift dramatically by regime: "
        "in STRESS, SENTINEL has 70% say (risk management dominates). "
        "in TREND, ATLAS has 50% say (technicals lead). "
        "This adaptive weighting is the core innovation."
    ),

    # ══════════════════════════════════════════════════════════════
    # PERFORMANCE & BACKTEST
    # ══════════════════════════════════════════════════════════════
    "win_rate": (
        "Percentage of trades that were profitable. "
        "> 50% is good — but not the full picture. "
        "40% win rate with 3:1 R:R is MORE profitable than 60% win rate with 1:1 R:R. "
        "Focus on expectancy = (win% × avg win) - (loss% × avg loss)."
    ),
    "profit_factor": (
        "Profit Factor = Total Gross Profit ÷ Total Gross Loss. "
        "PF > 2.0 = excellent strategy. "
        "PF 1.5-2.0 = good edge. "
        "PF 1.0-1.5 = marginal (costs may eliminate edge). "
        "PF < 1.0 = losing strategy — do not trade."
    ),
    "expectancy": (
        "Expectancy = average P&L per trade in percentage terms. "
        "Formula: (Win% × Avg Win%) - (Loss% × Avg Loss%). "
        "Positive expectancy = sustainable edge. "
        "This is THE most important metric — it tells you exactly "
        "how much you make on average per rupee risked."
    ),
    "max_drawdown": (
        "Max Drawdown = largest peak-to-trough equity decline during the test period. "
        "Shows worst-case scenario for this strategy. "
        "< 10% = low risk strategy. 10-20% = acceptable. > 25% = high risk. "
        "Sharpe Ratio > 1 with drawdown < 15% = excellent risk-adjusted performance."
    ),

    # ══════════════════════════════════════════════════════════════
    # BREADTH & SECTOR
    # ══════════════════════════════════════════════════════════════
    "ad_ratio": (
        "Advance/Decline Ratio = advancing stocks ÷ declining stocks. "
        "> 1.5 = broad bull market participation. "
        "0.8-1.5 = mixed market. < 0.7 = bearish breadth divergence. "
        "When Nifty rises but A/D deteriorates = narrow rally = top forming."
    ),
    "above_200sma": (
        "% of Nifty 500 stocks trading above their 200-day SMA. "
        "> 65% = strong bull market (buy breakouts). "
        "50-65% = mixed (selective). 35-50% = bear market rally. "
        "< 35% = bear market (defensive). "
        "Best predictor of sustainable vs fragile market rallies."
    ),
    "rrg": (
        "Relative Rotation Graph (RRG) — shows sector momentum vs market. "
        "4 quadrants: 🟢 LEADING (strong + improving), 🟡 WEAKENING (strong + slowing), "
        "🔵 IMPROVING (weak + accelerating), 🔴 LAGGING (weak + declining). "
        "Best trades: stocks in LEADING sectors. Avoid LAGGING sectors. "
        "Rotation: IMPROVING → LEADING is the money rotation."
    ),

    # ══════════════════════════════════════════════════════════════
    # MISC
    # ══════════════════════════════════════════════════════════════
    "fno": (
        "F&O = Futures & Options eligible stock. "
        "Benefits: can short-sell without delivery, hedge with puts, "
        "trade with higher leverage via futures. "
        "~180 stocks on NSE are F&O eligible. Cash stocks can only be shorted intraday."
    ),
    "holding_period": (
        "Expected holding period for this strategy type. "
        "Intraday: exit same day. BTST: exit next morning. "
        "Swing (5-20 days): hold through noise, honor stops. "
        "Positional (20-60 days): wide stops, fundamental tailwind needed."
    ),
    "btst": (
        "BTST = Buy Today Sell Tomorrow. "
        "Hold overnight, exit at open next day. "
        "Target: gap-up or gap-and-go continuation. "
        "Risk: gap-down if negative news overnight. "
        "Best used on ATH stocks or strong earnings plays."
    ),
}


def tip(key: str) -> str:
    """Return tooltip text for a key."""
    return TIPS.get(key, "")


def tip_md(key: str, label: str = "") -> str:
    """Return an info marker with tooltip as HTML title attribute.
    Usage: st.markdown(f"**SQI** {tip_md('sqi')}", unsafe_allow_html=True)
    """
    t = TIPS.get(key, "")
    if not t:
        return ""
    display = label or "ℹ️"
    # Escape single quotes for HTML attribute
    t_escaped = t.replace("'", "&#39;").replace('"', "&quot;")
    return (
        f'<span title="{t_escaped}" style="cursor:help;color:#5dade2;'
        f'font-size:.85em;vertical-align:middle;margin-left:3px;">{display}</span>'
    )


def tip_badge(key: str, label: str = "ℹ️") -> str:
    """Return a styled badge with tooltip.
    Usage in table headers, column labels, etc.
    """
    t = TIPS.get(key, "")
    if not t:
        return label
    t_escaped = t.replace("'", "&#39;").replace('"', "&quot;")
    return (
        f'<abbr title="{t_escaped}" style="text-decoration:none;'
        f'cursor:help;color:inherit;">{label}</abbr>'
    )
