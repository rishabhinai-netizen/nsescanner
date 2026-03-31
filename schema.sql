-- ============================================================
-- NSE Scanner Pro v16 — Supabase Schema
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- 1. SIGNALS — every scan result, deduplicated by day+strategy+symbol
-- ============================================================
CREATE TABLE IF NOT EXISTS signals (
    id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    date        DATE        NOT NULL,
    time        TEXT,
    strategy    TEXT        NOT NULL,
    strategy_name TEXT,
    symbol      TEXT        NOT NULL,
    signal      TEXT        NOT NULL,   -- BUY or SHORT
    cmp         NUMERIC(12,2),
    entry       NUMERIC(12,2),
    sl          NUMERIC(12,2),
    t1          NUMERIC(12,2),
    t2          NUMERIC(12,2),
    rr          NUMERIC(6,2),
    confidence  INTEGER,
    rs          NUMERIC(6,1),
    sqi         NUMERIC(6,1),
    sqi_grade   TEXT,
    sector      TEXT,
    regime      TEXT,
    regime_score INTEGER,
    regime_fit  TEXT,
    status      TEXT        DEFAULT 'OPEN',
    exit_date   DATE,
    exit_price  NUMERIC(12,2),
    exit_reason TEXT,
    pnl_pct     NUMERIC(8,2),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (date, strategy, symbol)
);

CREATE INDEX IF NOT EXISTS idx_signals_date     ON signals (date DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol   ON signals (symbol);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals (strategy);
CREATE INDEX IF NOT EXISTS idx_signals_status   ON signals (status);

-- ============================================================
-- 2. STRATEGY_PERFORMANCE — auto-learning: actual vs blended PF
-- ============================================================
CREATE TABLE IF NOT EXISTS strategy_performance (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    strategy     TEXT        NOT NULL,
    regime       TEXT        NOT NULL,
    prior_pf     NUMERIC(6,3),
    actual_pf    NUMERIC(6,3),
    blended_pf   NUMERIC(6,3),
    trade_count  INTEGER     DEFAULT 0,
    win_count    INTEGER     DEFAULT 0,
    loss_count   INTEGER     DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (strategy, regime)
);

-- ============================================================
-- 3. ALERT_STATE — prevents duplicate Telegram alerts
-- ============================================================
CREATE TABLE IF NOT EXISTS alert_state (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol       TEXT        NOT NULL,
    strategy     TEXT        NOT NULL,
    alert_type   TEXT        NOT NULL,  -- TRIGGER, SL_HIT, T1_HIT
    trigger_price NUMERIC(12,2),
    alerted_at   TIMESTAMPTZ DEFAULT NOW(),
    alerted_date DATE        DEFAULT CURRENT_DATE,
    UNIQUE (symbol, strategy, alert_type, alerted_date)
);

CREATE INDEX IF NOT EXISTS idx_alert_date ON alert_state (alerted_date DESC);

-- ============================================================
-- 4. PORTFOLIO — live open positions with P&L tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol       TEXT        NOT NULL,
    strategy     TEXT,
    signal       TEXT        NOT NULL,  -- BUY or SHORT
    entry_price  NUMERIC(12,2) NOT NULL,
    stop_loss    NUMERIC(12,2),
    target1      NUMERIC(12,2),
    target2      NUMERIC(12,2),
    qty          INTEGER     NOT NULL DEFAULT 1,
    entry_date   DATE        NOT NULL DEFAULT CURRENT_DATE,
    status       TEXT        DEFAULT 'OPEN',  -- OPEN, CLOSED, STOPPED, TARGET
    exit_price   NUMERIC(12,2),
    exit_date    DATE,
    exit_reason  TEXT,
    pnl          NUMERIC(12,2),
    pnl_pct      NUMERIC(8,2),
    sector       TEXT,
    regime_at_entry TEXT,
    sqi_at_entry NUMERIC(6,1),
    notes        TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_status ON portfolio (status);
CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio (symbol);

-- ============================================================
-- 5. JOURNAL — detailed trade notes (replaces trade_journal.json)
-- ============================================================
CREATE TABLE IF NOT EXISTS journal (
    id           SERIAL      PRIMARY KEY,
    symbol       TEXT        NOT NULL,
    strategy     TEXT,
    signal       TEXT,
    entry        NUMERIC(12,2),
    stop         NUMERIC(12,2),
    target1      NUMERIC(12,2),
    qty          INTEGER,
    status       TEXT        DEFAULT 'open',
    exit_price   NUMERIC(12,2),
    pnl          NUMERIC(12,2),
    notes        TEXT,
    reasons      JSONB,
    entry_date   DATE,
    exit_date    DATE,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- 6. WEEKLY_REPORTS — auto-generated performance summaries
-- ============================================================
CREATE TABLE IF NOT EXISTS weekly_reports (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    week_ending  DATE        NOT NULL UNIQUE,
    total_signals INTEGER,
    closed_trades INTEGER,
    wins         INTEGER,
    losses       INTEGER,
    win_rate     NUMERIC(5,1),
    total_pnl_pct NUMERIC(8,2),
    best_strategy TEXT,
    regime_summary TEXT,
    report_json  JSONB,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Row-level security: open for now (add auth later for multi-user)
-- ============================================================
ALTER TABLE signals             ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE alert_state         ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio           ENABLE ROW LEVEL SECURITY;
ALTER TABLE journal             ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_reports      ENABLE ROW LEVEL SECURITY;

-- Allow all operations for service role (used by GitHub Actions)
CREATE POLICY "service_all" ON signals             FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_all" ON strategy_performance FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_all" ON alert_state         FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_all" ON portfolio           FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_all" ON journal             FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_all" ON weekly_reports      FOR ALL USING (true) WITH CHECK (true);

-- ============================================================
-- Seed initial strategy_performance with prior values from backtests
-- (auto_learner.py will update these with real data over time)
-- ============================================================
INSERT INTO strategy_performance (strategy, regime, prior_pf, actual_pf, blended_pf, trade_count) VALUES
    ('VCP',                  'EXPANSION',    1.80, 1.80, 1.80, 0),
    ('VCP',                  'ACCUMULATION', 1.30, 1.30, 1.30, 0),
    ('VCP',                  'DISTRIBUTION', 0.60, 0.60, 0.60, 0),
    ('VCP',                  'PANIC',        0.30, 0.30, 0.30, 0),
    ('EMA21_Bounce',         'EXPANSION',    1.96, 1.96, 1.96, 0),
    ('EMA21_Bounce',         'ACCUMULATION', 1.40, 1.40, 1.40, 0),
    ('EMA21_Bounce',         'DISTRIBUTION', 0.80, 0.80, 0.80, 0),
    ('EMA21_Bounce',         'PANIC',        0.50, 0.50, 0.50, 0),
    ('52WH_Breakout',        'EXPANSION',    1.86, 1.86, 1.86, 0),
    ('52WH_Breakout',        'ACCUMULATION', 1.50, 1.50, 1.50, 0),
    ('52WH_Breakout',        'DISTRIBUTION', 0.70, 0.70, 0.70, 0),
    ('52WH_Breakout',        'PANIC',        0.30, 0.30, 0.30, 0),
    ('Failed_Breakout_Short','EXPANSION',    1.00, 1.00, 1.00, 0),
    ('Failed_Breakout_Short','ACCUMULATION', 1.30, 1.30, 1.30, 0),
    ('Failed_Breakout_Short','DISTRIBUTION', 1.66, 1.66, 1.66, 0),
    ('Failed_Breakout_Short','PANIC',        1.55, 1.55, 1.55, 0),
    ('Last30Min_ATH',        'EXPANSION',    2.10, 2.10, 2.10, 0),
    ('Last30Min_ATH',        'ACCUMULATION', 1.30, 1.30, 1.30, 0),
    ('Last30Min_ATH',        'DISTRIBUTION', 0.60, 0.60, 0.60, 0),
    ('Last30Min_ATH',        'PANIC',        0.30, 0.30, 0.30, 0),
    ('ORB',                  'EXPANSION',    1.72, 1.72, 1.72, 0),
    ('ORB',                  'ACCUMULATION', 1.20, 1.20, 1.20, 0),
    ('ORB',                  'DISTRIBUTION', 0.80, 0.80, 0.80, 0),
    ('ORB',                  'PANIC',        0.50, 0.50, 0.50, 0),
    ('VWAP_Reclaim',         'EXPANSION',    1.84, 1.84, 1.84, 0),
    ('VWAP_Reclaim',         'ACCUMULATION', 1.40, 1.40, 1.40, 0),
    ('VWAP_Reclaim',         'DISTRIBUTION', 0.90, 0.90, 0.90, 0),
    ('VWAP_Reclaim',         'PANIC',        0.50, 0.50, 0.50, 0),
    ('Lunch_Low',            'EXPANSION',    1.52, 1.52, 1.52, 0),
    ('Lunch_Low',            'ACCUMULATION', 1.30, 1.30, 1.30, 0),
    ('Lunch_Low',            'DISTRIBUTION', 0.90, 0.90, 0.90, 0),
    ('Lunch_Low',            'PANIC',        0.60, 0.60, 0.60, 0)
ON CONFLICT (strategy, regime) DO NOTHING;
