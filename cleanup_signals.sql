-- ============================================================
-- NSE Scanner Pro — Signal History Cleanup SQL
-- Run this in Supabase SQL Editor (once) to fix historical data
-- ============================================================

-- STEP 1: Fix SHORT PnL calculations that were computed wrong
-- SHORT STOPPED: was (entry/sl - 1)*100, correct is (entry-sl)/entry*100
-- SHORT TARGET:  was (entry/t1 - 1)*100, correct is (entry-t1)/entry*100

UPDATE signals
SET pnl_pct = ROUND(((entry - exit_price) / entry * 100)::numeric, 2)
WHERE signal = 'SHORT'
  AND status IN ('STOPPED', 'TARGET', 'EXPIRED')
  AND exit_price IS NOT NULL
  AND entry IS NOT NULL
  AND entry > 0;

-- STEP 2: Mark cross-day duplicates as SUPERSEDED
-- For each (symbol, strategy) combination, keep only the OLDEST original row
-- as the "canonical" signal. Newer duplicate rows get status = 'DUPLICATE'.

-- First identify the canonical row per symbol+strategy (earliest date, OPEN or resolved)
WITH canonical AS (
    SELECT DISTINCT ON (symbol, strategy)
        id, symbol, strategy, date
    FROM signals
    ORDER BY symbol, strategy, date ASC, created_at ASC
),
duplicates AS (
    SELECT s.id
    FROM signals s
    LEFT JOIN canonical c ON s.id = c.id
    WHERE c.id IS NULL
      AND s.status = 'OPEN'  -- only mark OPEN duplicates; resolved ones keep their outcome
)
UPDATE signals
SET status = 'DUPLICATE',
    exit_reason = 'Cross-day duplicate — same signal already open from earlier date'
WHERE id IN (SELECT id FROM duplicates);

-- STEP 3: Expire genuinely old OPEN signals past their max hold period
-- EMA21 Bounce and Last30Min: 15 trading days max
-- Breakout and Failed Breakout Short: 25 trading days max

UPDATE signals
SET status = 'EXPIRED',
    exit_date = CURRENT_DATE::text,
    exit_price = cmp,
    exit_reason = 'Expired — max hold period exceeded',
    pnl_pct = CASE
        WHEN signal = 'BUY'   THEN ROUND(((cmp - entry) / entry * 100)::numeric, 2)
        WHEN signal = 'SHORT' THEN ROUND(((entry - cmp) / entry * 100)::numeric, 2)
        ELSE 0
    END
WHERE status = 'OPEN'
  AND entry IS NOT NULL AND entry > 0
  AND cmp IS NOT NULL
  AND (
      (strategy IN ('EMA21_Bounce', 'Last30Min_ATH')
       AND (CURRENT_DATE - date::date) > 15)
    OR
      (strategy IN ('52WH_Breakout', 'Failed_Breakout_Short')
       AND (CURRENT_DATE - date::date) > 25)
  );

-- STEP 4: Recalculate BUY PnL for STOPPED/TARGET rows that might have wrong values
-- (BUY PnL formula was correct but double-check with actual exit_price)
UPDATE signals
SET pnl_pct = ROUND(((exit_price - entry) / entry * 100)::numeric, 2)
WHERE signal = 'BUY'
  AND status IN ('STOPPED', 'TARGET', 'EXPIRED')
  AND exit_price IS NOT NULL
  AND entry IS NOT NULL
  AND entry > 0;

-- STEP 5: Add unique constraint to prevent future cross-day duplicates
-- Only one OPEN row allowed per symbol+strategy
-- (This creates a partial unique index on OPEN signals)
CREATE UNIQUE INDEX IF NOT EXISTS signals_unique_open_per_strategy
ON signals (symbol, strategy)
WHERE status = 'OPEN';

-- STEP 6: Verification — check the cleaned data
SELECT
    strategy,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE status = 'OPEN')      as open,
    COUNT(*) FILTER (WHERE status = 'TARGET')    as target,
    COUNT(*) FILTER (WHERE status = 'STOPPED')   as stopped,
    COUNT(*) FILTER (WHERE status = 'EXPIRED')   as expired,
    COUNT(*) FILTER (WHERE status = 'DUPLICATE') as duplicate,
    ROUND(
        COUNT(*) FILTER (WHERE status = 'TARGET') * 100.0 /
        NULLIF(COUNT(*) FILTER (WHERE status IN ('TARGET','STOPPED','EXPIRED')), 0)
    , 1) as win_rate_pct,
    ROUND(AVG(pnl_pct) FILTER (WHERE status IN ('TARGET','STOPPED','EXPIRED')), 2) as avg_pnl_pct
FROM signals
GROUP BY strategy
ORDER BY strategy;

