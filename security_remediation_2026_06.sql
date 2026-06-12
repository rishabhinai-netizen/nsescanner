-- ============================================================================
-- NSE Scanner Pro — Security Remediation (June 2026 audit)
-- Project: aiebaqvclyzxajigvkfd
-- Run in: Supabase → SQL Editor → New query → paste → Run
--
-- WHAT THIS FIXES (verified live on 2026-06-12):
--   1. app_config (holds your LIVE Breeze brokerage session token) is
--      world-readable AND world-writable via the anon key
--      (policy "service_all_config": roles=public, USING(true)).
--   2. signals / alert_state / journal / portfolio / strategy_performance /
--      weekly_reports — all world-writable (roles=public, USING(true)).
--      Anyone with the anon key could inject fake trade signals into your
--      alert pipeline.
--   3. paper_trades / paper_portfolio — "allow_all" policies silently
--      override the per-user policies (policies OR together), so any user
--      can read/modify every other user's paper trades.
--   4. RLS fully DISABLED on: ar_daily_scores, ar_market_pulse,
--      ar_n250f_snapshot, ar_universe, and all 8 re_* tables.
--
-- SAFETY: The Streamlit app, GitHub Actions, AlphaRadar and ResultsEdge all
-- use the SERVICE ROLE key, which BYPASSES RLS entirely — none of them are
-- affected by any change below. Only anon-key access is being locked down.
-- User-facing features (auth, paper trading game) use authenticated JWTs
-- and keep working through the auth.uid() / authenticated policies.
-- ============================================================================

-- ── 1. app_config: service_role only (CRITICAL — brokerage token) ──────────
DROP POLICY IF EXISTS "service_all_config" ON app_config;
CREATE POLICY "config_service_only" ON app_config
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ── 2. Core scanner tables: service_role writes; signals readable by
--      logged-in users (Virtual Game reads them with a user JWT) ───────────
DROP POLICY IF EXISTS "service_all_signals" ON signals;
DROP POLICY IF EXISTS "signals_read_all"    ON signals;
DROP POLICY IF EXISTS "signals_write_service" ON signals;
CREATE POLICY "signals_service_all" ON signals
  FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "signals_read_authenticated" ON signals
  FOR SELECT TO authenticated USING (true);

DROP POLICY IF EXISTS "service_all_alert" ON alert_state;
CREATE POLICY "alert_state_service_only" ON alert_state
  FOR ALL TO service_role USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "service_all_journal" ON journal;
CREATE POLICY "journal_service_only" ON journal
  FOR ALL TO service_role USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "service_all_portfolio" ON portfolio;
CREATE POLICY "portfolio_service_only" ON portfolio
  FOR ALL TO service_role USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "service_all_strat_perf" ON strategy_performance;
CREATE POLICY "strat_perf_service_all" ON strategy_performance
  FOR ALL TO service_role USING (true) WITH CHECK (true);
-- signal_quality.py reads PF via service key; no anon read needed.

DROP POLICY IF EXISTS "service_all_reports" ON weekly_reports;
CREATE POLICY "reports_service_only" ON weekly_reports
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ── 3. Paper trading game: remove the allow_all overrides ──────────────────
-- The per-user policies (users_own_paper_*) already exist and remain.
DROP POLICY IF EXISTS "allow_all_paper_trades"    ON paper_trades;
DROP POLICY IF EXISTS "allow_all_paper_portfolio" ON paper_portfolio;

-- xp_log / game_achievements have no user_id column → restrict to
-- authenticated users (blocks anonymous abuse; per-user scoping would
-- need a schema change — flagged in the audit report).
DROP POLICY IF EXISTS "allow_all_xp_log" ON xp_log;
CREATE POLICY "xp_log_authenticated" ON xp_log
  FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "xp_log_service" ON xp_log
  FOR ALL TO service_role USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "allow_all_game_achievements" ON game_achievements;
CREATE POLICY "achievements_authenticated" ON game_achievements
  FOR ALL TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "achievements_service" ON game_achievements
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ── 4. Enable RLS where it was fully disabled ───────────────────────────────
-- AlphaRadar tables (service_role policies already exist on them)
ALTER TABLE ar_daily_scores   ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_market_pulse   ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_n250f_snapshot ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_universe       ENABLE ROW LEVEL SECURITY;

-- ResultsEdge tables (no policies existed at all)
ALTER TABLE re_companies     ENABLE ROW LEVEL SECURITY;
ALTER TABLE re_concall       ENABLE ROW LEVEL SECURITY;
ALTER TABLE re_config        ENABLE ROW LEVEL SECURITY;
ALTER TABLE re_financials    ENABLE ROW LEVEL SECURITY;
ALTER TABLE re_guidance      ENABLE ROW LEVEL SECURITY;
ALTER TABLE re_pipeline_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE re_scores        ENABLE ROW LEVEL SECURITY;
ALTER TABLE re_technical     ENABLE ROW LEVEL SECURITY;

CREATE POLICY "re_companies_service"     ON re_companies     FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "re_concall_service"       ON re_concall       FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "re_config_service"        ON re_config        FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "re_financials_service"    ON re_financials    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "re_guidance_service"      ON re_guidance      FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "re_pipeline_logs_service" ON re_pipeline_logs FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "re_scores_service"        ON re_scores        FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "re_technical_service"     ON re_technical     FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ── 5. Verification query — run after the migration ─────────────────────────
-- Every row should show rowsecurity=true and NO policy with roles={public}
-- and qual=true (except none should remain).
-- SELECT t.tablename, t.rowsecurity, p.policyname, p.roles, p.qual
-- FROM pg_tables t LEFT JOIN pg_policies p ON p.tablename = t.tablename
-- WHERE t.schemaname='public' ORDER BY t.tablename;
