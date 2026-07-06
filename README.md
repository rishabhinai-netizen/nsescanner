# NSE Scanner NX — v2.0

**One live strategy that works beats eight that don't.**

Clean rebuild of NSE Scanner Pro, redesigned around a single principle: *nothing
gets alerted until forward-test data has proven it.* Built from the audit of
2,226 tracked v15 signals (Apr–Jul 2026).

---

## Why the rebuild

| v15 problem | NX fix |
|---|---|
| 5 of 6 strategies had negative live expectancy (VCP: 0/18) | Killed. EMA21_Bounce is the only LIVE strategy — it earned it (+1.33% avg, 61.5% WR at RS≥90) |
| Regime "gate" only annotated — 711 longs fired in PANIC | The Gate actually gates: hard blocks, promotion/demotion by live PF |
| Hardcoded `STRATEGY_REGIME_PF` priors were fiction (claimed 2.10, live was negative) | Zero priors. `nx_strategy_stats` is recomputed from real outcomes and *is* the gate |
| 40% of signals expired unresolved | RR locked to the 2.0–3.0 tested sweet spot; expiry rate tracked as a first-class health metric |
| 3,391-line app.py monolith + overflow file | Thin router + `core/` + `ui/` + `jobs/` — 22 files, ~1,400 lines total |
| 73 signal CSVs committed to git by Actions | Supabase is the single source of truth. The workflow has **no write permission** to the repo |
| Two AI pipelines (Claude + legacy Groq/Gemini) | Removed from v1 core; add back one Claude pipeline only when it earns its cost |

## The Gate (core innovation)

Every signal is scored (SQI v2) with **weights derived from forward-test data**,
then routed:

- **LIVE** — alerted on Telegram (top 8 by SQI, Tier A/B only)
- **INCUBATING** — persisted and outcome-tracked, never alerted. A strategy
  promotes to LIVE only at **PF ≥ 1.2 on n ≥ 30 closed trades** per regime.
  A LIVE strategy demotes if live PF < 0.8.
- **BLOCKED** — e.g. longs in DISTRIBUTION with RS < 90 (39% WR live)

Evidence behind the SQI weights:

| Factor | Live result |
|---|---|
| RS ≥ 90 | 61.5% WR, +2.2% avg (n=205) |
| RS 70–80 | 38.3% WR — dead zone, penalized |
| ACCUMULATION | 59.5% WR, +2.0% |
| PANIC (RS ≥ 70) | 55.1% WR, +1.5% — bounces work in panic |
| DISTRIBUTION | 39.0% WR — the regime to avoid |
| RR 2.0–3.0 | sweet spot; RR > 3 collapsed to +0.49% |

## Setup

1. **Supabase** — schema already applied to project `uzmpecnlhgbebhxpvafx` (via 3 migrations, July 2026 rebuild). `db/schema.sql` kept for reference/rebuilds.
   RLS is on from day one; jobs use the service key, the app uses anon + auth.
2. **Migrate history** — `python -m migrate.migrate path/to/tracker.csv`
   — optional: `nx_strategy_stats` is already seeded with exact v15 stats; run this only if you also want row-level signal history in the Tracker page.
3. **GitHub secrets** — `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`,
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Workflow runs 7:00 PM IST weekdays.
4. **Streamlit Cloud** — point at `app.py`; add the same secrets plus
   `SUPABASE_ANON_KEY` (and `BREEZE_API_KEY`/`BREEZE_API_SECRET` if using intraday).
5. **Breeze token** — refresh daily via Settings page (→ `nx_app_config`,
   admin-only RLS).

## SaaS-ready by design

`nx_user_profiles` with free/pro/admin tiers, per-user Telegram chat IDs,
signup trigger, and RLS that already separates product data (signals readable
by authenticated users) from admin data (config). Adding payments later means
adding a tier check, not a rewrite.

## Structure

```
app.py                 thin router + T09 Charcoal & Ivory design system
core/
  config.py            secrets (st.secrets → env), never hardcoded
  data.py              yfinance batch (MultiIndex-safe) + Breeze (20s timeout)
  regime.py            EXPANSION / ACCUMULATION / DISTRIBUTION / PANIC
  strategies.py        EMA21_Bounce (LIVE) + incubator interface
  gate.py              SQI v2 + promotion/demotion — the brain
  db.py                Supabase persistence, outcomes, stats rollup
  alerts.py            Telegram (LIVE tier A/B only, capped)
jobs/
  scan.py              Actions entrypoint — Supabase only, zero git writes
  universe.py          dynamic Nifty 500 from NSE (fallback included)
ui/
  dashboard.py         regime, live signals, incubator, live PF matrix
  tracker.py           win rate, PF, equity curve, expiry health, SQI calibration
  settings.py          Breeze token, connection tests, gate policy
db/schema.sql          nx_ tables + RLS + signup trigger
migrate/migrate.py     v15 tracker.csv → nx_signals
```

⚠️ Signals are decision support, not advice. Risk 1–2% per trade, max 6–8 positions.
