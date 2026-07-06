-- NSE Scanner NX v2 — schema (nx_ prefix). SaaS-ready with RLS from day one.
-- Run in Supabase SQL editor. Service role bypasses RLS (used by GitHub Actions).

-- ============ SIGNALS (single source of truth — no CSVs in git) ============
create table if not exists nx_signals (
  id            bigint generated always as identity primary key,
  signal_date   date not null,
  created_at    timestamptz not null default now(),
  strategy      text not null,
  symbol        text not null,
  side          text not null check (side in ('LONG','SHORT')),
  entry         numeric(12,2) not null,
  stop          numeric(12,2) not null,
  target1       numeric(12,2) not null,
  target2       numeric(12,2),
  rr            numeric(6,2),
  rs_rank       numeric(5,1),
  sector        text,
  regime        text,
  regime_score  numeric(6,1),
  sqi           numeric(5,1),
  sqi_tier      text,                       -- A / B / C
  gate          text not null,              -- LIVE / INCUBATING / BLOCKED
  status        text not null default 'OPEN',  -- OPEN/TARGET/STOPPED/EXPIRED/INVALID
  exit_date     date,
  exit_price    numeric(12,2),
  exit_reason   text,
  pnl_pct       numeric(8,2),
  meta          jsonb default '{}'::jsonb,
  unique (signal_date, strategy, symbol)
);
create index if not exists nx_signals_status_idx  on nx_signals (status);
create index if not exists nx_signals_date_idx    on nx_signals (signal_date desc);
create index if not exists nx_signals_strat_idx   on nx_signals (strategy, regime);

-- ============ STRATEGY STATS (live PF — feeds the gate, replaces fiction) ====
create table if not exists nx_strategy_stats (
  strategy     text not null,
  regime       text not null,
  n_closed     int not null default 0,
  n_wins       int not null default 0,
  win_rate     numeric(5,1),
  avg_pnl      numeric(8,2),
  profit_factor numeric(8,2),
  updated_at   timestamptz default now(),
  primary key (strategy, regime)
);

-- ============ APP CONFIG (Breeze daily token etc.) ==========================
create table if not exists nx_app_config (
  key        text primary key,
  value      text,
  updated_at timestamptz default now()
);

-- ============ USER PROFILES (SaaS-ready: tiers, telegram per user) ==========
create table if not exists nx_user_profiles (
  user_id     uuid primary key references auth.users (id) on delete cascade,
  email       text,
  name        text,
  tier        text not null default 'free' check (tier in ('free','pro','admin')),
  telegram_chat_id text,
  created_at  timestamptz default now()
);

-- ============ RLS — locked by default =======================================
alter table nx_signals        enable row level security;
alter table nx_strategy_stats enable row level security;
alter table nx_app_config     enable row level security;
alter table nx_user_profiles  enable row level security;

-- Signals & stats: readable by any authenticated user (product data).
-- Writes only via service role (Actions/jobs) — no client-side write policy.
create policy nx_signals_read  on nx_signals        for select to authenticated using (true);
create policy nx_stats_read    on nx_strategy_stats for select to authenticated using (true);

-- App config: admin-only read (contains Breeze token); writes via service role.
create policy nx_config_admin_read on nx_app_config for select to authenticated
  using (exists (select 1 from nx_user_profiles p where p.user_id = auth.uid() and p.tier = 'admin'));

-- Profiles: users see/update only their own row.
create policy nx_profile_self_read   on nx_user_profiles for select to authenticated using (user_id = auth.uid());
create policy nx_profile_self_update on nx_user_profiles for update to authenticated using (user_id = auth.uid());

-- Auto-create profile on signup
create or replace function nx_handle_new_user() returns trigger
language plpgsql security definer set search_path = public as $$
begin
  insert into nx_user_profiles (user_id, email) values (new.id, new.email)
  on conflict (user_id) do nothing;
  return new;
end; $$;
drop trigger if exists nx_on_auth_user_created on auth.users;
create trigger nx_on_auth_user_created after insert on auth.users
  for each row execute function nx_handle_new_user();
