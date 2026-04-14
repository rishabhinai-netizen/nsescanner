import os, pandas as pd
from supabase import create_client
sb = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_KEY'])
res = sb.table('signals').select('symbol,strategy,signal,date,entry,sl,t1,cmp,status,exit_price,pnl_pct,exit_reason,regime,rs,sqi,sqi_grade,scan_count,first_seen_ist').order('date').execute()
df = pd.DataFrame(res.data or [])
print('SIGNAL_DATA_START')
print(df.to_csv(index=False))
print('SIGNAL_DATA_END')
print('TOTAL:', len(df))
