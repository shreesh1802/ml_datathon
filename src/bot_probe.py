"""Have October velocity bots changed shape? Look for bursts the model calls legit, by device, by merchant, by sender."""
import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS, Win
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
al = load_all(); n = len(al); te = al.is_test.values; month = al.timestamp.str[:7].values
tags = ['E1', 'M1']; P = sum(np.load(os.path.join(CACHE, f'oof_{t}.npy')) for t in tags) / len(tags)
P[te] = sum(np.load(os.path.join(CACHE, f'test_{t}.npy')) for t in tags) / len(tags)
legitp = P[:, 0] > 0.5; t = al.t.values.astype(np.int64)
per = np.where(te, 'OCT', month); per = np.where(~te & (month >= '2026-10'), 'x', per)
login = al.login_to_txn_sec.values; amt = al.amount.values; mer = al.receiver_type.values == 'MERCHANT'
cat = al.merchant_category.fillna('p2p').values; gm = np.isin(cat, ['gaming', 'ecommerce'])
def burst_counts(keyname, key, w, thr, extra=None):
    W = Win(key, t); c = W.count(key, t, -w, w)
    m = (c >= thr) & legitp & (extra if extra is not None else True)
    out = {p: int((m & (per == p)).sum()) for p in ['2026-07', '2026-08', '2026-09', 'OCT']}
    print(f'{keyname:40s} legit-predicted rows in bursts (>= {thr} within +-{w}s): ', out)
    return m
dv = pd.factorize(al.device_id)[0].astype(np.int64); rc = pd.factorize(al.receiver_id)[0].astype(np.int64); sd = pd.factorize(al.sender_id)[0].astype(np.int64)
burst_counts('device', dv, 600, 5)
burst_counts('device, merchant rows', dv, 600, 5, mer)
burst_counts('device 1h', dv, 3600, 8)
burst_counts('sender', sd, 600, 5)
burst_counts('sender 1h', sd, 3600, 8)
burst_counts('merchant (gaming/ecom) 10m', rc, 600, 15, gm & mer)
fast = login < 8
fk = rc * 2 + fast  # fast-login rows at the same merchant
burst_counts('merchant fast-login rows 10m', fk, 600, 5, fast & mer)
burst_counts('merchant fast-login rows 1h', fk, 3600, 8, fast & mer)
# fast-login rows overall to gaming/ecom by period (per-month counts; October has half volume)
for p in ['2026-07', '2026-08', '2026-09', 'OCT']:
    q = per == p
    print(p, 'fast(<8s) gaming/ecom merchant rows:', int((q & fast & gm & mer).sum()), ' of which predicted bot', int((q & fast & gm & mer & (P.argmax(1) == 2)).sum()),
          ' fast rows overall', int((q & fast).sum()), '  login<5', int((q & (login < 5)).sum()))
# distribution of login seconds for gaming/ecom rows not predicted bot, Oct vs Sep
for p in ['2026-09', 'OCT']:
    q = (per == p) & gm & mer & (P.argmax(1) != 2)
    print(p, 'gaming/ecom non-bot login quantiles', np.quantile(login[q], [.001, .01, .05, .1]).round(1), ' n', q.sum())
