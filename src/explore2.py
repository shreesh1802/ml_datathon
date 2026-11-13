"""Probe: test density vs train, burst structure per fraud class, label-noise signatures."""
import sys, os, pandas as pd, numpy as np
D = sys.argv[1]; C = sys.argv[2]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
os.makedirs(C, exist_ok=True)
p = f'{C}/all.pkl'
if os.path.exists(p):
    al = pd.read_pickle(p)
else:
    tr = pd.read_csv(f'{D}/train.csv'); te = pd.read_csv(f'{D}/test.csv'); te['label'] = 'TEST'
    al = pd.concat([tr, te], ignore_index=True)
    al['ts'] = pd.to_datetime(al.timestamp); al['t'] = (al.ts.astype('int64') // 10**9).astype('int64')
    al['is_test'] = (al.label == 'TEST')
    al.to_pickle(p)
al['m'] = al.timestamp.str[:7]; al['d'] = al.timestamp.str[:10]
tr = al[~al.is_test]; te = al[al.is_test]
print('train ts range', tr.timestamp.min(), tr.timestamp.max()); print('test ts range', te.timestamp.min(), te.timestamp.max())
print('\n== rows per day: mean by month, split'); print(al.groupby(['is_test', 'm']).d.value_counts().groupby(level=[0, 1]).agg(['mean', 'min', 'max', 'size']).round(0))
print('\n== active entities per month')
print(al.groupby(['is_test', 'm']).agg(rows=('txn_id', 'size'), send=('sender_id', 'nunique'), recv=('receiver_id', 'nunique'), dev=('device_id', 'nunique')))
print('\n== train rows dated Oct: by day/label'); x = tr[tr.m == '2026-10']; print(x.groupby('d').label.value_counts().unstack(fill_value=0))
print('\n== test rows first/last days'); print(te.d.value_counts().sort_index().iloc[[0, 1, 2, -6, -5, -4, -3, -2, -1]])

# signatures
h = al.ts.dt.hour
home = pd.read_csv(f'{D}/accounts.csv').set_index('account_id').home_city
al['away'] = (al.device_city != al.sender_id.map(home)).astype(int)
al['own_dev'] = (al.device_id.str[3:] == al.sender_id.str[2:]).astype(int)
al['round100'] = (al.amount % 100 == 0).astype(int); al['round500'] = (al.amount % 500 == 0).astype(int)
al['under'] = ((al.amount % 1000 >= 900) | ((al.amount % 500 >= 450) & (al.amount % 500 < 500))).astype(int)
al['amt_lt10'] = (al.amount <= 10).astype(int)
al['sim_fresh'] = (al.sim_age_hours < 72).astype(int); al['dev_new'] = (al.device_age_days < 2).astype(int)
al['fast_login'] = (al.login_to_txn_sec < 10).astype(int)
al['evening'] = ((h >= 19) & (h <= 23)).astype(int)
sig = ['away', 'own_dev', 'round100', 'round500', 'under', 'amt_lt10', 'sim_fresh', 'dev_new', 'fast_login', 'evening', 'new_payee']
print('\n== signature rates by label'); print(al.groupby('label')[sig].mean().round(3))
print('\n== amount quantiles by label'); print(al.groupby('label').amount.quantile([.05, .25, .5, .75, .95]).unstack().round(0))
print('\n== sim_age / device_age / login quantiles'); print(al.groupby('label')[['sim_age_hours', 'device_age_days', 'login_to_txn_sec']].quantile([.1, .5]).unstack().round(1))

# neighbourhoods
al = al.sort_values('t').reset_index(drop=True)
cols = ['timestamp', 'sender_id', 'receiver_id', 'device_id', 'amount', 'txn_type', 'merchant_category', 'device_city', 'sim_age_hours', 'device_age_days', 'new_payee', 'login_to_txn_sec', 'qr_vpa_mismatch', 'label']
rng = np.random.default_rng(1)
tr2 = al[(~al.is_test) & (al.m == '2026-08')]
for lab in sorted(set(tr2.label) - {'legit'}):
    f = tr2[tr2.label == lab]
    for i in rng.choice(len(f), 3, replace=False):
        r = f.iloc[i]
        ents = {r.sender_id, r.receiver_id}
        nb = al[(al.t.between(r.t - 3 * 3600, r.t + 3 * 3600)) & (al.sender_id.isin(ents) | al.receiver_id.isin(ents) | (al.device_id == r.device_id))]
        print(f'\n##### {lab}  seed {r.txn_id}  ({len(nb)} rows within 3h on sender/receiver/device)')
        print(nb[cols].head(14).to_string(index=False))
