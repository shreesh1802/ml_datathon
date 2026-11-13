import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
al = load_all(); n = len(al); te = al.is_test.values; month = al.timestamp.str[:7].values; tr = ~te & (month < '2026-10')
yc = np.load(os.path.join(CACHE, 'y_d1.npy'))
tags = ['E1', 'M1']; P = sum(np.load(os.path.join(CACHE, f'oof_{t}.npy')) for t in tags) / len(tags)
P[te] = sum(np.load(os.path.join(CACHE, f'test_{t}.npy')) for t in tags) / len(tags)
seg = pd.read_pickle(os.path.join(CACHE, 'feats_seg_s1.pkl')); cls = seg.seg_class.values.astype(int)
acc = pd.read_csv('upi-fraud-rings/accounts.csv').set_index('account_id')
S, A = LABELS.index('sim_swap'), LABELS.index('account_takeover')
dn = al.device_id.str[2:].astype(int).values; sn = al.sender_id.str[2:].astype(int).values
kind = np.where(dn >= 420000, 'extra', np.where(dn == sn, 'own', 'other'))
d = pd.DataFrame({'snd': al.sender_id.values, 'day': al.t.values // 86400, 'y': yc, 'given': al.label.values, 'kind': kind, 'cls': cls,
                  'sim': al.sim_age_hours.values, 'dag': al.device_age_days.values, 'away': (al.device_city.values != al.sender_id.map(acc.home_city).values),
                  'hour': al.ts.dt.hour.values, 'amt': al.amount.values, 'bal': al.balance_before.values, 'pin': al.pin_retries.values, 'login': al.login_to_txn_sec.values,
                  'pS': P[:, S], 'pA': P[:, A], 'te': te, 'tr': tr, 'dev': al.device_id.values, 'newp': al.new_payee.values, 'p2p': (al.receiver_type.values == 'P2P'),
                  't': al.t.values})
rows = d[np.isin(yc, [S, A]) & tr]
g = rows.groupby(['snd', 'day'])
ses = pd.DataFrame({'n': g.size(), 'y': g.y.agg(lambda v: v.mode()[0]), 'kinds': g.kind.agg(lambda v: '+'.join(sorted(set(v)))), 'ndev': g.dev.nunique(),
                    'blk': g.cls.max(), 'sim_min': g.sim.min(), 'sim_max': g.sim.max(), 'dag_min': g.dag.min(), 'away': g.away.mean(), 'night': g.hour.agg(lambda h: ((h >= 22) | (h < 5)).mean()),
                    'pin': g.pin.max(), 'login': g.login.median(), 'drain': g.amt.sum() / g.bal.max(), 'pS': g.pS.mean(), 'pA': g.pA.mean(), 'newp': g.newp.mean(), 'span': g.t.max() - g.t.min()})
ses['pred'] = np.where(ses.pS > ses.pA, 'SIM', 'ATO'); ses['lab'] = np.where(ses.y == S, 'SIM', 'ATO')
print('sessions by label, device kinds and model call:'); print(pd.crosstab([ses.lab, ses.kinds], ses.pred).to_string())
print('\nsim-age (hours) minimum in session: quantiles by label and device kind')
print(ses.groupby(['lab', 'kinds']).sim_min.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
print('\ndevice-age minimum (days):'); print(ses.groupby(['lab', 'kinds']).dag_min.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
own = ses[ses.kinds == 'own']
print('\nOWN-phone-only sessions: mean features by label x model call'); print(own.groupby(['lab', 'pred'])[['n', 'sim_min', 'sim_max', 'dag_min', 'away', 'night', 'pin', 'login', 'drain', 'newp', 'span']].mean().round(2).to_string())
