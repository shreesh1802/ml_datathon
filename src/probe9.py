import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 1000)
al = load_all()
oof = np.load(os.path.join(CACHE, 'oof_C5.npy')); pt = np.load(os.path.join(CACHE, 'test_C5.npy'))
te = al.is_test.values; month = al.timestamp.str[:7].values; P = oof.copy(); P[te] = pt
S5 = np.array(['legit', 'mule', 'bot', 'layer', 'sim', 'ato', 'phish', 'qr', 'refund', 'fakecb', 'loan', 'merch'])
top = S5[P.argmax(1)]; lab = pd.Series(al.label.values).map(dict(zip(LABELS, S5))).fillna('TEST').values
dn = al.device_id.str[2:].astype(int).values
d = pd.DataFrame({'dn': dn, 'lab': lab, 'top': top, 'te': te})
x = d[(dn >= 423000) & (dn < 428500) & ~te]
b = (x.dn // 100) * 100
ct = pd.crosstab(b, x.lab); ct['ndev'] = x.groupby(b).dn.nunique()
print('TRAIN fraud block, 100-wide buckets, given labels'); print(ct.to_string())
x = al[(dn >= 473631) & (dn < 474560) & te].copy(); x['dn'] = dn[(dn >= 473631) & (dn < 474560) & te]; x['top'] = top[x.index]; x['pmax'] = P[x.index].max(1).round(2)
x['hr'] = x.ts.dt.hour; x['p2p'] = (x.receiver_type == 'P2P').astype(int)
g = x.sort_values('t').groupby('dn')
s = pd.DataFrame({'n': g.size(), 'pred': g.top.agg(lambda v: '/'.join(v)), 'pmax': g.pmax.max(), 'sim_min': g.sim_age_hours.min().round(0), 'dag_max': g.device_age_days.max().round(1),
                  'city': g.device_city.first(), 'hr': g.hr.first(), 'amt': g.amount.agg(lambda v: '/'.join(str(int(a)) for a in v)), 'tt': g.txn_type.first(), 'p2p': g.p2p.mean().round(1),
                  'newp': g.new_payee.mean().round(1), 'login': g.login_to_txn_sec.median(), 'cat': g.merchant_category.first()})
print('\nTEST fraud block devices'); print(s.to_string())
