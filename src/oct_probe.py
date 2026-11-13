"""How have patterns changed in October? Compare September (halved, since October has half the volume) to the test month."""
import os, sys
import numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
tag, ytag = sys.argv[1], sys.argv[2]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
al = load_all(); F = pd.read_pickle(os.path.join(CACHE, 'feats.pkl'))
oof = np.load(os.path.join(CACHE, f'oof_{tag}.npy')); pt = np.load(os.path.join(CACHE, f'test_{tag}.npy'))
y = np.load(os.path.join(CACHE, f'y_{ytag}.npy'))
month = al.timestamp.str[:7].values; te = al.is_test.values; sep = (month == '2026-09') & ~te
P = oof.copy(); P[te] = pt
print('== rows with class probability in bucket: Sep/2 vs Oct')
bk = [0.02, 0.1, 0.3, 0.5, 0.8, 1.01]
out = {}
for k in range(1, len(LABELS)):
    a = np.histogram(P[sep, k], bk)[0] / 2; b = np.histogram(P[te, k], bk)[0]
    out[LABELS[k]] = [f'{x:.0f}/{z}' for x, z in zip(a, b)]
print(pd.DataFrame(out, index=['.02-.1', '.1-.3', '.3-.5', '.5-.8', '.8-1']).T.to_string())

print('\n== QR flag by month'); qr = al.txn_type.values == 'QR'
d = pd.DataFrame({'m': np.where(te, 'TEST', month), 'checked': al.qr_vpa_mismatch.notna().values, 'mis': (al.qr_vpa_mismatch.values == 1)})[qr]
print(d.groupby('m').agg(rows=('checked', 'size'), checked=('checked', 'mean'), mis_rate=('mis', 'mean'), mis=('mis', 'sum')).round(4))
nonqr = ~qr
d = pd.DataFrame({'m': np.where(te, 'TEST', month), 'checked': al.qr_vpa_mismatch.notna().values, 'mis': (al.qr_vpa_mismatch.values == 1)})[nonqr]
print('non-QR rows:'); print(d.groupby('m').agg(rows=('checked', 'size'), checked=('checked', 'mean'), mis=('mis', 'sum')).round(4))
tr = ~te & (month < '2026-10')
q = tr & (y == LABELS.index('qr_swap'))
print('clean qr_swap rows: txn_type', pd.Series(al.txn_type.values[q]).value_counts().to_dict(), ' mismatch==1 share', round((al.qr_vpa_mismatch.values[q] == 1).mean(), 3),
      ' flag blank share', round(np.isnan(al.qr_vpa_mismatch.values[q]).mean(), 3))
print('rcv_qrm_6h quantiles among clean qr_swap', np.nanquantile(F.rcv_qrm_6h.values[q], [.05, .25, .5, .75]).round(2))
# receivers with several mismatches in a day: Sep/2 vs Oct
for thr in [2, 3, 5]:
    c = (F.rcv_qrm_1d.values * F.rcv_cnt_1d.values)
    print(f'rows at receivers with >= {thr} mismatches within +-1d: Sep/2 {((c >= thr) & sep).sum() / 2:.0f}  Oct {((c >= thr) & te).sum()}')

print('\n== velocity bots'); vb = tr & (y == LABELS.index('velocity_bot'))
cols = ['dev_cnt_2m', 'dev_cnt_10m', 'dev_cnt_1h', 'dev_cnt_6h', 'dev_gap_min', 'login_to_txn_sec', 'own_dev', 'dev_n_acc', 'dev_n_acc_6h', 'amount', 'new_payee', 'device_age_days', 'snd_cnt_10m']
print(F.loc[vb, cols].describe(percentiles=[.05, .25, .5, .75, .95]).T.round(2).to_string())
print('category', pd.Series(al.merchant_category.fillna('p2p').values[vb]).value_counts(normalize=True).round(3).to_dict())
# device-day sessions
al['dday'] = al.t // 86400
g = al.assign(own=F.own_dev.values, login=al.login_to_txn_sec.values, isvb=(y == LABELS.index('velocity_bot')) & tr,
              pvb=P[:, LABELS.index('velocity_bot')]).groupby(['device_id', 'dday'])
s = g.agg(n=('t', 'size'), span=('t', lambda x: x.max() - x.min()), login=('login', 'median'), own=('own', 'mean'), nacc=('sender_id', 'nunique'),
          vb=('isvb', 'sum'), pvb=('pvb', 'mean'), tst=('is_test', 'max'), m=('timestamp', 'first')) if False else None
key = al.device_id.values + '_' + al.dday.astype(str).values
df = pd.DataFrame({'key': key, 't': al.t.values, 'login': al.login_to_txn_sec.values, 'own': F.own_dev.values, 'snd': al.sender_id.values,
                   'vb': ((y == LABELS.index('velocity_bot')) & tr).astype(int), 'pvb': P[:, LABELS.index('velocity_bot')], 'te': te, 'mon': month,
                   'gm': al.merchant_category.isin(['gaming', 'ecommerce']).values.astype(int), 'amt': al.amount.values})
gg = df.groupby('key')
s = pd.DataFrame({'n': gg.t.size(), 'span': gg.t.max() - gg.t.min(), 'login': gg.login.median(), 'own': gg.own.mean(), 'nacc': gg.snd.nunique(),
                  'vb': gg.vb.sum(), 'pvb': gg.pvb.mean(), 'te': gg.te.max(), 'mon': gg.mon.first(), 'gm': gg.gm.mean(), 'amt': gg.amt.median()})
s = s[s.n >= 4]; s['gap'] = s.span / (s.n - 1)
s['per'] = np.where(s.te, 'OCT', np.where(s.mon == '2026-09', 'SEP', 'other'))
s['gapb'] = pd.cut(s.gap, [-1, 30, 120, 600, 3600, 1e9], labels=['<30s', '30s-2m', '2-10m', '10m-1h', '>1h'])
s['nb'] = pd.cut(s.n, [3, 5, 8, 15, 1000], labels=['4-5', '6-8', '9-15', '16+'])
x = s[s.per != 'other']
print('\ndevice-days with >=4 rows: count by mean gap and period (SEP is full volume; halve it)')
print(pd.crosstab([x.gapb], [x.per]).to_string())
print('\nsame, mean velocity-bot probability'); print(x.pivot_table(index='gapb', columns='per', values='pvb', aggfunc='mean').round(3).to_string())
print('\nsame, median login seconds / own-device share / accounts on device')
print(x.groupby(['gapb', 'per'], observed=True)[['login', 'own', 'nacc', 'gm', 'amt', 'n']].median().round(2).to_string())
print('\nOCT device-days with n>=4, low bot probability (<0.3): by gap bucket, own share bucket')
o = x[(x.per == 'OCT') & (x.pvb < 0.3)]; print(pd.crosstab(o.gapb, pd.cut(o.own, [-0.1, 0.1, 0.9, 1.0])).to_string())
o2 = x[(x.per == 'SEP') & (x.pvb < 0.3)]; print('SEP same:'); print(pd.crosstab(o2.gapb, pd.cut(o2.own, [-0.1, 0.1, 0.9, 1.0])).to_string())
