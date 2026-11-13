import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
al = load_all()
oof = np.load(os.path.join(CACHE, 'oof_C5.npy')); pt = np.load(os.path.join(CACHE, 'test_C5.npy'))
te = al.is_test.values; month = al.timestamp.str[:7].values; P = oof.copy(); P[te] = pt; top = P.argmax(1)
y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
yc = np.load(os.path.join(CACHE, 'y_c2.npy')); tr = ~te & (month < '2026-10')
dn = al.device_id.str[2:].astype(int).values; sn = al.sender_id.str[2:].astype(int).values
rn = al.receiver_id.str[2:].astype(int).values; p2p = al.receiver_type.values == 'P2P'
print('device number range', dn.min(), dn.max(), ' accounts max', sn.max())
extra = dn >= 420000
print('rows on extra devices (number >= 420000): train', (extra & tr).sum(), ' test', (extra & te).sum(), ' distinct train', len(np.unique(dn[extra & tr])), ' test', len(np.unique(dn[extra & te])))
print('extra-device number quantiles by month:')
for m in ['2026-07', '2026-08', '2026-09']:
    q = extra & tr & (month == m); print(m, np.quantile(dn[q], [0, .1, .5, .9, 1]).astype(int))
print('TEST', np.quantile(dn[extra & te], [0, .1, .5, .9, 1]).astype(int))
patt = tr & (y > 0) & (yc > 0)
d = pd.DataFrame({'lab': al.label.values, 'dn': dn, 'extra': extra, 'own': dn == sn})[patt]
print('\npatterned fraud rows (given label): share on extra device / own device'); print(d.groupby('lab')[['extra', 'own']].mean().round(3).to_string())
print('\nextra-device number quantiles by given label (patterned rows):'); print(d[d.extra].groupby('lab').dn.describe(percentiles=[.05, .5, .95]).round(0).to_string())
print('\nlegit rows on extra devices:', (extra & tr & (y == 0) & (yc == 0)).sum(), ' on non-own account devices:', ((dn != sn) & ~extra & tr & (y == 0)).sum())
# bucket the extra-device number and show label mix
b = pd.cut(dn[patt & extra], np.arange(420000, dn.max() + 1000, 1000))
print('\nlabel mix by extra-device number bucket (patterned fraud rows)'); print(pd.crosstab(b, al.label.values[patt & extra]).to_string())
# how do extra devices appear over time: first-seen time vs number
dd = pd.DataFrame({'dn': dn[extra], 't': al.t.values[extra]}).groupby('dn').t.min()
print('\ncorr(device number, first seen time):', np.corrcoef(dd.index.values, dd.values)[0, 1].round(3))
# receiver/sender id numbers by class
for nm, arr, mask in [('sender', sn, patt), ('P2P receiver', rn, patt & p2p), ('merchant receiver', rn, patt & ~p2p)]:
    print(f'\n{nm} id number quantiles by given label'); print(pd.DataFrame({'lab': al.label.values[mask], 'n': arr[mask]}).groupby('lab').n.describe(percentiles=[.05, .5, .95]).round(0)[['count', 'min', '5%', '50%', '95%', 'max']].to_string())
print('legit sender id quantiles', np.quantile(sn[tr & (y == 0)], [0, .05, .5, .95, 1]).astype(int), ' legit P2P receiver', np.quantile(rn[tr & (y == 0) & p2p], [0, .05, .5, .95, 1]).astype(int))
# txn_id: any structure?
print('\ntxn_id sample', al.txn_id.head(3).tolist())
