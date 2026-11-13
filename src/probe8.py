import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 400)
al = load_all()
oof = np.load(os.path.join(CACHE, 'oof_C5.npy')); pt = np.load(os.path.join(CACHE, 'test_C5.npy'))
te = al.is_test.values; month = al.timestamp.str[:7].values; P = oof.copy(); P[te] = pt; top = np.array([l[:5] for l in LABELS])[P.argmax(1)]
dn = al.device_id.str[2:].astype(int).values; sn = al.sender_id.str[2:].astype(int).values
lab = al.label.str[:5].values
d = pd.DataFrame({'dn': dn, 'lab': lab, 'top': top, 'te': te, 'm': month, 't': al.t.values, 'snd': sn})
x = d[(dn >= 420000) & ~te]
print('TRAIN extra devices: rows by 500-wide device-number bucket and given label')
b = (x.dn // 500) * 500
ct = pd.crosstab(b, x.lab); ct['ndev'] = x.groupby(b).dn.nunique(); ct['first_day'] = (x.groupby(b).t.min() // 86400); ct['last_day'] = (x.groupby(b).t.max() // 86400)
print(ct[ct.index < 431000].to_string())
print('... legit-only region sample:'); print(ct[(ct.index >= 431000)].iloc[::12].to_string())
x = d[(dn >= 420000) & te]
print('\nTEST extra devices: rows by 500-wide bucket and model prediction')
b = (x.dn // 500) * 500
ct = pd.crosstab(b, x.top); ct['ndev'] = x.groupby(b).dn.nunique(); ct['first_day'] = (x.groupby(b).t.min() // 86400); ct['last_day'] = (x.groupby(b).t.max() // 86400)
print(ct.to_string())
# train rows dated october on extra devices
x = d[(dn >= 420000) & ~te & (month >= '2026-10')]; print('\ntrain rows dated Oct on extra devices:'); print(x[['dn', 'lab']].sort_values('dn').to_string(index=False))
# within the train fraud block: are there legit-labelled rows?
x = d[(dn >= 420000) & (dn < 428500) & ~te]
print('\ntrain rows on devices 420000-428500: given label counts', x.lab.value_counts().to_dict())
print('devices in that range used by >1 account:', (x.groupby('dn').snd.nunique() > 1).sum(), ' of', x.dn.nunique())
# number of distinct device numbers present vs span -> are numbers dense?
u = np.unique(dn[(dn >= 420000)])
print('distinct extra device numbers overall', len(u), ' span', u.min(), u.max())
