import os, sys, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
tag = sys.argv[1]
al = load_all(); te = al.is_test.values; pt = np.load(os.path.join(CACHE, f'test_{tag}.npy'))
S5 = np.array(['leg', 'mul', 'bot', 'lay', 'sim', 'ato', 'phi', 'qr', 'ref', 'fak', 'loa', 'mer'])
dn = al.device_id.str[2:].astype(int).values[te]; top = S5[pt.argmax(1)]
m = (dn >= 473631) & (dn < 474620)
d = pd.DataFrame({'dn': dn[m], 'top': top[m]}).groupby('dn').top.agg(lambda v: v.value_counts().index[0] + (f'x{len(v)}' if len(v) > 1 else ''))
print(' '.join(f'{str(k)[3:]}:{v}' for k, v in d.items()))
# train-side check: sim/ATO split by device number vs given labels
tr = ~te; oof = np.load(os.path.join(CACHE, f'oof_{tag}.npy')); dnt = al.device_id.str[2:].astype(int).values
q = tr & (dnt >= 424500) & (dnt < 426800) & al.label.isin(['sim_swap', 'account_takeover']).values
x = pd.DataFrame({'b': (dnt[q] // 100) * 100, 'given': al.label.values[q], 'pred': np.array(LABELS)[oof[q].argmax(1)]})
print(pd.crosstab(x.b, [x.given, x.pred]).to_string())
