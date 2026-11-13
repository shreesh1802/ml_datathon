"""Trap check: for rows the model calls fraud, do features look the same in October as in September?
Standardised mean difference (Oct predicted-fraud vs Sep predicted-fraud, both by out-of-fold/test predictions)."""
import os, sys, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
tag = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
al = load_all(); F = pd.read_pickle(os.path.join(CACHE, 'feats.pkl'))
F['ip_raw'] = al.ip_risk_score.values; F['app_version'] = al.app_version.values
oof = np.load(os.path.join(CACHE, f'oof_{tag}.npy')); pt = np.load(os.path.join(CACHE, f'test_{tag}.npy'))
te = al.is_test.values; month = al.timestamp.str[:7].values; P = oof.copy(); P[te] = pt
pred = P.argmax(1); conf = P.max(1)
sep = (~te) & np.isin(month, ['2026-08', '2026-09']);
sd = F[sep | te].std().replace(0, 1)
rows = []
for k in range(1, len(LABELS)):
    a = sep & (pred == k) & (conf > 0.8); b = te & (pred == k) & (conf > 0.8)
    d = ((F[b].mean() - F[a].mean()) / sd)
    # same comparison for legit rows, to subtract the general October shift
    for c in d.abs().sort_values(ascending=False).head(12).index:
        rows.append((LABELS[k], c, round(F.loc[a, c].mean(), 3), round(F.loc[b, c].mean(), 3), round(d[c], 2)))
print(pd.DataFrame(rows, columns=['class', 'feature', 'AugSep_mean', 'Oct_mean', 'std_diff']).to_string(index=False))
la = sep & (pred == 0); lb = te & (pred == 0)
d = ((F[lb].mean() - F[la].mean()) / sd).sort_values(key=abs, ascending=False).head(15)
print('\nlegit rows general shift:'); print(d.round(2).to_string())
