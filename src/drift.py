"""Which features behave differently in October? Per-feature train-month vs October separability (AUC),
with September-vs-(Jul+Aug) as the natural-noise reference. Writes cache/drift.csv."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_all, CACHE

al = load_all(); F = pd.read_pickle(os.path.join(CACHE, 'feats.pkl'))
month = al.timestamp.str[:7].values
rng = np.random.default_rng(0); samp = rng.random(len(al)) < 0.15
octm = al.is_test.values & samp; sep = (month == '2026-09') & ~al.is_test.values & samp; ja = (month < '2026-09') & samp
rows = []
for c in F.columns:
    v = F[c].values.astype(np.float64); v = np.where(np.isnan(v), -1e9, v)
    def auc(a, b):
        s = roc_auc_score(np.r_[np.zeros(a.sum()), np.ones(b.sum())], np.r_[v[a], v[b]]); return abs(s - 0.5)
    rows.append((c, auc(sep, octm), auc(ja, sep)))
d = pd.DataFrame(rows, columns=['feature', 'sep_vs_oct', 'julaug_vs_sep']).sort_values('sep_vs_oct', ascending=False)
d.to_csv(os.path.join(CACHE, 'drift.csv'), index=False)
pd.set_option('display.max_rows', 200); pd.set_option('display.width', 200)
print(d.head(70).round(3).to_string(index=False))
