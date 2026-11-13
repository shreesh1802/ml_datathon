"""Label cleaning from out-of-fold probabilities. usage: python src/clean.py OOF_TAG OUT_TAG
Rules (probabilities are out-of-fold, so a row never votes on its own label):
  1. fraud label, model says legit with p > 0.9          -> legit   (random rows filed as fraud)
  2. legit label, model says a fraud class with p > 0.5  -> that class (unreported fraud)
  3. fraud label c', model says class c with p > 0.7 and p(c') < 0.1 -> c (filed under the wrong type)"""
import os, sys
import numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
src, out = sys.argv[1], sys.argv[2]
al = load_all(); p = np.load(os.path.join(CACHE, f'oof_{src}.npy'))
y0 = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
prev = os.path.join(CACHE, f'y_{src}.npy')
y = np.load(prev) if os.path.exists(prev) else y0.copy()   # labels the source model was trained on
tr = (~al.is_test.values) & (al.timestamp.str[:7].values < '2026-10')
top = p.argmax(1); conf = p.max(1); own = p[np.arange(len(p)), np.maximum(y, 0)]
new = y.copy()
r1 = tr & (y > 0) & (p[:, 0] > 0.9); new[r1] = 0
r2 = tr & (y == 0) & (top > 0) & (conf > 0.5); new[r2] = top[r2]
r3 = tr & (y > 0) & (top > 0) & (top != y) & (conf > 0.7) & (own < 0.1); new[r3] = top[r3]
print(f'rule1 fraud->legit {r1.sum()}   rule2 legit->fraud {r2.sum()}   rule3 retyped {r3.sum()}')
d = pd.DataFrame({'given': pd.Series(y0[tr]).map(dict(enumerate(LABELS))).value_counts(), 'clean': pd.Series(new[tr]).map(dict(enumerate(LABELS))).value_counts()})
print(d.to_string())
np.save(os.path.join(CACHE, f'y_{out}.npy'), new)
