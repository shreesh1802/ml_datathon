"""Blend saved probabilities, tune per-class weights, write a submission.
usage: python src/make_sub.py OUT_NAME TAG[,TAG...] [--labels y_tag] [--noweights]"""
import os, argparse
import numpy as np, pandas as pd
from common import load_all, CACHE, SUBS, LABELS, macro_f1, tune_weights
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('tags'); ap.add_argument('--labels', default=None)
ap.add_argument('--noweights', action='store_true'); a = ap.parse_args()
al = load_all(); K = len(LABELS)
y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
if a.labels: y = np.load(os.path.join(CACHE, f'y_{a.labels}.npy'))
tr = (~al.is_test.values) & (al.timestamp.str[:7].values < '2026-10')
tags = a.tags.split(',')
oof = sum(np.load(os.path.join(CACHE, f'oof_{t}.npy')) for t in tags) / len(tags)
pt = sum(np.load(os.path.join(CACHE, f'test_{t}.npy')) for t in tags) / len(tags)
print('argmax OOF macro-F1', round(macro_f1(y[tr], oof[tr].argmax(1)), 4))
w = np.ones(K)
if not a.noweights:
    w, s = tune_weights(oof[tr], y[tr]); print('tuned', round(s, 4), 'weights', dict(zip(LABELS, w.round(2))))
f = macro_f1(y[tr], (oof[tr] * w).argmax(1), per_class=True); print(dict(zip(LABELS, f.round(3))))
sub = pd.DataFrame({'id': al.txn_id[al.is_test].values, 'label': np.array(LABELS)[(pt * w).argmax(1)]})
os.makedirs(SUBS, exist_ok=True); sub.to_csv(os.path.join(SUBS, f'{a.out}.csv'), index=False)
print((sub.label.value_counts()).to_string()); print('wrote', a.out, sub.shape)
