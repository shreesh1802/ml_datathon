"""Submission with the F1-optimal decision rule: call a row class k when p_k exceeds half of that class's
achievable F1 (a row is worth predicting once its chance of being right is above F/2).
usage: python src/make_sub2.py OUT TAG[,TAG...] [--scale s]"""
import os, argparse
import numpy as np, pandas as pd
from common import load_all, CACHE, SUBS, LABELS, macro_f1
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('tags'); ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--suffix', default=''); ap.add_argument('--seg', default=None); ap.add_argument('--evallabels', default=None); a = ap.parse_args()
al = load_all(); K = len(LABELS)
y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
tr = (~al.is_test.values) & (al.timestamp.str[:7].values < '2026-10')
if a.evallabels:   # clean-test view: patterned rows take their cleaned label, ordinary-looking fraud keeps its given label
    yc = np.load(os.path.join(CACHE, f'y_{a.evallabels}.npy')); y = np.where(yc > 0, yc, y)
tags = a.tags.split(',')
oof = sum(np.load(os.path.join(CACHE, f'oof_{t}.npy')) for t in tags) / len(tags)
pt = sum(np.load(os.path.join(CACHE, f'test_{t}{a.suffix}.npy')) for t in tags) / len(tags)

def decide(p, tau):
    r = p[:, 1:] / tau[1:]
    k = r.argmax(1) + 1
    return np.where(r.max(1) > 1, k, 0)

tau = np.full(K, 0.25)
for it in range(6):
    f = macro_f1(y[tr], decide(oof[tr], tau), per_class=True)
    tau = np.r_[1.0, np.clip(f[1:] / 2, 0.1, 0.45)] * a.scale
pred = decide(oof[tr], tau); f = macro_f1(y[tr], pred, per_class=True)
print('rule macro-F1 on given labels:', round(f.mean(), 4), ' argmax:', round(macro_f1(y[tr], oof[tr].argmax(1)), 4))
print('tau', dict(zip(LABELS[1:], tau[1:].round(3))))
tp = decide(pt, tau); am = pt.argmax(1)
if a.seg:   # a phone's block decides the label of every row on it (on training: block right 83 times, model 2, when they disagree)
    cls = pd.read_pickle(os.path.join(CACHE, f'feats_seg_{a.seg}.pkl')).seg_class.values.astype(int)
    q = cls[tr] > 0; p2 = pred.copy(); p2[q] = cls[tr][q]
    print('block override on training rows:', int((p2 != pred).sum()), 'changed; check', round(macro_f1(y[tr], pred), 4), '->', round(macro_f1(y[tr], p2), 4))
    ct = cls[al.is_test.values]; q = ct > 0; print('block override on test rows:', int((tp[q] != ct[q]).sum()), 'of', int(q.sum()), 'changed'); tp[q] = ct[q]
sub = pd.DataFrame({'id': al.txn_id[al.is_test].values, 'label': np.array(LABELS)[tp]})
os.makedirs(SUBS, exist_ok=True); sub.to_csv(os.path.join(SUBS, f'{a.out}.csv'), index=False)
c = pd.DataFrame({'rule': pd.Series(np.array(LABELS)[tp]).value_counts(), 'argmax': pd.Series(np.array(LABELS)[am]).value_counts(),
                  'oof_rule_per_month': (pd.Series(np.array(LABELS)[pred]).value_counts() / 3).round(0)})
print(c.to_string()); print('wrote', a.out)
