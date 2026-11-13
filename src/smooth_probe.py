"""Event-level type voting (the IEEE-CIS "UID post-processing" idea): keep each row's fraud/legit decision,
but choose its fraud TYPE from probabilities averaged over the event it belongs to."""
import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS, macro_f1
al = load_all(); n = len(al); K = len(LABELS); te = al.is_test.values; month = al.timestamp.str[:7].values; tr = ~te & (month < '2026-10')
y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
yc = np.load(os.path.join(CACHE, 'y_d1.npy')); ye = np.where(yc > 0, yc, y)
tags = ['E1', 'M1']; P = sum(np.load(os.path.join(CACHE, f'oof_{t}.npy')) for t in tags) / len(tags)
P[te] = sum(np.load(os.path.join(CACHE, f'test_{t}.npy')) for t in tags) / len(tags)
cls = pd.read_pickle(os.path.join(CACHE, 'feats_seg_s1.pkl')).seg_class.values.astype(int)
tau = np.r_[1.0, [0.212, 0.256, 0.272, 0.239, 0.21, 0.236, 0.253, 0.257, 0.185, 0.196, 0.191]]
def decide(p):
    r = p[:, 1:] / tau[1:]; k = r.argmax(1) + 1; return np.where(r.max(1) > 1, k, 0)
base = decide(P); base[cls > 0] = cls[cls > 0]
fraud = base > 0
day = al.t.values // 86400
groups = {'sender-day': pd.factorize(al.sender_id.values + '_' + day.astype(str))[0],
          'sender-2day': pd.factorize(al.sender_id.values + '_' + (al.t.values // (2 * 86400)).astype(str))[0],
          'receiver-day': pd.factorize(al.receiver_id.values + '_' + day.astype(str))[0],
          'device-day': pd.factorize(al.device_id.values + '_' + day.astype(str))[0]}
print(f'baseline  check(given) {macro_f1(y[tr], base[tr]):.4f}  clean-view {macro_f1(ye[tr], base[tr]):.4f}')
def smooth(g, mask, classes=None):
    """average fraud-type probabilities over fraud-predicted rows of the same group"""
    out = base.copy(); idx = np.where(mask)[0]
    q = P[idx, 1:] / P[idx, 1:].sum(1, keepdims=True)          # type distribution given fraud
    agg = pd.DataFrame(q).groupby(g[idx]).transform('mean').values
    newk = agg.argmax(1) + 1
    if classes is not None:
        ok = np.isin(base[idx], classes) & np.isin(newk, classes); out[idx[ok]] = newk[ok]
    else: out[idx] = newk
    out[cls > 0] = cls[cls > 0]
    return out
S, A = LABELS.index('sim_swap'), LABELS.index('account_takeover')
for nm, g in groups.items():
    for cl_nm, cl in [('all types', None), ('sim/ato only', [S, A])]:
        o = smooth(g, fraud, cl)
        print(f'{nm:13s} {cl_nm:13s} check {macro_f1(y[tr], o[tr]):.4f}  clean-view {macro_f1(ye[tr], o[tr]):.4f}  rows changed train {(o != base)[tr].sum():5d}  test {(o != base)[te].sum():4d}')
o = smooth(groups['sender-day'], fraud, [S, A]); f0 = macro_f1(ye[tr], base[tr], per_class=True); f1 = macro_f1(ye[tr], o[tr], per_class=True)
print('clean-view per class sim/ato before', f0[[S, A]].round(3), ' after', f1[[S, A]].round(3))
