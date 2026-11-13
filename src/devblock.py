"""Device-number coordinates. usage: python src/devblock.py STAGE1_TAG OUT_NAME [BASE_LABELS]

Phones that do not belong to an account (device number above the account range) are numbered in the order the
simulator created them: first the phones used in fraud events, grouped by fraud type, then ordinary second phones.
October has its own run of new numbers with the same layout.  We never hard-code where a block starts or ends.
Instead each row on such a phone gets, for every fraud type, the share of that type's confidently-predicted rows
(first-stage model, out-of-fold for training rows) that sit on lower device numbers in the same run.  Those shares
are a coordinate system the second model can learn on Jul-Sep and reuse on October's run.

Also writes labels y_OUT.npy: BASE_LABELS, except that a fraud label given to a row on an extra phone is kept
(those rows look ordinary on their own, so label cleaning had turned them into legit)."""
import os, sys
import numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
tag, out = sys.argv[1], sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 else None
al = load_all(); n = len(al); K = len(LABELS)
te = al.is_test.values; month = al.timestamp.str[:7].values
P = np.load(os.path.join(CACHE, f'oof_{tag}.npy')).astype(np.float64); P[te] = np.load(os.path.join(CACHE, f'test_{tag}.npy'))
dn = al.device_id.str[2:].astype(np.int64).values; sn = al.sender_id.str[2:].astype(np.int64).values
n_acc = 420000
ext = dn >= n_acc
train_max = dn[~te & (month < '2026-10')].max()          # last number handed out in the training period
epoch = np.where(dn > train_max, 1, 0)
pred = P.argmax(1); conf = P.max(1)
tk = np.isin(pred, [LABELS.index('sim_swap'), LABELS.index('account_takeover')])
G = {'dev_extra': ext.astype(np.float32), 'dev_other_acct': ((dn != sn) & ~ext).astype(np.float32)}
groups = {LABELS[k][:5]: (pred == k) & (conf > 0.5) for k in range(1, K)}
groups['take'] = tk & (conf > 0.5); groups['any'] = (pred > 0) & (conf > 0.5)
for nm, mask in groups.items():
    cdf = np.full(n, np.nan); below = np.full(n, np.nan); above = np.full(n, np.nan)
    for e in [0, 1]:
        ref = np.sort(dn[mask & ext & (epoch == e)]); q = ext & (epoch == e)
        if len(ref) == 0:
            cdf[q] = -1; below[q] = 0; above[q] = 0; continue
        lo = np.searchsorted(ref, dn[q], 'left'); hi = len(ref) - np.searchsorted(ref, dn[q], 'right')
        cdf[q] = lo / len(ref); below[q] = np.minimum(lo, 25); above[q] = np.minimum(hi, 25)
    G[f'db_cdf_{nm}'] = cdf; G[f'db_below_{nm}'] = below; G[f'db_above_{nm}'] = above
# local votes: first-stage probabilities averaged over the 20 nearest extra phones on each side (by rank), own phone excluded
dev = pd.DataFrame({'dn': dn[ext], **{f'p{k}': P[ext, k] for k in range(K)}}).groupby('dn').mean()
dev['e'] = (dev.index.values > train_max).astype(int)
R = 20
for k in range(K):
    v = np.full(len(dev), np.nan)
    for e in [0, 1]:
        m = (dev.e.values == e); x = dev[f'p{k}'].values[m]
        cs = np.concatenate([[0.0], np.cumsum(x)]); i = np.arange(len(x)); lo = np.maximum(i - R, 0); hi = np.minimum(i + R + 1, len(x))
        v[m] = (cs[hi] - cs[lo] - x) / np.maximum(hi - lo - 1, 1)
    col = np.full(n, np.nan); col[ext] = pd.Series(v, index=dev.index).reindex(dn[ext]).values
    G[f'db_nb_{LABELS[k][:5]}'] = col
# sim swap and takeover on a new phone look alike row by row, so also vote on the merged pair
v = np.full(n, np.nan); v[ext] = (G['db_nb_sim_s'] + G['db_nb_accou'])[ext]; G['db_nb_take'] = v
rk = np.full(n, np.nan)
for e in [0, 1]:
    q = ext & (epoch == e); u = np.unique(dn[q]); rk[q] = np.searchsorted(u, dn[q]) / max(len(u), 1)
G['db_rank'] = rk
G['dev_rows'] = np.where(ext, np.minimum(pd.Series(dn).map(pd.Series(dn).value_counts()).values, 10), np.nan)
pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in G.items()}).to_pickle(os.path.join(CACHE, f'feats_db_{out}.pkl'))
print('saved', len(G), 'features; extra-phone rows: train', (ext & ~te).sum(), 'test', (ext & te).sum(), ' new-run rows', (epoch == 1).sum())
if base:
    y0 = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
    y = np.load(os.path.join(CACHE, f'y_{base}.npy')).copy()
    keep = ext & ~te & (y0 > 0)
    print('labels restored on extra phones:', (y[keep] != y0[keep]).sum(), 'of', keep.sum())
    y[keep] = y0[keep]; np.save(os.path.join(CACHE, f'y_{out}.npy'), y)
