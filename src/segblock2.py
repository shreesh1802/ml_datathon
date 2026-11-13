"""Segment extra-phone numbers into ordered fraud-type blocks. usage: python src/segblock.py STAGE1_TAG OUT_NAME

The simulator numbers extra phones run by run: fraud-event phones first, one fraud type after another in a fixed
order, then ordinary second phones.  We recover the blocks with a left-to-right dynamic programme (a hidden Markov
chain that can only move forward through the block order):
  1. On the training run, segment using the given labels -> each phone's block.
  2. From those blocks, measure how a block-blind model (STAGE1_TAG, out-of-fold) labels phones of each block, and
     how many rows a phone of each block has.
  3. Segment both runs again using only the block-blind predictions and row counts.  Step 3 is identical for
     training and October phones, so the downstream model sees the same kind of (imperfect) block feature in both.
No block boundary or test label is typed in by hand."""
import os, sys
import numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
tag, out = sys.argv[1], sys.argv[2]
al = load_all(); n = len(al); K = len(LABELS); L = {l: i for i, l in enumerate(LABELS)}
te = al.is_test.values; month = al.timestamp.str[:7].values
P = np.load(os.path.join(CACHE, f'oof_{tag}.npy')).astype(np.float64); P[te] = np.load(os.path.join(CACHE, f'test_{tag}.npy'))
dn = al.device_id.str[2:].astype(np.int64).values
ext = dn >= 420000
train_max = dn[~te & (month < '2026-10')].max()
y0 = al.label.map(L).fillna(-1).astype(int).values
# block order as it appears in the training run (checked in probe8/probe9): shared bot phones first
ORDER = ['velocity_bot', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover', 'phishing_collect',
         'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion', 'legit']
ST = np.array([L[o] for o in ORDER]); S = len(ST)

def viterbi(logE):   # logE: devices x states; states may be skipped, never revisited
    N = len(logE); dp = np.empty((N, S)); bp = np.zeros((N, S), dtype=np.int16)
    dp[0] = logE[0]
    for i in range(1, N):
        pm = np.maximum.accumulate(dp[i - 1]); arg = np.zeros(S, dtype=np.int16); best = 0
        for s in range(1, S):
            if dp[i - 1, s] > dp[i - 1, best]: best = s
            arg[s] = best
        dp[i] = pm + logE[i]; bp[i] = arg
    st = np.empty(N, dtype=np.int16); st[-1] = dp[-1].argmax()
    for i in range(N - 1, 0, -1): st[i - 1] = bp[i, st[i]]
    return st

d = pd.DataFrame({'dn': dn[ext], 'y': y0[ext], 'te': te[ext], **{f'p{k}': P[ext, k] for k in range(K)}})
g = d.groupby('dn')
dev = pd.DataFrame({'rows': g.size(), 'e': (g.size().index.values > train_max).astype(int)})
pm = g[[f'p{k}' for k in range(K)]].mean().values; dev['pred'] = pm.argmax(1)
# given label per phone: the commonest fraud label on it, else legit, -1 if the phone has no labelled row
def given(v):
    v = v[v >= 0]
    if len(v) == 0: return -1
    f = v[v > 0]
    return np.bincount(f).argmax() if len(f) else 0
dev['given'] = g.y.agg(lambda s: given(s.values))
dev['cnt'] = np.minimum(dev.rows.values, 4) - 1
# missing phone numbers just before this phone (fraud blocks have gaps, each with its own density; legit has none)
dev['gap'] = np.minimum(np.r_[0, np.diff(dev.index.values) - 1], 30)
dev.loc[dev.e.ne(dev.e.shift()).values, 'gap'] = 0
# gap after the phone: a legit phone is never followed by a missing number
dev['gapn'] = np.minimum(np.r_[np.diff(dev.index.values) - 1, 0], 30)
dev.loc[dev.e.ne(dev.e.shift(-1)).values, 'gapn'] = 0
# 1. training run, given labels
e0 = dev[dev.e == 0]
EA = np.full((S, K), 0.03 / (K - 2))
for s in range(S):
    k = ST[s]
    if k == 0: EA[s] = 0.002 / (K - 1); EA[s, 0] = 0.998
    else: EA[s, k] = 0.85; EA[s, 0] = 0.12
obs = e0.given.values; logE = np.where(obs[:, None] >= 0, np.log(EA[:, np.maximum(obs, 0)].T), 0.0)
st_true = viterbi(logE)
print('training run blocks (first/last device number, phones):')
for s in range(S):
    q = st_true == s
    if q.any(): print(f'  {s:2d} {ORDER[s]:20s} {e0.index.values[q].min()} - {e0.index.values[q].max()}  phones {q.sum()}')
# 2. how the block-blind model sees each block
E2 = np.full((S, K), 0.5); EC = np.full((S, 4), 0.5)
np.add.at(E2, (st_true, e0.pred.values), 1); np.add.at(EC, (st_true, e0.cnt.values), 1)
E2 /= E2.sum(1, keepdims=True); EC /= EC.sum(1, keepdims=True)
DEN = np.array([(1 + (st_true == s).sum()) / (2 + (st_true == s).sum() + e0.gap.values[st_true == s].sum()) for s in range(S)]); DEN = np.clip(DEN, 0.01, 0.999)
print('block density (share of phone numbers present):', dict(zip([o[:6] for o in ORDER], DEN.round(3))))
GW = float(os.environ.get('GAPW', '1'))
def gapE(de):
    g = np.outer(de.gap.values, np.log(1 - DEN)) + np.log(DEN)[None, :]
    g[:, S - 1] = de.gapn.values * np.log(1 - DEN[S - 1]) + np.log(DEN[S - 1])
    return GW * g
print('\nP(block-blind prediction | block), rows = block, cols = legit + fraud types:'); print(pd.DataFrame(E2.round(2), index=[f'{s}:{o[:6]}' for s, o in enumerate(ORDER)], columns=[l[:5] for l in LABELS]).to_string())
# 3. segment both runs from predictions + row counts only
state = pd.Series(-1, index=dev.index); pos = pd.Series(np.nan, index=dev.index); size = pd.Series(np.nan, index=dev.index)
for e in [0, 1]:
    de = dev[dev.e == e]
    if len(de) == 0: continue
    st = viterbi(np.log(E2[:, de.pred.values].T) + np.log(EC[:, de.cnt.values].T) + gapE(de))
    state[de.index] = st
    for s in np.unique(st):
        q = st == s; idx = de.index[q]; pos[idx] = np.arange(q.sum()) / max(q.sum() - 1, 1); size[idx] = q.sum()
    print(f'\nrun {e} blocks from predictions:')
    for s in np.unique(st):
        q = st == s; print(f'  {s:2d} {ORDER[s]:20s} {de.index.values[q].min()} - {de.index.values[q].max()}  phones {q.sum()}  rows {de.rows.values[q].sum()}')
    if e == 0: print('  agreement with label-based blocks:', round((st == st_true).mean(), 4), ' on fraud blocks:', round((st == st_true)[st_true < S - 1].mean(), 4))
stt = state.reindex(dn).fillna(-1).astype(int).values; cls = np.where(stt >= 0, ST[np.maximum(stt, 0)], -1)
G = {'dev_extra': ext.astype(np.float32), 'seg_state': np.where(ext, stt, -1), 'seg_class': np.where(ext, cls, -1),
     'seg_pos': np.where(ext, pos.reindex(dn).values, np.nan), 'seg_size': np.where(ext, np.minimum(size.reindex(dn).values, 2000), np.nan),
     'seg_fraud': np.where(ext, (cls > 0), np.nan), 'dev_rows': np.where(ext, np.minimum(dev.rows.reindex(dn).values, 10), np.nan)}
for k in range(1, K): G[f'seg_is_{LABELS[k][:5]}'] = np.where(ext, (cls == k), np.nan)
pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in G.items()}).to_pickle(os.path.join(CACHE, f'feats_seg_{out}.pkl'))
print('\nsaved', len(G), 'features')
