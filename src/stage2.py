"""Second-stage inputs: what did the first model say about the rows around this one?
usage: python src/stage2.py TAG   -> cache/feats_s2_TAG.pkl
Train rows use out-of-fold probabilities (their month was never seen by the model that scored them)."""
import os, sys, time
import numpy as np, pandas as pd
from common import load_all, Win, CACHE, LABELS
tag = sys.argv[1]
al = load_all(); n = len(al); K = len(LABELS)
P = np.load(os.path.join(CACHE, f'oof_{tag}.npy')).astype(np.float64); te = al.is_test.values
P[te] = np.load(os.path.join(CACHE, f'test_{tag}.npy'))
late = (~te) & (al.timestamp.str[:7].values >= '2026-10')   # labelled rows dated October: score them like test rows is not possible; use zeros
t = al.t.values.astype(np.int64)
codes, _ = pd.factorize(pd.concat([al.sender_id, al.receiver_id], ignore_index=True))
S = codes[:n].astype(np.int64); R = codes[n:].astype(np.int64); Dv = pd.factorize(al.device_id)[0].astype(np.int64)
NE = codes.max() + 1; PR = pd.factorize(S * NE + R)[0].astype(np.int64)
H, Dd = 3600, 86400
G = {}
short = [l[:4] for l in LABELS]
groups = [('snd1d', S, S, Dd), ('snd2h', S, S, 2 * H), ('rcv6h', R, R, 6 * H), ('rcv1d', R, R, Dd), ('dev1h', Dv, Dv, H), ('pair7d', PR, PR, 7 * Dd)]
wins = {}
for nm, key, q, w in groups:
    W = wins.setdefault(id(key), Win(key, t))
    cnt = W.count(q, t, -w, w) - 1
    G[f's2_{nm}_n'] = np.minimum(cnt, 30)
    for k in range(1, K):
        s = W.sum(P[:, k], q, t, -w, w, f'p{k}') - P[:, k]
        G[f's2_{nm}_{short[k]}'] = np.where(cnt > 0, s / np.maximum(cnt, 1), -1)
    s = W.sum(1 - P[:, 0], q, t, -w, w, 'pf') - (1 - P[:, 0])
    G[f's2_{nm}_fraudsum'] = s
# money flow neighbours: the receiver's outgoing rows after this one, the sender's incoming rows before this one
WSo = wins[id(S)]; WRi = wins[id(R)]
for k in [LABELS.index('mule_account'), LABELS.index('layering_chain'), LABELS.index('fake_cashback'), LABELS.index('phishing_collect')]:
    c = WSo.count(R, t, 1, 2 * H); G[f's2_rcvout2h_{short[k]}'] = np.where(c > 0, WSo.sum(P[:, k], R, t, 1, 2 * H, f'p{k}') / np.maximum(c, 1), -1)
    c = WRi.count(S, t, -2 * H, -1); G[f's2_sndin2h_{short[k]}'] = np.where(c > 0, WRi.sum(P[:, k], S, t, -2 * H, -1, f'p{k}') / np.maximum(c, 1), -1)
    # reverse direction: rows where the receiver pays the sender (cashback bait)
c = WSo.count(R, t, 1, 2 * H); G['s2_rcvout2h_fraud'] = np.where(c > 0, WSo.sum(1 - P[:, 0], R, t, 1, 2 * H, 'pf') / np.maximum(c, 1), -1)
c = WRi.count(S, t, -2 * H, -1); G['s2_sndin2h_fraud'] = np.where(c > 0, WRi.sum(1 - P[:, 0], S, t, -2 * H, -1, 'pf') / np.maximum(c, 1), -1)
for k in range(K):
    G[f's2_self_{short[k]}'] = P[:, k]
out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in G.items()})
out.to_pickle(os.path.join(CACHE, f'feats_s2_{tag}.pkl')); print('saved', out.shape)
