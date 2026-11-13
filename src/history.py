"""Fraud history of each account / receiver / phone / pair, from training labels in OTHER months.
usage: python src/history.py LABEL_TAG OUT_NAME
A training row never sees labels from its own month (so nothing leaks inside a burst, and the held-out-month check
stays honest).  October rows see all three training months.  Counts are divided by the number of months used."""
import os, sys
import numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
ltag, out = sys.argv[1], sys.argv[2]
al = load_all(); n = len(al); K = len(LABELS)
y = np.load(os.path.join(CACHE, f'y_{ltag}.npy')); te = al.is_test.values; month = al.timestamp.str[:7].values
MONTHS = ['2026-07', '2026-08', '2026-09']
mi = np.full(n, -1); 
for i, m in enumerate(MONTHS): mi[(month == m) & ~te] = i
lab = (mi >= 0) & (y > 0)
codes, _ = pd.factorize(pd.concat([al.sender_id, al.receiver_id], ignore_index=True)); S = codes[:n]; R = codes[n:]; NE = codes.max() + 1
Dv = pd.factorize(al.device_id)[0]; PR = pd.factorize(S.astype(np.int64) * NE + R)[0]
G = {}
def hist(src_codes, qry_codes, size, name):
    """src: who carried the label; qry: whose history we look up (same id space)."""
    M = np.zeros((3, size, K), dtype=np.float32)
    np.add.at(M, (mi[lab], src_codes[lab], y[lab]), 1)
    tot = M.sum(0)
    h = np.empty((n, K), dtype=np.float32)
    for i in range(3):
        q = mi == i; h[q] = (tot[qry_codes[q]] - M[i][qry_codes[q]]) / 2
    q = mi < 0; h[q] = tot[qry_codes[q]] / 3
    for k in range(1, K): G[f'h_{name}_{LABELS[k][:5]}'] = h[:, k]
    G[f'h_{name}_any'] = h[:, 1:].sum(1)
hist(S, S, NE, 'snd')            # sender was the sender of fraud before
hist(R, R, NE, 'rcv')            # receiver received fraud before
hist(Dv, Dv, Dv.max() + 1, 'dev')
hist(PR, PR, PR.max() + 1, 'pair')
hist(R, S, NE, 'snd_as_rcv')     # this sender received fraud before (mule, chain hop)
hist(S, R, NE, 'rcv_as_snd')     # this receiver sent fraud before
pd.DataFrame(G).to_pickle(os.path.join(CACHE, f'feats_h_{out}.pkl')); print('saved', len(G))
