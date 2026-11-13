"""How were the ordinary-looking fraud rows generated? Tests: (1) clustering in time around patterned fraud of the same
class, (2) exact copies of values from other rows, (3) repeated (sender, receiver) pairs or senders with patterned fraud."""
import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS, Win
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
al = load_all(); n = len(al); te = al.is_test.values; month = al.timestamp.str[:7].values; tr = ~te & (month < '2026-10')
y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
yc = np.load(os.path.join(CACHE, 'y_c2.npy'))
ext = al.device_id.str[2:].astype(int).values >= 420000
ordn = tr & (y > 0) & (yc == 0) & ~ext; patt = tr & (y > 0) & (yc > 0)
rng = np.random.default_rng(0); leg = tr & (y == 0) & (yc == 0) & ~ext & (rng.random(n) < 0.05)
t = al.t.values.astype(np.int64)
print(f'ordinary fraud rows {ordn.sum()}  patterned {patt.sum()}  legit sample {leg.sum()}')
# (1) time clustering: patterned rows of class k within +-w of the row (any entity), ordinary-k vs legit
print('\n(1) mean count of patterned rows of the same class within +-w seconds')
rows = []
for k in range(1, len(LABELS)):
    W = Win(np.zeros(patt.sum() if False else int(((yc == k) & patt).sum()), dtype=np.int64), t[(yc == k) & patt])
    z0 = np.zeros(n, dtype=np.int64)
    r = [LABELS[k]]
    for w in [10, 60, 600]:
        c = W.count(z0, t, -w, w)
        r += [round(c[ordn & (y == k)].mean(), 4), round(c[leg].mean(), 4)]
    rows.append(r)
print(pd.DataFrame(rows, columns=['class', 'ord_10s', 'leg_10s', 'ord_60s', 'leg_60s', 'ord_600s', 'leg_600s']).to_string(index=False))
# exact same timestamp as another row
vc = pd.Series(t).value_counts(); same_t = pd.Series(t).map(vc).values - 1
print('\nshare of rows sharing their exact second with another row: ordinary', round((same_t[ordn] > 0).mean(), 4), ' legit', round((same_t[leg] > 0).mean(), 4), ' patterned', round((same_t[patt] > 0).mean(), 4))
# (2) copied values
print('\n(2) share of rows whose value appears in at least one OTHER row')
for c in ['amount', 'balance_before', 'sim_age_hours', 'device_age_days', 'ip_risk_score']:
    v = al[c].values
    nonint = (v % 1 != 0) if c != 'ip_risk_score' else np.ones(n, bool)
    cnt = pd.Series(v).map(pd.Series(v).value_counts()).values - 1
    print(f'  {c:18s} ordinary {np.mean(cnt[ordn & nonint] > 0):.4f}  legit {np.mean(cnt[leg & nonint] > 0):.4f}  patterned {np.mean(cnt[patt & nonint] > 0):.4f}')
for cols in [['amount', 'balance_before'], ['sender_id', 'amount'], ['sender_id', 'receiver_id', 'amount'], ['receiver_id', 'amount'], ['sim_age_hours', 'device_age_days']]:
    k = pd.util.hash_pandas_object(al[cols], index=False).values
    cnt = pd.Series(k).map(pd.Series(k).value_counts()).values - 1
    print(f'  {"+".join(cols):30s} ordinary {np.mean(cnt[ordn] > 0):.4f}  legit {np.mean(cnt[leg] > 0):.4f}  patterned {np.mean(cnt[patt] > 0):.4f}')
# (3) the label of the nearest-in-time row with the same sender / receiver / device (other rows only)
print('\n(3) share whose previous/next row on the same entity (within 30 days, other month allowed) is fraud-labelled')
lab_any = (y > 0) & tr
for ent in ['sender_id', 'receiver_id', 'device_id']:
    code = pd.factorize(al[ent])[0].astype(np.int64); W = Win(code, t)
    jp = W.prv(code, t); jn = W.nxt(code, t)
    hit = ((jp >= 0) & lab_any[np.maximum(jp, 0)] & (t - t[np.maximum(jp, 0)] < 30 * 86400)) | ((jn >= 0) & lab_any[np.maximum(jn, 0)] & (t[np.maximum(jn, 0)] - t < 30 * 86400))
    print(f'  {ent:12s} ordinary {hit[ordn].mean():.4f}  legit {hit[leg].mean():.4f}  patterned {hit[patt].mean():.4f}')
