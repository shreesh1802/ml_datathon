"""What does the label noise look like? Uses out-of-fold probabilities of a trained tag."""
import os, sys
import numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
tag = sys.argv[1]
al = load_all(); oof = np.load(os.path.join(CACHE, f'oof_{tag}.npy'))
K = len(LABELS); y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
tr = (~al.is_test.values) & (al.timestamp.str[:7].values < '2026-10')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
p = oof; top = p.argmax(1); conf = p.max(1)
# 1. confident predictions vs given label
for thr in [0.5, 0.8]:
    m = tr & (top > 0) & (conf > thr)
    cm = pd.crosstab(pd.Series(np.array(LABELS)[top[m]], name='model says'), pd.Series(np.array(LABELS)[y[m]], name='label'))
    cm = cm.reindex(index=LABELS[1:], columns=LABELS, fill_value=0)
    print(f'\n== model confident (p>{thr}) fraud class vs given label (row-normalised %)'); print((cm.div(cm.sum(1), axis=0) * 100).round(1).to_string()); print('rows', cm.sum(1).to_dict())
# 2. fraud-labelled rows: how many look ordinary
fr = tr & (y > 0)
d = pd.DataFrame({'label': np.array(LABELS)[y[fr]], 'p_legit': p[fr, 0], 'p_own': p[fr, y[fr]]})
print('\n== fraud-labelled rows by p_legit bucket (share)')
print(pd.crosstab(d.label, pd.cut(d.p_legit, [0, .2, .5, .8, .9, .95, 1.0]), normalize='index').round(3).to_string())
# legit-labelled p_legit distribution for reference
lg = tr & (y == 0)
print('legit rows p_legit quantiles', np.quantile(p[lg, 0], [.001, .01, .05, .5]).round(4))
# 3. do ordinary-looking fraud rows share entities with other fraud rows?
ordinary = fr & (p[:, 0] > 0.9); patt = fr & (p[:, 0] < 0.5)
print('\nordinary-looking fraud rows', ordinary.sum(), ' patterned fraud rows', patt.sum())
for ent in ['sender_id', 'receiver_id', 'device_id']:
    e = al[ent].values
    cnt_f = pd.Series(e[fr]).value_counts(); cnt_p = pd.Series(e[patt]).value_counts()
    def rate(mask, self_in_f, self_in_p):
        nf = pd.Series(e[mask]).map(cnt_f).fillna(0).values - self_in_f
        npat = pd.Series(e[mask]).map(cnt_p).fillna(0).values - self_in_p
        return round((nf > 0).mean(), 4), round((npat > 0).mean(), 4)
    rs = np.random.default_rng(0); lsub = lg & (rs.random(len(al)) < 0.05)
    if ent == 'receiver_id':
        pp = (al.receiver_type == 'P2P').values
        print(ent, 'P2P only  ordinary-fraud:', rate(ordinary & pp, 1, 0), ' patterned:', rate(patt & pp, 1, 1), ' legit:', rate(lsub & pp, 0, 0))
        print(ent, 'MERCH only ordinary-fraud:', rate(ordinary & ~pp, 1, 0), ' patterned:', rate(patt & ~pp, 1, 1), ' legit:', rate(lsub & ~pp, 0, 0))
    else:
        print(ent, 'share with another (any fraud-labelled, patterned-fraud) row:  ordinary-fraud:', rate(ordinary, 1, 0), ' patterned:', rate(patt, 1, 1), ' legit:', rate(lsub, 0, 0))
# 4. time proximity: is an ordinary-looking fraud row near a patterned fraud row of the same sender/receiver?
