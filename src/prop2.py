import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
J=pd.read_pickle(f'{CACHE}/prop_links.pkl'); J['dt']=(J.t-J.st).abs()
sh=np.array([l[:6] for l in LABELS])
d=J[(J.raw>0)&(J.raw!=J.scls)]
a=d[~d.te]
print('by time gap (train, row-link pairs): label==block / label==model / n')
for lo,hi in [(0,600),(600,3600),(3600,6*3600),(6*3600,86400)]:
    g=a[(a.dt>=lo)&(a.dt<hi)]; print(f'  {lo:6d}-{hi:6d}s: {(g.y==g.scls).sum():4d} / {(g.y==g.raw).sum():4d} / {len(g)}')
print('by (block type, model type), train, gap<6h:'); g=a[a.dt<6*3600]
x=g.assign(blk=sh[g.scls],mdl=sh[g.raw],okb=(g.y==g.scls),okm=(g.y==g.raw)).groupby(['blk','mdl']).agg(n=('i','size'),label_block=('okb','sum'),label_model=('okm','sum'))
print(x.to_string())
t_=d[d.te&(d.dt<6*3600)]; print('TEST candidates gap<6h:'); print(t_.assign(blk=sh[t_.scls],mdl=sh[t_.raw]).groupby(['blk','mdl']).i.nunique().to_string())
# also seeds agreeing: number of block seeds voting per target row (conflicts?)
for win in [3600,6*3600,86400]:
    g=d[d.dt<win].sort_values('dt').drop_duplicates('i')
    a=g[~g.te]; print(f'window {win}: train rows flipped {len(a)} correct {(a.y==a.scls).sum()} wrong(model right) {(a.y==a.raw).sum()} | test rows {g.te.sum()}')
