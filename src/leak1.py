import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; lab=al.label.values
tr=(~te)&(mo<'2026-10')
oof=np.load(f'{CACHE}/oof_E1.npy'); pl=oof[:,0]
leg=tr&(lab=='legit'); ordf=tr&(lab!='legit')&(pl>0.9); pat=tr&(lab!='legit')&(pl<0.5)
v=al.txn_id.str[2:].apply(lambda s:int(s,16)).values.astype(np.uint64)
print('id length', al.txn_id.str.len().value_counts().to_dict(), 'dup ids', al.txn_id.duplicated().sum())
bits=np.array([(v>>np.uint64(b))&np.uint64(1) for b in range(32)]).astype(float)
print('bit means  legit   ', bits[:,leg].mean(1).round(3)); print('bit means  ordinary', bits[:,ordf].mean(1).round(3)); print('bit means  patterned', bits[:,pat].mean(1).round(3)); print('bit means  test    ', bits[:,te].mean(1).round(3))
for m in [3,5,7,9,10,11,13,16,17,97,100,256,1000]:
    r=(v%np.uint64(m)).astype(int); a=np.bincount(r[leg],minlength=m)/leg.sum(); b=np.bincount(r[ordf],minlength=m)/ordf.sum()
    chi=ordf.sum()*((b-a)**2/np.maximum(a,1e-12)).sum(); print(f'mod {m}: chi2 {chi:.1f} (dof {m-1})')
# value range and order vs time
print('value quantiles legit',np.quantile(v[leg].astype(float),[.01,.5,.99]).round(0),' ordinary',np.quantile(v[ordf].astype(float),[.01,.5,.99]).round(0))
print('corr(id value, time)=',np.corrcoef(v.astype(float),al.t.values.astype(float))[0,1].round(4))
# other raw-string quirks: timestamp format, amount string decimals, device/sender string lengths
import os
from common import DATA
raw=pd.read_csv(os.path.join(DATA,'train.csv'),dtype=str,nrows=400000)
fr=raw.label!='legit'
for c in raw.columns:
    if c in('label','txn_id'): continue
    L=raw[c].str.len()
    a=L[~fr].value_counts(normalize=True,dropna=False); b=L[fr].value_counts(normalize=True,dropna=False)
    d=(a.reindex(a.index.union(b.index)).fillna(0)-b.reindex(a.index.union(b.index)).fillna(0)).abs().sum()/2
    print(f'{c:26s} string-length distribution distance fraud vs legit: {d:.3f}')
