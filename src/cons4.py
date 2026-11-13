import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
oof=np.load(f'{CACHE}/oof_E1.npy'); pl=oof[:,0]; t=al.t.values.astype(np.int64)
leg=tr&(lab=='legit')&(pl>0.9)
def gap(q,ref):
    ref=np.sort(ref); i=np.searchsorted(ref,q); lo=ref[np.clip(i-1,0,len(ref)-1)]; hi=ref[np.clip(i,0,len(ref)-1)]
    return np.minimum(np.abs(q-lo),np.abs(hi-q))
qs=[.05,.1,.25,.5,.75,.9]
for k in range(1,12):
    c=LABELS[k]; pat=tr&(lab==c)&(oof[:,k]>0.5); ordk=tr&(lab==c)&(pl>0.9)
    g1=gap(t[ordk],t[pat]); g0=gap(t[leg][::50],t[pat])
    print(f'{c:20s} ord {np.quantile(g1,qs).round(0)}  legit {np.quantile(g0,qs).round(0)}')
# time-of-day / day distribution of ordinary fraud vs legit
h=(t//3600)%24
print('hour dist ord  ', np.round(np.bincount(h[tr&(lab!='legit')&(pl>0.9)],minlength=24)/ (tr&(lab!='legit')&(pl>0.9)).sum(),3))
print('hour dist legit', np.round(np.bincount(h[leg],minlength=24)/leg.sum(),3))
d=t//86400
o=np.bincount(d[tr&(lab!='legit')&(pl>0.9)],minlength=93)[:92]; l=np.bincount(d[leg],minlength=93)[:92]
print('daily ord/legit ratio x1000', np.round(o/l*1000,1))
