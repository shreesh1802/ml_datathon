import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,Win
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
dn=al.device_id.str[2:].astype(int).values; pool=(dn>=420000)&(dn<422978)
t=al.t.values.astype(np.int64); D=pd.factorize(al.device_id)[0].astype(np.int64); S=pd.factorize(al.sender_id)[0].astype(np.int64); R=pd.factorize(al.receiver_id)[0].astype(np.int64)
cat=al.merchant_category.values; a=al.amount.values
ge=np.isin(cat,['ecommerce','gaming'])&(al.txn_type.values=='PAY')&(a<600)
def cells(nm,c,bins,base):
    for lo,hi in bins:
        m=base&(c>=lo)&(c<hi)&~pool
        e=(m&tr).sum()/6; o=(m&te).sum()
        print(f'{nm} [{lo},{hi}): expected {e:8.1f}  test {o:6d}  z {(o-e)/np.sqrt(e+1):6.1f}   bot-labelled share train {np.mean(lab[m&tr]=="velocity_bot") if (m&tr).sum() else 0:.2f}')
bins=[(2,3),(3,4),(4,6),(6,10),(10,30),(30,9999)]
gi=np.where(ge)[0]
for key,kn in [(D,'device'),(S,'sender')]:
    W=Win(key[gi],t[gi])
    for w,wn in [(3600,'1h'),(6*3600,'6h'),(86400,'1d'),(7*86400,'7d')]:
        c=W.count(key,t,-w,w); cells(f'small ecom/gaming PAY per {kn} +-{wn}',c,bins,ge); print()
# distinct senders per device over whole month (non-pool)
for nm,m in [('train Sep',tr&(mo=='2026-09')),('test',te)]:
    g=pd.DataFrame({'d':D[m&~pool],'s':S[m&~pool]}).groupby('d').s.nunique()
    print(nm,'distinct senders per non-pool device:',g.value_counts().sort_index().head(8).to_dict(),'max',g.max())
