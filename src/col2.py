import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,Win,SHIFT
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
G=np.full(len(al),'',object); G[te]=pd.read_csv('../submissions/sub_G1.csv').label.values
t=al.t.values.astype(np.int64); S=pd.factorize(al.sender_id)[0].astype(np.int64); R=pd.factorize(al.receiver_id)[0].astype(np.int64)
mer=al.receiver_type.values=='MERCHANT'; a=al.amount.values
P=S*SHIFT+R; mi=np.where(mer)[0]; W=Win(P[mi],t[mi]); D=86400
c=W.count(P,t,-2*D,2*D)
kind=np.where(a%500==0,'round500',np.where(a%1==0,'integer','decimal'))
for kd in ['round500','integer','decimal']:
    for lo,hi in [(2,3),(3,4),(4,6),(6,99)]:
        m=mer&(kind==kd)&(c>=lo)&(c<hi)
        e=(m&tr).sum()/3; o=(m&te).sum()
        ll=lab[m&tr]
        print(f'{kd:9s} pair rows in +-2d [{lo},{hi}): train/mo {e:7.1f} (collusion {np.mean(ll=="merchant_collusion") if len(ll) else 0:.2f}, legit {np.mean(ll=="legit") if len(ll) else 0:.2f})  test {o:5d} ratio {o/max(e,1):.2f} | G1: col {(G[m&te]=="merchant_collusion").sum()} legit {(G[m&te]=="legit").sum()} other {((G[m&te]!="merchant_collusion")&(G[m&te]!="legit")).sum()}')
