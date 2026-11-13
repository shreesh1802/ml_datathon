import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,Win
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values; K=LABELS.index('merchant_collusion')
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
P=oof.copy(); P[te]=pt
a=al.amount.values; mer=al.receiver_type.values=='MERCHANT'; isint=(a%1==0)
t=al.t.values.astype(np.int64); R=pd.factorize(al.receiver_id)[0].astype(np.int64); S=pd.factorize(al.sender_id)[0].astype(np.int64)
q=mer&isint&(a>=1000)
qi=np.where(q)[0]; W=Win(R[qi],t[qi]); D=86400
for w in [1,2,3]:
    c=W.count(R,t,-w*D,w*D)
    for th in [4,6,8,10]:
        sel=q&(c>=th)
        y=lab[sel&tr]
        print(f'w={w}d th={th} train/mo {(sel&tr).sum()/3:.0f} col% {np.mean(y=="merchant_collusion"):.2f} qr% {np.mean(y=="qr_swap"):.2f} legit% {np.mean(y=="legit"):.2f} test {(sel&te).sum()}  mean Pcol train {P[sel&tr,K].mean():.2f} test {P[sel&te,K].mean():.2f}')
