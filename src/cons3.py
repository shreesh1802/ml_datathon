import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
oof=np.load(f'{CACHE}/oof_E1.npy'); pl=oof[:,0]
S=al.sender_id.values; R=al.receiver_id.values; D=al.device_id.values
leg=tr&(lab=='legit')&(pl>0.9)
rng=np.random.RandomState(0)
print('share of ordinary fraud among fraud labels:', round((tr&(lab!='legit')&(pl>0.9)).sum()/(tr&(lab!='legit')).sum(),3))
for k in range(1,12):
    c=LABELS[k]; pat=tr&(lab==c)&(oof[:,k]>0.5); ordk=tr&(lab==c)&(pl>0.9)
    ent=set(S[pat])|set(R[pat]); dev=set(D[pat])
    def rate(m):
        return np.mean([ (s in ent) or (r in ent) for s,r in zip(S[m],R[m])]), np.mean([d in dev for d in D[m]]), np.mean([s in ent for s in S[m]]), np.mean([r in ent for r in R[m]])
    li=np.where(leg)[0]; li=rng.choice(li,20000,replace=False); lm=np.zeros(len(al),bool); lm[li]=True
    a=rate(ordk); b=rate(lm)
    print(f'{c:20s} pat {pat.sum():5d} ord {ordk.sum():5d} | ord any-entity {a[0]:.3f} dev {a[1]:.3f} snd {a[2]:.3f} rcv {a[3]:.3f} | legit {b[0]:.3f} {b[1]:.3f} {b[2]:.3f} {b[3]:.3f}')
