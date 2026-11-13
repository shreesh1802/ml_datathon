"""G1 with thresholds from October per-class F1 (public probes): tau_k = F_oct/2 for probed classes."""
import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; Kn=len(LABELS)
tr=(~te)&(mo<'2026-10')
y=al.label.map({l:i for i,l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(Kn,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
probe={'mule_account':.541,'account_takeover':.511,'sim_swap':.524,'phishing_collect':.518,'fake_cashback':.517,'merchant_collusion':.300}
t2=tau.copy()
for c,v in probe.items(): t2[LABELS.index(c)]=v/2
cls=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl').seg_class.values.astype(int)[te]
def ov(p): p=p.copy(); q=cls>0; p[q]=cls[q]; return p
a=ov(decide(pt,tau)); b=ov(decide(pt,t2))
print('rows changed', (a!=b).sum()); print(pd.crosstab(np.array(LABELS)[a[a!=b]],np.array(LABELS)[b[a!=b]]))
# training-side check of same tau change
fa=macro_f1(y[tr],decide(oof[tr],tau)); fb=macro_f1(y[tr],decide(oof[tr],t2)); print('train macro', round(fa,4), '->', round(fb,4))
pd.DataFrame({'id':al.txn_id[te].values,'label':np.array(LABELS)[b]}).to_csv(os.path.join(SUBS,'sub_G2oct.csv'),index=False)
