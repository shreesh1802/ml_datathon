import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
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
cls=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl').seg_class.values.astype(int)
def ov(p,c):
    p=p.copy(); q=c>0; p[q]=c[q]; return p
base=ov(decide(pt,tau),cls[te])
G=pd.read_csv('../submissions/sub_G1.csv').label.map({l:i for i,l in enumerate(LABELS)}).values
print('reproduces G1:', (base==G).mean())
# sensitivity: per class, rows of test with p_k/tau_k in [0.7,1.4]
for k in range(1,Kn):
    r=pt[:,k]/tau[k]
    print(f'{LABELS[k]:20s} tau {tau[k]:.3f} test rows r in[.7,1): {((r>=.7)&(r<1)).sum():4d}  [1,1.4): {((r>=1)&(r<1.4)).sum():4d}  ; oof/mo same: {(((oof[tr,k]/tau[k])>=.7)&((oof[tr,k]/tau[k])<1)).sum()/3:.0f} {(((oof[tr,k]/tau[k])>=1)&((oof[tr,k]/tau[k])<1.4)).sum()/3:.0f}')
