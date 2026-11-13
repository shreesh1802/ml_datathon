import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,Win,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; Kn=len(LABELS)
tr=(~te)&(mo<'2026-10'); K=LABELS.index('merchant_collusion')
y=al.label.map({l:i for i,l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
P=oof.copy(); P[te]=pt
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(Kn,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
base=decide(P,tau)
cls=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl').seg_class.values.astype(int)
def score(pred,mask):  # collusion F1 and macro on rows
    return macro_f1(y[mask],pred[mask],per_class=True)
def ov(pred):
    p=pred.copy(); q=cls>0; p[q]=cls[q]; return p
b=ov(base); fb=score(b,tr)
print('base: col F1 %.4f macro %.4f  test col %d'%(fb[K],fb.mean(),(b[te]==K).sum()))
for m in [1.2,1.4,1.6,1.8,2.0,2.5]:
    t2=tau.copy(); t2[K]=tau[K]/m; p=ov(decide(P,t2)); f=score(p,tr)
    print(f'mult {m}: col F1 {f[K]:.4f} macro {f.mean():.4f} pred/mo {(p[tr]==K).sum()/3:.0f} test col {(p[te]==K).sum()}')
# neighbour propagation within merchant integer bursts
a=al.amount.values; mer=al.receiver_type.values=='MERCHANT'; isint=(a%1==0)
t=al.t.values.astype(np.int64); R=pd.factorize(al.receiver_id)[0].astype(np.int64)
q=mer&isint&(a>=100)
qi=np.where(q)[0]; W=Win(R[qi],t[qi])
from features2 import win_min
