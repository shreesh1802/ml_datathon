"""Size-aware decision rule for macro-F1. Assigning a row to class k changes F1_k by about 2(p_k - F_k/2)/(pred_k + true_k),
so choose the class with the largest positive (p_k - F_k/2)/N_k, where N_k = predicted + true rows of class k.
The old rule compares p_k/(F_k/2), which ignores class size."""
import sys,numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
tags=sys.argv[1].split(',')
oof=sum(np.load(f'{CACHE}/oof_{t}.npy') for t in tags)/len(tags); pt=sum(np.load(f'{CACHE}/test_{t}.npy') for t in tags)/len(tags)
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
def old(p,fit):
    def decide(p,tau):
        r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
    tau=np.full(K,.25)
    for it in range(6):
        f=macro_f1(y[fit],decide(oof[fit],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
    return decide(p,tau),tau
def new(p,fit,power=1.0,iters=8):
    tau=np.full(K,.25); N=np.ones(K); T=np.bincount(y[fit],minlength=K).astype(float)
    def decide(p,tau,N):
        g=(p[:,1:]-tau[1:])/N[1:]**power; k=g.argmax(1)+1; return np.where(g.max(1)>0,k,0)
    for it in range(iters):
        d=decide(oof[fit],tau,N); f=macro_f1(y[fit],d,per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]; N=np.bincount(d,minlength=K)+T; N=N/N[1:].mean()
    return decide(p,tau,N),tau,N
def ov(p,m): p=p.copy(); q=cls[m]>0; p[q]=cls[m][q]; return p
months=['2026-07','2026-08','2026-09']
print('fit on all three months:')
a=ov(old(oof[tr],tr)[0],tr); print('  old rule',round(macro_f1(y[tr],a),4))
for pw in [0.5,1.0]:
    b=ov(new(oof[tr],tr,pw)[0],tr); f=macro_f1(y[tr],b,per_class=True); print(f'  size-aware (power {pw})',round(f.mean(),4),'| rows changed',(a!=b).sum())
print('fit on two months, scored on the third:')
for pw in [0.5,1.0]:
    so=[];sn=[]
    for h in months:
        fit=tr&(mo!=h); ev=tr&(mo==h)
        so.append(macro_f1(y[ev],ov(old(oof[ev],fit)[0],ev))); sn.append(macro_f1(y[ev],ov(new(oof[ev],fit,pw)[0],ev)))
    print(f'  power {pw}: old {np.round(so,4)} mean {np.mean(so):.4f} | size-aware {np.round(sn,4)} mean {np.mean(sn):.4f}')
a=ov(old(oof[tr],tr)[0],tr); b=ov(new(oof[tr],tr,1.0)[0],tr); fa=macro_f1(y[tr],a,per_class=True); fb=macro_f1(y[tr],b,per_class=True)
print(pd.DataFrame({'old':fa.round(4),'size-aware':fb.round(4),'delta':(fb-fa).round(4)},index=LABELS).to_string())
d=a!=b; print(pd.crosstab(np.array(LABELS)[a[d]],np.array(LABELS)[b[d]],rownames=['old'],colnames=['new']).to_string())
