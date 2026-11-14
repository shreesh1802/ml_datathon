import numpy as np,pandas as pd,os,sys
from common import load_all,CACHE,LABELS,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
def run(tags):
    oof=sum(np.load(f'{CACHE}/oof_{t}.npy') for t in tags)/len(tags); pt=sum(np.load(f'{CACHE}/test_{t}.npy') for t in tags)/len(tags); P=oof.copy(); P[te]=pt
    def decide(p,tau):
        r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
    tau=np.full(K,.25)
    for it in range(6):
        f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
    p=decide(P,tau); q=cls>0; p[q]=cls[q]; return p,macro_f1(y[tr],p[tr],per_class=True)
res={}
for tags in sys.argv[1:]:
    p,f=run(tags.split(',')); res[tags]=(p,f)
    print(f'{tags:16s} macro-F1 {f.mean():.4f} by month {[round(macro_f1(y[tr&(mo==h)],p[tr&(mo==h)]),4) for h in ["2026-07","2026-08","2026-09"]]} | test fraud predictions {(p[te]>0).sum()}')
print(pd.DataFrame({k:v[1].round(4) for k,v in res.items()},index=LABELS).to_string())
ks=list(res); a,b=res[ks[0]][0],res[ks[-1]][0]
d=tr&(a!=b); print(f'train rows differing {ks[0]} vs {ks[-1]}: {d.sum()} | {ks[-1]} right {(b[d]==y[d]).sum()} | {ks[0]} right {(a[d]==y[d]).sum()}')
d=te&(a!=b); print('test rows differing:',d.sum()); print(pd.crosstab(np.array(LABELS)[a[d]],np.array(LABELS)[b[d]],rownames=[ks[0]],colnames=[ks[-1]]).to_string())
