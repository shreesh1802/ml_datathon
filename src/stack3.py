"""Do the agent's feats3 add anything on top of the current model? Second model on [E1+M1 probabilities + feats3],
leave-one-month-out, same decision rule and block override. usage: python src/stack3.py [OUT_SUB]"""
import sys,os,numpy as np,pandas as pd,lightgbm as lgb
from common import load_all,CACHE,SUBS,LABELS,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
yc=np.load(f'{CACHE}/y_d1.npy')            # cleaned training labels, as used by E1/M1
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2; pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
P=oof.copy(); P[te]=pt
F=pd.read_pickle(f'{CACHE}/feats3.pkl')
use=os.environ.get('COLS','all'); 
if use!='all': F=F[[c for c in F.columns if not c.startswith(('rcv_ring','snd_ring','rcv_was','snd_was'))]] if use=='noring' else F[[]]
X=np.hstack([P.astype(np.float32),F.values.astype(np.float32)]); names=[f'p_{l[:5]}' for l in LABELS]+list(F.columns); print('features',X.shape[1],'mode',use)
rng=np.random.RandomState(0)
keep=tr&(yc>=0)&((yc>0)|(rng.rand(len(al))<0.12))
w=np.where(yc==0,1/0.12,1.0)
par=dict(objective='multiclass',num_class=K,learning_rate=.05,num_leaves=15,min_data_in_leaf=40,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=5,verbose=-1,seed=0)
Q=np.zeros_like(P); 
for h in ['2026-07','2026-08','2026-09']:
    a=keep&(mo!=h); b=tr&(mo==h)
    m=lgb.train(par,lgb.Dataset(X[a],yc[a],weight=w[a]),200); Q[b]=m.predict(X[b])
m=lgb.train(par,lgb.Dataset(X[keep],yc[keep],weight=w[keep]),200); Q[te]=m.predict(X[te])
imp=pd.Series(m.feature_importance('gain'),names).sort_values(ascending=False); print('top new-feature importances:',imp[[n for n in imp.index if not n.startswith('p_')]].head(8).round(0).to_dict())
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
def run(PR,nm):
    def decide(p,tau):
        r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
    tau=np.full(K,.25)
    for it in range(6):
        f=macro_f1(y[tr],decide(PR[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
    p=decide(PR,tau); q=cls>0; p[q]=cls[q]
    f=macro_f1(y[tr],p[tr],per_class=True); print(f'{nm:10s} training macro-F1 {f.mean():.4f}  by month {[round(macro_f1(y[tr&(mo==h)],p[tr&(mo==h)]),4) for h in ["2026-07","2026-08","2026-09"]]}')
    return p,f
p0,f0=run(P,'current'); p1,f1=run(Q,'stacked'); p2,f2=run((P+Q)/2,'average')
print(pd.DataFrame({'current':f0.round(4),'stacked':f1.round(4),'delta':(f1-f0).round(4)},index=LABELS).to_string())
print('test rows that differ (stacked vs current):',(p0[te]!=p1[te]).sum()); d=te&(p0!=p1); print(pd.crosstab(np.array(LABELS)[p0[d]],np.array(LABELS)[p1[d]],rownames=['current'],colnames=['stacked']).to_string())
dt=tr&(p0!=p1); print('train rows that differ:',dt.sum(),'| stacked right',(p1[dt]==y[dt]).sum(),'| current right',(p0[dt]==y[dt]).sum())
np.save(f'{CACHE}/stack3_Q.npy',Q.astype(np.float32))
