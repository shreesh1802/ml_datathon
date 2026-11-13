import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; Kn=len(LABELS)
tr=(~te)&(mo<'2026-10'); lab=al.label.values; KC=LABELS.index('merchant_collusion')
y=al.label.map({l:i for i,l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(Kn,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl').seg_class.values.astype(int)
pred=decide(oof,tau); q=cls>0; pred[q]=cls[q]
mer=al.receiver_type.values=='MERCHANT'; a=al.amount.values
c5=mer&(al.merchant_category.values!='lending')&np.isin(a,[1000.,2000.,2500.,5000.,10000.])
oth500=mer&(al.merchant_category.values!='lending')&(a%500==0)&~c5
print('TRAIN labels on the 5-amount set:'); print(pd.Series(lab[c5&tr]).value_counts().to_dict())
print('TRAIN other multiples of 500:', pd.Series(lab[oth500&tr]).value_counts().to_dict(), '| test other500', (oth500&te).sum(), 'test c5', (c5&te).sum())
rng=np.random.RandomState(0)
for sub in [1.0,0.5]:
  tot=np.zeros(12); base=[];new=[]
  for m in ['2026-07','2026-08','2026-09']:
    mm=tr&(mo==m)
    if sub<1: mm=mm&(rng.rand(len(al))<sub)
    d=pd.DataFrame({'r':al.receiver_id.values[mm],'s':al.sender_id.values[mm],'c5':c5[mm],'mer':mer[mm]})
    mt=d.r.map(d[d.mer].groupby('r').size()).fillna(0).values; mc=d.r.map(d[d.c5].groupby('r').size()).fillna(0).values; sc=d.s.map(d[d.c5].groupby('s').size()).fillna(0).values
    t1=d.c5.values&(((mc>=2)&(mc/np.maximum(mt,1)>=.1))|(sc>=2)); t2=d.c5.values&(mc==1)&(mt<=3)
    p=pred[mm]; yy=y[mm]; lb=lab[mm]
    for nm,t in [('tier1',t1),('tier2',t2)]:
        rec=t&(p==0)
        print(f'sub{sub} {m} {nm}: rows {t.sum()} (labels: {pd.Series(lb[t]).value_counts().head(4).to_dict()}) | model said legit: {rec.sum()} -> their labels {pd.Series(lb[rec]).value_counts().to_dict()}')
    p1=p.copy(); p1[t1&(p==0)]=KC; p2=p1.copy(); p2[t2&(p==0)]=KC
    f0=macro_f1(yy,p,per_class=True); f1=macro_f1(yy,p1,per_class=True); f2=macro_f1(yy,p2,per_class=True)
    print(f'   macro {f0.mean():.4f} -> W1 {f1.mean():.4f} -> W2 {f2.mean():.4f} | collusion F1 {f0[KC]:.3f} -> {f1[KC]:.3f} -> {f2[KC]:.3f}')
