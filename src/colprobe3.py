import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; Kn=len(LABELS)
tr=(~te)&(mo<'2026-10')
y=al.label.map({l:i for i,l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(Kn,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl').seg_class.values.astype(int)
p=decide(oof,tau); q=cls>0; p[q]=cls[q]
rows=[]
for m in ['2026-07','2026-08','2026-09']:
    mm=tr&(mo==m); rows.append(macro_f1(y[mm],p[mm],per_class=True))
f=macro_f1(y[tr],p[tr],per_class=True)
probe={'mule_account':.541,'account_takeover':.511,'sim_swap':.524,'phishing_collect':.518,'fake_cashback':.517,'merchant_collusion':.300}
d=pd.DataFrame(rows,columns=LABELS,index=['jul','aug','sep']).T; d['all']=f; d['oct_probe']=pd.Series(probe); d['diff']=d.oct_probe-d['all']
print(d.round(3).to_string()); print('macro',f.mean().round(4))
