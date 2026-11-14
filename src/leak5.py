import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS,macro_f1
al=load_all(); te=al.is_test.values; lab=al.label.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
pred=decide(oof,tau); q=cls>0; pred[q]=cls[q]; pred[te]=pd.read_csv(os.path.join(SUBS,'sub_G8.csv')).label.map(L).values
sets={'sender+receiver+amount':['sender_id','receiver_id','amount'],'sender+amount+balance':['sender_id','amount','balance_before'],'amount+balance':['amount','balance_before'],'balance+ip':['balance_before','ip_risk_score'],
 'receiver+timestamp':['receiver_id','timestamp'],'sender+sim_age':['sender_id','sim_age_hours'],'sender+balance':['sender_id','balance_before'],'sender+dag':['sender_id','device_age_days'],'sender+ip':['sender_id','ip_risk_score'],
 'receiver+amount+day':['receiver_id','amount','d'],'receiver+balance':['receiver_id','balance_before'],'sender+login+pin':['sender_id','login_to_txn_sec','pin_retries'],'device+dag':['device_id','device_age_days'],'device+sim':['device_id','sim_age_hours']}
al['d']=al.timestamp.str[:10]
for nm,c in sets.items():
    n=al.groupby(c,dropna=False).txn_id.transform('size').values; dup=n>1
    a=dup&tr&(pred==0); b=dup&te&(pred==0)
    print(f'{nm:24s} dup rows {dup.sum():6d} | predicted legit: train {a.sum():6d}, of which fraud-labelled {np.mean(y[a]>0) if a.any() else 0:.3f} | test predicted legit {b.sum()}')
