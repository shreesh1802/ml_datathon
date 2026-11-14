import numpy as np,pandas as pd,os
from common import load_all,CACHE,LABELS
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; lab=al.label.values
tr=(~te)&(mo<'2026-10'); oof=np.load(f'{CACHE}/oof_E1.npy'); pl=oof[:,0]
ordf=(tr&(lab!='legit')&(pl>0.9))
d=al[tr&(pl>0.9)].copy(); d['f']=(d.label!='legit').astype(int); base=d.f.mean(); print('rows',len(d),'ordinary-fraud share',round(base,4))
d['day']=d.timestamp.str[:10]; d['minute']=d.timestamp.str[:16]; d['hourk']=d.timestamp.str[:13]
keys={'sender':['sender_id'],'receiver':['receiver_id'],'device':['device_id'],'pair':['sender_id','receiver_id'],'sender+day':['sender_id','day'],'receiver+day':['receiver_id','day'],
 'timestamp':['timestamp'],'minute':['minute'],'amount':['amount'],'balance':['balance_before'],'sim_age':['sim_age_hours'],'device_age':['device_age_days'],'ip':['ip_risk_score'],'login+pin+amount':['login_to_txn_sec','pin_retries','amount'],
 'receiver+hour':['receiver_id','hourk'],'city+minute':['device_city','minute'],'acct_age+day':['sender_account_age_days','day']}
for nm,k in keys.items():
    g=d.groupby(k).f; n=g.transform('size'); s=g.transform('sum')
    m=(n>1)
    # for fraud rows in groups of size>1: share of the OTHER rows in the group that are fraud
    fr=m&(d.f==1); lg=m&(d.f==0)
    of=((s-1)/(n-1))[fr].mean(); ol=(s/(n-1))[lg].mean()
    print(f'{nm:18s} groups>1 cover {m.mean():.2f} of rows | other rows fraud share: given fraud {of:.4f}, given legit {ol:.4f}, lift {of/max(ol,1e-9):.2f}  (n fraud rows {fr.sum()})')
