import numpy as np,pandas as pd
from common import load_all,CACHE
al=load_all(); te=al.is_test.values; lab=al.label.values
cols_sets={'all but id/time/label':['sender_id','receiver_id','device_id','amount','txn_type','device_city','sim_age_hours','device_age_days','pin_retries','login_to_txn_sec','balance_before','ip_risk_score'],
 'sender+receiver+amount':['sender_id','receiver_id','amount'],'sender+amount+balance':['sender_id','amount','balance_before'],'amount+balance':['amount','balance_before'],
 'sim+dag+ip':['sim_age_hours','device_age_days','ip_risk_score'],'balance+ip':['balance_before','ip_risk_score'],'sender+timestamp':['sender_id','timestamp'],'receiver+timestamp':['receiver_id','timestamp'],'sender+sim_age':['sender_id','sim_age_hours'],'sender+balance':['sender_id','balance_before'],'login+ip+sim':['login_to_txn_sec','ip_risk_score','sim_age_hours']}
f=np.where(te,-1,(lab!='legit').astype(int))
for nm,c in cols_sets.items():
    g=al.groupby(c,dropna=False); n=g.txn_id.transform('size').values
    dup=n>1
    print(f'{nm:26s} rows in duplicate groups: {dup.sum():7d} | train fraud share in dups {np.mean(f[dup&~te]==1) if (dup&~te).any() else 0:.4f} (base {np.mean(f[~te]==1):.4f}) | test rows in dups {(dup&te).sum()}')
