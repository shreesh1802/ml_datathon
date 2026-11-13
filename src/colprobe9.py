import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',260); pd.set_option('display.max_columns',40)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values; K=2
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
for tg in ['E1','M1']:
    p=np.load(f'{CACHE}/test_{tg}.npy')[:,2]; print(tg, np.histogram(p,[0,.02,.05,.1,.15,.18,.2,.256,.3,.5,.8,1])[0])
print('oof', (np.histogram(oof[tr,2],[0,.02,.05,.1,.15,.18,.2,.256,.3,.5,.8,1])[0]/3).round())
P=np.zeros(len(al)); P[te]=pt[:,2]; P[tr]=oof[tr,2]
dn=al.device_id.str[2:].astype(int).values
s=te&(P>=.15)&(P<.256)
d=al[s]; print(len(d)); print('device ranges', pd.cut(dn[s],[0,420000,422978,473632,10**7],right=False).value_counts().to_dict())
print(d[['timestamp','sender_id','receiver_id','receiver_type','device_id','amount','txn_type','merchant_category','sim_age_hours','device_age_days','login_to_txn_sec','pin_retries','new_payee']].head(30).to_string())
print('senders', d.sender_id.nunique(), 'devices', d.device_id.nunique())
print('top probs of others for these rows', pd.DataFrame(pt[s[te]],columns=LABELS).mean().round(3).to_dict())
