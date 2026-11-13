import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,Win
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
oof=np.load(f'{CACHE}/oof_E1.npy'); K=2
dn=al.device_id.str[2:].astype(int).values; pool=(dn>=420000)&(dn<422978)
pat=tr&(lab=='velocity_bot')&(oof[:,K]>0.5)
print('patterned bot rows/mo', pat.sum()/3, 'in pool', (pat&pool).sum()/3, 'not pool', (pat&~pool).sum()/3)
d=al[pat]
print('cat',d.merchant_category.value_counts(normalize=True).round(2).head(5).to_dict(),'txn',d.txn_type.value_counts(normalize=True).round(2).to_dict())
print('amount q',d.amount.quantile([.1,.5,.9]).round(0).to_dict())
# pool usage: rows per pool phone per burst
t=al.t.values.astype(np.int64); D=pd.factorize(al.device_id)[0].astype(np.int64); S=pd.factorize(al.sender_id)[0].astype(np.int64)
WD=Win(D,t); WS=Win(S,t)
c10=WD.count(D,t,-600,600); s10=WS.count(S,t,-600,600)
print('pool rows: dev count in +-10m  train', np.quantile(c10[pool&tr],[.1,.25,.5,.75,.9]), ' test', np.quantile(c10[pool&te],[.1,.25,.5,.75,.9]))
# burst cells: device-level and sender-level bursts, expected (train/mo * .5) vs test
mer=al.receiver_type.values=='MERCHANT'
for nm,c in [('device rows in +-10m',c10),('sender rows in +-10m',s10)]:
    for lo,hi in [(3,4),(4,6),(6,10),(10,20),(20,999)]:
        m=(c>=lo)&(c<hi)
        print(f'{nm} [{lo},{hi}): train/mo {(m&tr).sum()/3:8.1f} (bot-labelled {(m&tr&(lab=="velocity_bot")).sum()/3:7.1f}, pool {(m&tr&pool).sum()/3:7.1f})  test {(m&te).sum():6d} (pool {(m&te&pool).sum()})')
