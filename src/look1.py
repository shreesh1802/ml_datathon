import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
pd.set_option('display.width',260); pd.set_option('display.max_rows',200)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
pred=np.load(f'{CACHE}/pred_G9raw.npy'); dn=al.device_id.str[2:].astype(int).values
al['pred']=np.array([l[:6] for l in LABELS])[pred]; al['lab']=al.label.str[:6]
m=tr&(y==L['mule_account'])&(pred==L['layering_chain'])&(dn<420000)
ix=np.where(m)[0][:3]
cols=['timestamp','sender_id','receiver_id','amount','balance_before','new_payee','lab','pred']
for i in ix:
    s,r,t=al.sender_id.values[i],al.receiver_id.values[i],al.t.values[i]
    nb=al[(~te)&(np.abs(al.t.values-t)<6*3600)&(al.sender_id.isin([s,r])|al.receiver_id.isin([s,r]))].sort_values('timestamp')
    print(f'--- row {al.txn_id.values[i]}  {s} -> {r}'); print(nb[cols].to_string(index=False))
# summary: for mule-labelled-called-layering rows vs layering-labelled-called-layering rows: role statistics
t=al.t.values.astype(np.int64)
from common import Win
codes=pd.factorize(np.r_[al.sender_id.values,al.receiver_id.values])[0]; S=codes[:len(al)].astype(np.int64); R=codes[len(al):].astype(np.int64)
p2p=al.receiver_type.values=='P2P'; pi=np.where(p2p)[0]
WR=Win(R[pi],t[pi]); WS=Win(S[pi],t[pi])
in_s=WR.count(S,t,-7200,-1); out_s=WS.count(S,t,-7200,7200); in_r=WR.count(R,t,-7200,7200); out_r=WS.count(R,t,1,7200)
for nm,mm in [('mule label, called layering',m),('layering label, called layering',tr&(y==L['layering_chain'])&(pred==L['layering_chain'])&(dn<420000)),('mule label, called mule',tr&(y==L['mule_account'])&(pred==L['mule_account'])&(dn<420000))]:
    print(f'{nm:34s} n={mm.sum():5d} | sender got money in prior 2h: mean {in_s[mm].mean():.2f}, share>=2 {np.mean(in_s[mm]>=2):.2f} | sender out +-2h mean {out_s[mm].mean():.2f} | receiver in +-2h mean {in_r[mm].mean():.2f} share>=3 {np.mean(in_r[mm]>=3):.2f} | receiver sends on in 2h: {np.mean(out_r[mm]>=1):.2f}')
