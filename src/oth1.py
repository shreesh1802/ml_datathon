import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
pred=np.load(f'{CACHE}/pred_G9raw.npy'); dn=al.device_id.str[2:].astype(int).values; sn=al.sender_id.str[2:].astype(int).values
oth=(dn!=sn)&(dn<420000)
print('share of rows on another account\'s phone, by predicted class (train):')
for k in range(K:=len(LABELS)):
    m=tr&(pred==k); print(f'  {LABELS[k]:20s} {oth[m].mean():.3f}  (n={m.sum()})')
# phones (other-account) that carry predicted-fraud rows: how many rows, senders, and label consistency
m=tr&oth&(pred>0)
d=pd.DataFrame({'dev':dn[m],'snd':sn[m],'y':y[m],'p':pred[m],'day':al.timestamp.str[:10].values[m]})
g=d.groupby('dev').agg(rows=('y','size'),senders=('snd','nunique'),days=('day','nunique'),labels=('y',lambda s:s[s>0].nunique()),preds=('p','nunique'))
print('other-account phones carrying predicted-fraud rows:',len(g)); print(g.describe().loc[['mean','50%','max']].round(2).to_string())
print('phones with fraud rows on 2+ days:',(g.days>=2).sum(),'| with 2+ senders:',(g.senders>=2).sum(),'| with 2+ distinct fraud labels:',(g.labels>=2).sum())
# is the phone owner involved in the event (as sender/receiver of fraud rows near in time)?
rn=np.where(al.receiver_type.values=='P2P',al.receiver_id.str[2:].astype(int).values,-1)
fr=tr&(pred>0); acc_fraud=set(sn[fr])|set(rn[fr])
own_inv=np.fromiter((x in acc_fraud for x in dn[m]),bool,m.sum()); base=np.fromiter((x in acc_fraud for x in dn[tr&oth&(pred==0)][:200000]),bool,min(200000,(tr&oth&(pred==0)).sum()))
print('phone owner is itself a party to predicted fraud: fraud rows',own_inv.mean().round(3),'| legit rows',base.mean().round(3))
# lookalike pairs: does the other-phone share differ between variant and real?
for a,b in [('account_takeover','sim_swap'),('mule_account','layering_chain'),('velocity_bot','mule_account'),('fake_cashback','phishing_collect')]:
    v=tr&(pred==L[b])&(y==L[a]); r=tr&(pred==L[b])&(y==L[b]); print(f'  predicted {b[:10]}: other-phone share  label {a[:10]} {oth[v].mean():.3f} (n={v.sum()}) | label {b[:10]} {oth[r].mean():.3f} (n={r.sum()})')
