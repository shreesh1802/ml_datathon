"""Scan for non-threshold structure the trees cannot build: digit patterns and exact repeats.
For each derived value, among rows the G9 model calls legit, find cells with a high fraud-labelled share."""
import numpy as np,pandas as pd,os
from common import load_all,CACHE,LABELS,macro_f1
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E2n.npy')+np.load(f'{CACHE}/oof_M2.npy'))/2; pt=(np.load(f'{CACHE}/test_E2n.npy')+np.load(f'{CACHE}/test_M2.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
P=oof.copy(); P[te]=pt; pred=decide(P,tau); q=cls>0; pred[q]=cls[q]
np.save(f'{CACHE}/pred_G9raw.npy',pred)
amt=al.amount.values; bal=al.balance_before.values; ai=amt%1==0
paise=np.round(amt*100).astype(np.int64)%100; rup=np.floor(amt).astype(np.int64)
bpaise=np.round(bal*100).astype(np.int64)%100
ctx=(al.receiver_type.values+'/'+al.txn_type.values)
t=al.t.values.astype(np.int64)
D={}
D['amt_int']=ai.astype(int); D['paise']=paise; D['paise_mod5']=paise%5; D['paise_mod10']=paise%10; D['paise_tens']=paise//10
D['rup_last']=rup%10; D['rup_last2']=rup%100; D['rup_mod5']=rup%5; D['rup_mod25']=rup%25; D['rup_mod50']=rup%50
D['bal_int']=(bal%1==0).astype(int); D['bal_paise_mod10']=bpaise%10; D['bal_paise']=bpaise
D['balrup_last']=np.floor(bal).astype(np.int64)%10
D['login']=np.minimum(al.login_to_txn_sec.values,60).astype(int); D['login_mod5']=al.login_to_txn_sec.values.astype(int)%5; D['pin']=al.pin_retries.values.astype(int)
D['sim_dec']=np.round(al.sim_age_hours.values*10).astype(np.int64)%10; D['dag_dec']=np.round(al.device_age_days.values*10).astype(np.int64)%10
D['ip_last']=np.round(al.ip_risk_score.fillna(0).values*1000).astype(np.int64)%10
D['sec']=t%60; D['aag_mod7']=al.sender_account_age_days.values.astype(int)%7
# exact repeats
g=pd.DataFrame({'S':al.sender_id.values,'R':al.receiver_id.values,'a':amt,'b':bal,'d':al.timestamp.str[:10].values})
D['same_snd_amt_day']=np.minimum(g.groupby(['S','a','d']).a.transform('size').values,4)
D['same_rcv_amt_day']=np.minimum(g.groupby(['R','a','d']).a.transform('size').values,4)
D['same_snd_amt_all']=np.minimum(g.groupby(['S','a']).a.transform('size').values,4)
D['same_rcv_amt_all']=np.minimum(g.groupby(['R','a']).a.transform('size').values,5)
D['same_snd_bal_all']=np.minimum(g.groupby(['S','b']).a.transform('size').values,3)
D['bal_eq_other_amt']=np.minimum(pd.Series(bal).map(pd.Series(amt[~ai]).value_counts()).fillna(0).values,3).astype(int)
leg=pred==0
base=np.mean(y[tr&leg]>0); print('base fraud share among rows called legit:',round(base,4), 'rows',(tr&leg).sum())
rows=[]
for nm,v in D.items():
    d=pd.DataFrame({'c':ctx[tr&leg],'v':v[tr&leg],'f':(y[tr&leg]>0)})
    for by in (['v'],['c','v']):
        a=d.groupby(by).f.agg(['size','sum']); a=a[a['size']>=40]
        a['share']=a['sum']/a['size']; a['z']=(a['sum']-base*a['size'])/np.sqrt(base*(1-base)*a['size'])
        for idx,r in a[(a.z>3.8)|(a.z<-3.8)].iterrows(): rows.append((nm,str(idx),int(r['size']),int(r['sum']),round(r['share'],3),round(r['z'],1)))
out=pd.DataFrame(rows,columns=['feature','cell','rows','fraud','share','z']).sort_values('z',ascending=False)
print(out.head(60).to_string(index=False)); print('...'); print(out.tail(15).to_string(index=False))
