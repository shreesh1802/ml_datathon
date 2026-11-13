import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS,macro_f1
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values; lab=al.label.values
dn=al.device_id.str[2:].astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
pred=decide(oof,tau); q=cls>0; pred[q]=cls[q]; pred[te]=pd.read_csv(os.path.join(SUBS,'sub_G8.csv')).label.map(L).values
P=np.array(LABELS)[pred]
mer=al.receiver_type.values=='MERCHANT'; r5=mer&(al.amount.values%500==0)
# bursts: round-amount rows at one merchant, split when the gap exceeds 3 days
d=pd.DataFrame({'i':np.where(r5)[0],'R':al.receiver_id.values[r5],'t':al.t.values[r5],'run':te[r5]}).sort_values(['run','R','t'])
newb=(d.R!=d.R.shift())|(d.run!=d.run.shift())|((d.t-d.t.shift())>3*86400); d['b']=newb.cumsum()
d['S']=al.sender_id.values[d.i.values]; d['lab']=lab[d.i.values]; d['pred']=P[d.i.values]
trio=['merchant_collusion','qr_swap','refund_abuse']
def maj(s):
    s=s[s.isin(trio)]; return s.mode().iloc[0] if len(s) else 'none'
g=d.groupby('b').agg(run=('run','first'),R=('R','first'),rows=('i','size'),senders=('S','nunique'),lab=('lab',maj),pred=('pred',maj),span=('t',lambda s:(s.max()-s.min())/3600))
g['size']=pd.cut(g.rows,[0,2,4,6,9,12,100],labels=['1-2','3-4','5-6','7-9','10-12','13+'])
a=g[~g.run&(g.lab!='none')]
print('TRAIN bursts: majority label by size'); print(pd.crosstab(a['size'],a.lab).to_string())
print('\nTRAIN bursts where majority prediction != majority label, by size:'); x=a[a.pred!=a.lab]; print(pd.crosstab(x['size'],[x.lab,x.pred]).to_string())
print('\nTRAIN accuracy of majority prediction:', (a.pred==a.lab).mean().round(3), 'bursts', len(a))
b=g[g.run]; print('\nTEST bursts: predicted type by size'); print(pd.crosstab(b['size'],b.pred).to_string())
print('\nTEST suspicious: predicted collusion with <=9 rows, refund with >=11 or <=2, qr with >=7'); 
s=b[((b.pred=='merchant_collusion')&(b.rows<=9))|((b.pred=='refund_abuse')&((b.rows>=11)|(b.rows<=2)))|((b.pred=='qr_swap')&(b.rows>=7))]; print(s.to_string())
print('\n=== 3-4 row bursts in detail (TRAIN) ===')
qm=al.qr_vpa_mismatch.values; tt=al.txn_type.values
d['qm1']=(qm[d.i.values]==1); d['qr']=(tt[d.i.values]=='QR'); d['legitlab']=(d.lab=='legit')
h=d.groupby('b').agg(qm1=('qm1','sum'),qr=('qr','sum'),nleg=('legitlab','sum'))
g2=g.join(h); a=g2[~g2.run&g2.lab.isin(['qr_swap','refund_abuse'])&g2.rows.isin([3,4])]
print(pd.crosstab([a.rows,a.senders],a.lab).to_string())
print(pd.crosstab([a.rows,np.minimum(a.qm1,3)],a.lab,rownames=['rows','mismatch flags']).to_string())
print('span hours by label:'); print(a.groupby('lab').span.describe().round(1).to_string())
b=g2[g2.run&g2.rows.isin([3,4])]; print('TEST 3-4 row bursts:'); print(pd.crosstab([b.rows,np.minimum(b.qm1,3)],b.pred,rownames=['rows','mismatch flags']).to_string()); print(pd.crosstab([b.rows,b.senders],b.pred).to_string())
