import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS,macro_f1,Win
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
pred=decide(oof,tau); q=cls>0; pred[q]=cls[q]; pred[te]=pd.read_csv(os.path.join(SUBS,'sub_G8.csv')).label.map(L).values
amt=al.amount.values; p2p=al.receiver_type.values=='P2P'; mer=~p2p; tt=al.txn_type.values; newp=al.new_payee.values==1
ai=amt%1==0; nd10=ai&(amt%10!=0); 
t=al.t.values.astype(np.int64); S=pd.factorize(al.sender_id)[0].astype(np.int64); WS=Win(S,t)
snd1=WS.sum((nd10&(amt>10)).astype(float),S,t,-86400,86400,'a'); snd7=WS.sum((nd10&(amt>10)).astype(float),S,t,-7*86400,7*86400,'a')
ex=np.isin(amt,[1950.,1990.,1999.,4950.,4990.,4999.,9950.,9990.,9999.,24950.,24990.,24999.]); l900=np.isin(amt,np.arange(900.,990.,10))
sig={'p2p & nd10 & amt>=500':p2p&nd10&(amt>=500),'p2p & sender has >=2 nd10 in +-1d':p2p&(snd1>=2),'merchant & nd10>10 & sender >=2 in +-7d':mer&nd10&(amt>10)&(snd7>=2),
 'p2p & exact phishing amount':p2p&ex,'p2p COLLECT & 900..980':p2p&(tt=='COLLECT')&l900,'p2p COLLECT new payee >=500 nd10 (no under flag)':p2p&(tt=='COLLECT')&newp&(amt>=500)&nd10&~((amt%1000)>=900),
 'p2p QR >=500 nd10':p2p&(tt=='QR')&(amt>=500)&nd10,'p2p QR >=500 div10':p2p&(tt=='QR')&(amt>=500)&ai&~nd10}
for nm,m in sig.items():
    a=m&tr; fr=a&(y>0); miss=fr&(pred==0); fp=a&(y==0)&(pred>0); wrongtype=fr&(pred>0)&(pred!=y)
    b=m&te
    print(f'{nm:50s} train rows {a.sum():6d} fraud-labelled {fr.sum():6d} | model: caught {(fr&(pred>0)).sum():6d}, MISSED {miss.sum():4d}, wrong type {wrongtype.sum():4d}, legit-labelled called fraud {fp.sum():4d} | test rows {b.sum():5d}, G8 calls legit {(b&(pred==0)).sum():4d}')
# P2P QR type split
m=p2p&(tt=='QR')&(amt>=500)&tr
for nm,mm in [('nd10',m&nd10),('div10',m&ai&~nd10)]:
    print(nm,'labels',pd.Series(np.array(LABELS)[y[mm]]).value_counts().head(4).to_dict(),'| preds',pd.Series(np.array(LABELS)[pred[mm]]).value_counts().head(4).to_dict())
mt=p2p&(tt=='QR')&(amt>=500)&te&nd10; print('test p2p QR nd10 G8 preds',pd.Series(np.array(LABELS)[pred[mt]]).value_counts().to_dict())
