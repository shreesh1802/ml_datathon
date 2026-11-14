import numpy as np,pandas as pd,os,itertools
from common import load_all,CACHE,SUBS,LABELS,macro_f1
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E2n.npy')+np.load(f'{CACHE}/oof_M2.npy'))/2
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
T=np.bincount(y[tr],minlength=K).astype(float); N=np.ones(K); tau=np.full(K,.25)
def decide(p):
    g=(p[:,1:]-tau[1:])/N[1:]; k=g.argmax(1)+1; return np.where(g.max(1)>0,k,0)
for it in range(8):
    d=decide(oof[tr]); f=macro_f1(y[tr],d,per_class=True); tau=np.r_[1.0,np.clip(f[1:]/2,.1,.45)]; N=np.bincount(d,minlength=K)+T; N=N/N[1:].mean()
pred=decide(oof); pred[cls>0]=cls[cls>0]
G=np.zeros(len(al),int); G[te]=pd.read_csv(os.path.join(SUBS,'sub_G12.csv')).label.map(L).values; pred[te]=G[te]
amt=al.amount.values; dn=al.device_id.str[2:].astype(int).values; tt=al.txn_type.values; rt=al.receiver_type.values; hr=al.ts.dt.hour.values
ex=np.isin(amt,[1950.,1990.,1999.,4950.,4990.,4999.,9950.,9990.,9999.,24950.,24990.,24999.]); ai=amt%1==0
A={'tt':tt,'rt':rt,'kind':np.where(ex,'exact',np.where(~ai,'dec',np.where(amt%10!=0,'nd10',np.where(amt%500==0,'x500','x10')))),
   'newp':np.where(al.new_payee.values==1,'new','old'),'hour':np.where(hr<5,'night',np.where(hr>=20,'late',np.where(hr>=17,'eve','day'))),
   'size':np.where(amt<500,'<500',np.where(amt<2000,'<2k',np.where(amt<10000,'<10k','10k+'))),'qm':np.where(al.qr_vpa_mismatch.values==1,'m1',np.where(al.qr_vpa_mismatch.values==0,'m0','mNA')),
   'cat':al.merchant_category.fillna('none').values,'pin':np.where(al.pin_retries.values>0,'pin1','pin0'),'own':np.where(dn==al.sender_id.str[2:].astype(int).values,'own','oth')}
cellfix=(tt=='COLLECT')&(rt=='P2P')&(amt%10==0)&(amt%500!=0)&~ex
ordp=(dn<420000)
pred[tr&cellfix&(pred==L['phishing_collect'])&ordp]=0
months=['2026-07','2026-08','2026-09']; found=[]
for k in range(1,K):
    m=(pred==k)&ordp
    for r in (2,3):
        for keys in itertools.combinations(A,r):
            d=pd.DataFrame({c:A[c][m&tr] for c in keys}); d['ok']=(y[m&tr]==k); d['lg']=(y[m&tr]==0); d['mo']=mo[m&tr]
            g=d.groupby(list(keys)).agg(n=('ok','size'),ok=('ok','mean'),lg=('lg','mean'))
            for idx,row in g[(g.n>=30)&(g.ok<0.6*tau[k])&(g.lg>0.5)].iterrows():
                idx=idx if isinstance(idx,tuple) else (idx,)
                sel=np.ones(len(al),bool)
                for c,v in zip(keys,idx): sel&=(A[c]==v)
                pm=[np.mean(y[sel&m&(mo==h)]==k) if (sel&m&(mo==h)).sum()>=5 else np.nan for h in months]
                if np.nanmax(pm)<tau[k]: found.append((LABELS[k],'+'.join(keys),'/'.join(idx),int(row.n),round(row.ok,2),round(row.lg,2),np.round(pm,2).tolist(),int((sel&m&te).sum()),sel&m))
out=pd.DataFrame([f[:8] for f in found],columns=['predicted','keys','cell','train_n','correct','legit','correct_by_month','test_rows'])
print('threshold tau:',dict(zip([l[:6] for l in LABELS[1:]],tau[1:].round(2))))
print(out.sort_values(['predicted','train_n'],ascending=[True,False]).drop_duplicates(['predicted','train_n','test_rows']).head(60).to_string(index=False))
U=np.zeros(len(al),bool)
for f in found: U|=f[8]
p2=pred.copy(); p2[U]=0
f0=macro_f1(y[tr],pred[tr],per_class=True); f1=macro_f1(y[tr],p2[tr],per_class=True)
print(f'union of all cells: train rows {int((U&tr).sum())} (correct {int((y[U&tr]==pred[U&tr]).sum())}, legit {int((y[U&tr]==0).sum())}) | test rows {int((U&te).sum())} | train macro {f0.mean():.4f} -> {f1.mean():.4f}')
print('by month:',[ (round(macro_f1(y[tr&(mo==h)],pred[tr&(mo==h)]),4),round(macro_f1(y[tr&(mo==h)],p2[tr&(mo==h)]),4)) for h in months])
print(pd.Series(np.array(LABELS)[pred[U&te]]).value_counts().to_dict())
np.save(f'{CACHE}/scan5_union.npy',U)
