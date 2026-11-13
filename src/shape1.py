"""Can lookalike variants be separated from the real type by event shape? One binary model per pair, leave-one-month-out."""
import numpy as np,pandas as pd,lightgbm as lgb,os
from sklearn.metrics import roc_auc_score
from common import load_all,CACHE,LABELS,macro_f1,DATA,Win
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
dn=al.device_id.str[2:].astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2; pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
P=oof.copy(); P[te]=pt
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
raw=decide(P,tau)
keep=(raw>0)&(dn<420000)          # rows predicted as fraud on ordinary phones (train and test)
idx=np.where(keep)[0]; print('rows predicted fraud on ordinary phones:',len(idx))
F=pd.read_pickle(f'{CACHE}/feats.pkl').iloc[idx].reset_index(drop=True)
F2=pd.read_pickle(f'{CACHE}/feats2.pkl').iloc[idx].reset_index(drop=True)
X=pd.concat([F,F2],axis=1); X=X.loc[:,~X.columns.duplicated()]
drop=[c for c in X.columns if 'app_version' in c or 'ip_risk' in c]; X=X.drop(columns=drop)
# event-shape features among same-prediction rows
t=al.t.values.astype(np.int64)[idx]; S=pd.factorize(al.sender_id.values[idx])[0]; codes=pd.factorize(np.r_[al.sender_id.values[idx],al.receiver_id.values[idx]])[0]
Sc=codes[:len(idx)].astype(np.int64); Rc=codes[len(idx):].astype(np.int64); amt=al.amount.values[idx].astype(float); rw=raw[idx]
H=3600
for nm,key in [('s',Sc),('r',Rc)]:
    W=Win(key,t)
    for wn,w in [('2h',2*H),('1d',24*H)]:
        c=W.count(key,t,-w,w); X[f'ev_{nm}_n_{wn}']=c; X[f'ev_{nm}_amt_{wn}']=W.sum(amt,key,t,-w,w,'a'); X[f'ev_{nm}_amtmean_{wn}']=X[f'ev_{nm}_amt_{wn}']/np.maximum(c,1)
        X[f'ev_{nm}_before_{wn}']=W.count(key,t,-w,-1); X[f'ev_{nm}_after_{wn}']=W.count(key,t,1,w)
        X[f'ev_{nm}_int_{wn}']=W.sum((amt%1==0).astype(float),key,t,-w,w,'i')/np.maximum(c,1)
    d=pd.DataFrame({'k':key,'t':t,'a':amt})
    g=d.groupby('k'); X[f'ev_{nm}_span']=(g.t.transform('max')-g.t.transform('min')).values; X[f'ev_{nm}_tot']=g.t.transform('size').values
    X[f'ev_{nm}_amt_cv']=(g.a.transform('std')/g.a.transform('mean')).values; X[f'ev_{nm}_amt_ratio']=(d.a/g.a.transform('max')).values
    X[f'ev_{nm}_first']=(d.t-g.t.transform('min')).values
WSr=Win(Rc,t); X['ev_s_as_rcv_2h']=WSr.count(Sc,t,-2*H,2*H); WRs=Win(Sc,t); X['ev_r_as_snd_2h']=WRs.count(Rc,t,-2*H,2*H)
ac=pd.read_csv(os.path.join(DATA,'accounts.csv')).set_index('account_id')
for side,col in [('s','sender_id'),('r','receiver_id')]:
    j=ac.reindex(al[col].values[idx]); X[f'{side}_kyc']=j.kyc_level.astype('category').cat.codes.values; X[f'{side}_bank']=j.bank_code.astype('category').cat.codes.values
    X[f'{side}_os']=j.os.astype('category').cat.codes.values; X[f'{side}_open']=(pd.to_datetime(j.opened_on)-pd.Timestamp('2020-01-01')).dt.days.values
X=X.astype(np.float32); yy=y[idx]; mm=mo[idx]; tt=te[idx]
print('features',X.shape[1])
pairs=[('velocity_bot','mule_account'),('account_takeover','sim_swap'),('mule_account','layering_chain'),('fake_cashback','phishing_collect'),('phishing_collect','fake_cashback'),('loan_app_extortion','phishing_collect'),('refund_abuse','qr_swap'),('merchant_collusion','qr_swap'),('sim_swap','account_takeover')]
res={}
for a,b in pairs:
    la,lb=L[a],L[b]; m=(~tt)&(rw==lb)&np.isin(yy,[la,lb])&(mm<'2026-10'); z=(yy[m]==la).astype(int); Xm=X[m]; mon=mm[m]
    pr=np.zeros(m.sum())
    for h in ['2026-07','2026-08','2026-09']:
        a_=mon!=h; b_=mon==h
        mdl=lgb.train(dict(objective='binary',learning_rate=.03,num_leaves=15,min_data_in_leaf=15,feature_fraction=.3,bagging_fraction=.8,bagging_freq=1,lambda_l2=5,verbose=-1),lgb.Dataset(Xm[a_],z[a_]),300)
        pr[b_]=mdl.predict(Xm[b_])
    auc=roc_auc_score(z,pr); o=np.argsort(-pr); top=z[o][:max(1,z.sum()//2)].mean()
    imp=pd.Series(mdl.feature_importance('gain'),X.columns).sort_values(ascending=False).head(5).index.tolist()
    print(f'{a[:14]:14s} looks like {b[:14]:14s}: variant {z.sum():4d} of {len(z):5d} | AUC {auc:.3f} | precision in top half-variant-count {top:.2f} | top features {imp}')
    res[(a,b)]=(m,pr,z)
pd.to_pickle({k:(np.where(v[0])[0],v[1],v[2]) for k,v in res.items()},f'{CACHE}/shape_res.pkl'); X.to_pickle(f'{CACHE}/shape_X.pkl'); np.save(f'{CACHE}/shape_idx.npy',idx)
