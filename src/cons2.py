import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import load_all,CACHE,LABELS,DATA
import os
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
print(os.listdir(DATA))
oof=np.load(f'{CACHE}/oof_E1.npy')
X=pd.DataFrame(index=al.index)
a=al.amount.values; b=al.balance_before.values; t=al.t.values.astype(np.int64)
X['idx']=np.arange(len(al)); 
# position in original file order?
X['amt']=a; X['bal']=b; X['bal_amt']=b/np.maximum(a,.01); X['bal_minus']=b-a
X['amt_c']=np.round(a*100)%100; X['bal_c']=np.round(b*100)%100; X['amt_d1']=np.round(a*100)%10; X['bal_d1']=np.round(b*100)%10
X['sec']=t%60; X['minute']=(t//60)%60; X['hour']=(t//3600)%24; X['dow']=(t//86400)%7
X['dag']=al.device_age_days.values; X['aag']=al.sender_account_age_days.values; X['sim']=al.sim_age_hours.values
X['dag_aag']=X.dag/np.maximum(X.aag,1); X['sim_aag']=X.sim/np.maximum(X.aag*24,1); X['dag_gt']=(X.dag>X.aag+1).astype(int)
X['dag_d']=np.round(X.dag*10)%10; X['sim_d']=np.round(X.sim*10)%10
X['ip']=al.ip_risk_score.values; X['ip_d']=np.round(X.ip*1000)%10
X['login']=al.login_to_txn_sec.values; X['pin']=al.pin_retries.values; X['newp']=al.new_payee.values
X['qm']=al.qr_vpa_mismatch.values; X['appv']=al.app_version.values
for c in ['txn_type','receiver_type','merchant_category','app','device_city']: X[c]=al[c].astype('category').cat.codes.values
X['snd_n']=al.sender_id.str[2:].astype(int).values; X['rcv_n']=al.receiver_id.str[2:].astype(int).values; X['dev_n']=al.device_id.str[2:].astype(int).values
X['own']=(X.snd_n==X.dev_n).astype(int)
X['tx_hex']=al.txn_id.str[2:].apply(lambda s:int(s,16)).values
X['isnull_n']=al.isna().sum(1).values
leg=tr&(lab=='legit'); ordf=tr&(lab!='legit')&(oof[:,0]>0.9)
m=leg|ordf; y=ordf[m].astype(int); Xm=X[m]; mm=mo[m]
a_=mm<'2026-09'; b_=mm=='2026-09'
rng=np.random.RandomState(0); keep=a_&((y==1)|(rng.rand(len(y))<0.1))
ds=lgb.Dataset(Xm[keep],y[keep]); dv=lgb.Dataset(Xm[b_],y[b_])
mdl=lgb.train(dict(objective='binary',learning_rate=.05,num_leaves=31,min_data_in_leaf=100,feature_fraction=.6,bagging_fraction=.8,bagging_freq=1,verbose=-1,metric='auc'),ds,400,valid_sets=[dv],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(100)])
p=mdl.predict(Xm[b_]); print('AUC Sep', roc_auc_score(y[b_],p))
print(pd.Series(mdl.feature_importance('gain'),X.columns).sort_values(ascending=False).head(12))
for c in X.columns:
    v=Xm[c].values[b_]; ok=~np.isnan(v)
    au=roc_auc_score(y[b_][ok],v[ok]); 
    if abs(au-.5)>.01: print('single',c,round(au,4))
