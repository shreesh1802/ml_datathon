import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import load_all,CACHE,LABELS,DATA
import os
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
oof=np.load(f'{CACHE}/oof_E1.npy')
ac=pd.read_csv(os.path.join(DATA,'accounts.csv')).set_index('account_id'); me=pd.read_csv(os.path.join(DATA,'merchants.csv')).set_index('merchant_id')
X=pd.DataFrame(index=al.index)
for side,col in [('s','sender_id'),('r','receiver_id')]:
    j=ac.reindex(al[col].values)
    for c in ['home_city','bank_code','kyc_level','os']: X[f'{side}_{c}']=j[c].astype('category').cat.codes.values
    X[f'{side}_open']=(pd.to_datetime(j.opened_on)-pd.Timestamp('2020-01-01')).dt.days.values
j=me.reindex(al.receiver_id.values)
X['m_cat']=j.category.astype('category').cat.codes.values; X['m_city']=j.city.astype('category').cat.codes.values
X['m_onb']=(pd.to_datetime(j.onboarded_on)-pd.Timestamp('2020-01-01')).dt.days.values
X['m_cat_match']=(j.category.values==al.merchant_category.values).astype(int)
X['same_bank']=(X.s_bank_code==X.r_bank_code).astype(int); X['same_city']=(X.s_home_city==X.r_home_city).astype(int)
X['aag']=al.sender_account_age_days.values; X['aag_resid']=X.aag-((al.ts-pd.Timestamp('2020-01-01')).dt.days.values-X.s_open)
X['dcity_home']=(al.device_city.values==ac.reindex(al.sender_id.values).home_city.values).astype(int)
X['dcity_mcity']=(al.device_city.values==j.city.values).astype(int)
X['app']=al.app.astype('category').cat.codes.values; X['app_os']=X.app*3+X.s_os
print(X.aag_resid.describe())
leg=tr&(lab=='legit'); ordf=tr&(lab!='legit')&(oof[:,0]>0.9)
m=leg|ordf; y=ordf[m].astype(int); Xm=X[m]; mm=mo[m]
a_=mm<'2026-09'; b_=mm=='2026-09'
rng=np.random.RandomState(0); keep=a_&((y==1)|(rng.rand(len(y))<0.1))
mdl=lgb.train(dict(objective='binary',learning_rate=.05,num_leaves=31,min_data_in_leaf=100,feature_fraction=.6,bagging_fraction=.8,bagging_freq=1,verbose=-1,metric='auc'),lgb.Dataset(Xm[keep],y[keep]),400,valid_sets=[lgb.Dataset(Xm[b_],y[b_])],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(100)])
p=mdl.predict(Xm[b_]); print('AUC Sep', roc_auc_score(y[b_],p))
print(pd.Series(mdl.feature_importance('gain'),X.columns).sort_values(ascending=False).head(8))
for c in X.columns:
    v=Xm[c].values[b_].astype(float); ok=~np.isnan(v); au=roc_auc_score(y[b_][ok],v[ok])
    if abs(au-.5)>.008: print('single',c,round(au,4))
