import numpy as np,pandas as pd,os
from common import load_all,CACHE,DATA
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; lab=al.label.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); pl=np.load(f'{CACHE}/oof_E1.npy')[:,0]
leg=tr&(lab=='legit')&(pl>0.9); ordf=tr&(lab!='legit')&(pl>0.9)
ac=pd.read_csv(os.path.join(DATA,'accounts.csv')).set_index('account_id')
def consistency(col,key='sender_id'):
    d=pd.DataFrame({'k':al[key].values,'v':al[col].astype(str).values})
    n=d.groupby('k').v.transform('size'); same=d.groupby(['k','v']).v.transform('size')
    share=((same-1)/np.maximum(n-1,1)).values; ok=(n>1).values
    return share,ok
for col in ['app','device_city','app_version','merchant_category','txn_type','receiver_type']:
    for key in ['sender_id','device_id']:
        s,ok=consistency(col,key)
        print(f'{col:18s} by {key:10s}: share of the key\'s other rows with the same value: legit {s[leg&ok].mean():.4f}  ordinary fraud {s[ordf&ok].mean():.4f}  test {s[te&ok].mean():.4f}')
# app vs account os / bank
j=ac.reindex(al.sender_id.values)
for a,b in [('app','os'),('app','bank_code'),('app','kyc_level')]:
    ct_l=pd.crosstab(al[a].values[leg],j[b].values[leg],normalize='columns'); ct_f=pd.crosstab(al[a].values[ordf],j[b].values[ordf],normalize='columns')
    print(f'{a} x {b}: max abs diff legit vs ordinary fraud {np.abs(ct_l.values-ct_f.reindex(index=ct_l.index,columns=ct_l.columns).fillna(0).values).max():.4f}')
print(pd.crosstab(al.app.values[leg],j.os.values[leg],normalize='columns').round(3).to_string())
