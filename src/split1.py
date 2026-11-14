import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS
pd.set_option('display.width',250); pd.set_option('display.max_rows',80)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; lab=al.label.values
G=np.full(len(al),'',object); G[te]=pd.read_csv(os.path.join(SUBS,'sub_G10.csv')).label.values
dn=al.device_id.str[2:].astype(int).values
a=(~te)&(mo=='2026-10')
print('TRAIN rows dated October:',a.sum(),'labels',pd.Series(lab[a]).value_counts().to_dict())
print('  by day',pd.Series(al.timestamp.str[8:10].values[a]).value_counts().sort_index().to_dict())
pl=np.load(f'{CACHE}/oof_E1.npy')[:,0]
print('  model p_legit>0.9 (ordinary-looking):',(pl[a]>0.9).sum(),'| on block phones',(dn[a]>=420000).sum())
b=te&(mo=='2026-11')
print('TEST rows dated November:',b.sum(),'| G10 predictions',pd.Series(G[b]).value_counts().to_dict())
print('  by day',pd.Series(al.timestamp.str[8:10].values[b]).value_counts().sort_index().to_dict())
cols=['timestamp','sender_id','receiver_id','receiver_type','device_id','amount','txn_type','merchant_category','new_payee','label']
print(al[a].sort_values('timestamp')[cols].head(40).to_string(index=False))
# last days of September in train / last days of October in test: fraud share by day
d=al.timestamp.str[:10].values
x=pd.DataFrame({'d':d[~te],'f':lab[~te]!='legit'}).groupby('d').f.agg(['size','mean']); print(x.tail(14).round(4).T.to_string())
y=pd.DataFrame({'d':d[te],'p':G[te]!='legit'}).groupby('d').p.agg(['size','mean']); print(y.head(6).round(4).T.to_string()); print(y.tail(12).round(4).T.to_string())
