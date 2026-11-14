"""Export a small table for studying lookalike events: every row the G9 blend predicts as fraud, plus every row on a
fraud-block phone, with raw columns, account and merchant attributes, the given label (train) and the block class."""
import os,numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,DATA,ROOT
al=load_all(); te=al.is_test.values
pred=np.load(f'{CACHE}/pred_G9raw.npy'); cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
keep=(pred>0)|(cls>0)
d=al.loc[keep].drop(columns=['ts','t'],errors='ignore').copy()
d['split']=np.where(te[keep],'test','train'); d['predicted']=np.array(LABELS)[pred[keep]]
d['block_class']=np.where(cls[keep]>0,np.array(LABELS)[np.maximum(cls[keep],0)],'')
ac=pd.read_csv(os.path.join(DATA,'accounts.csv')).set_index('account_id'); me=pd.read_csv(os.path.join(DATA,'merchants.csv')).set_index('merchant_id')
for side,col in [('snd','sender_id'),('rcv','receiver_id')]:
    j=ac.reindex(d[col].values)
    for c in ['home_city','bank_code','kyc_level','opened_on','os']: d[f'{side}_{c}']=j[c].values
j=me.reindex(d.receiver_id.values); d['mer_category']=j.category.values; d['mer_city']=j.city.values; d['mer_onboarded_on']=j.onboarded_on.values
d.loc[d.split=='test','label']=''
d=d.drop(columns=['is_test']).sort_values(['split','timestamp'])
out=os.path.join(ROOT,'lookalike_rows.csv'); d.to_csv(out,index=False)
print(d.shape, round(os.path.getsize(out)/1e6,1),'MB'); print(d.split.value_counts().to_dict()); print(list(d.columns))
