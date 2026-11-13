import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); lab=al.label.values
dn=al.device_id.str[2:].astype(int).values
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
G=np.full(len(al),'',object); G[te]=pd.read_csv(os.path.join(SUBS,'sub_G7.csv')).label.values
mer=al.receiver_type.values=='MERCHANT'; a=al.amount.values; r5=mer&(a%500==0)
# round-amount merchant rows: label by txn_type in training
print('TRAIN round-500 merchant rows: label x txn_type'); print(pd.crosstab(lab[r5&tr],al.txn_type.values[r5&tr]).to_string())
# seeds: block-phone rows that are themselves round-amount merchant payments
seed=(cls>0)&(dn>=423001)&r5
print('\nseeds: train',(seed&tr).sum(),'test',(seed&te).sum())
print('TRAIN seeds: block class vs label'); print(pd.crosstab(np.array(LABELS)[cls[seed&tr]],lab[seed&tr]).to_string())
print('TEST seeds: block class'); print(pd.Series(np.array(LABELS)[cls[seed&te]]).value_counts().to_dict())
# for each seed: other round-amount rows at the same merchant in the same run -> labels (train) / predictions (test)
rows=[]
R=al.receiver_id.values; t=al.t.values
for i in np.where(seed)[0]:
    m=r5&(R==R[i])&(te==te[i])&(dn<420000)&(np.abs(t-t[i])<=7*86400)
    if te[i]: rows.append(dict(run='test',seed_dev=dn[i],merchant=R[i],block=LABELS[cls[i]],seed_type=al.txn_type.values[i],n=m.sum(),others=pd.Series(G[m]).value_counts().to_dict(),types=pd.Series(al.txn_type.values[m]).value_counts().to_dict()))
    else: rows.append(dict(run='train',seed_dev=dn[i],merchant=R[i],block=LABELS[cls[i]],seed_type=al.txn_type.values[i],n=m.sum(),others=pd.Series(lab[m]).value_counts().to_dict(),types=pd.Series(al.txn_type.values[m]).value_counts().to_dict(),seed_label=lab[i]))
d=pd.DataFrame(rows); print(d[d.n>0].sort_values(['run','block']).to_string())
