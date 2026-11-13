import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
G=np.full(len(al),'',object); G[te]=pd.read_csv('../submissions/sub_G1.csv').label.values
tt=al.txn_type.values; qm=al.qr_vpa_mismatch.values; rt=al.receiver_type.values
m=np.isnan(qm)&(tt=='QR')
print('train labels on blank-QR:',pd.Series(lab[m&tr]).value_counts().to_dict())
print('train rt on blank-QR:',pd.crosstab(lab[m&tr],rt[m&tr]))
print('test G1 on blank-QR:',pd.crosstab(G[m&te],rt[m&te]))
print('test blank-QR by day:',pd.Series(al.timestamp.str[:10].values[m&te]).value_counts().sort_index().to_dict())
print('train blank-QR by month:',pd.Series(mo[m&~te]).value_counts().sort_index().to_dict())
# qm==1 labels in train
print('train labels qm==1:',pd.Series(lab[(qm==1)&tr]).value_counts().head(6).to_dict())
print('test G1 qm==1:',pd.Series(G[(qm==1)&te]).value_counts().to_dict())
print('qm values test', pd.Series(qm[te]).value_counts(dropna=False).to_dict(), 'train/3', (pd.Series(qm[tr]).value_counts(dropna=False)/3).round().to_dict())
