import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
dn=al.device_id.str[2:].astype(int).values
seg=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl'); cls=seg.seg_class.values.astype(int)
G=np.full(len(al),'',object); G[te]=pd.read_csv('../submissions/sub_G1.csv').label.values
pool=(dn>=420000)&(dn<=422977)
print('pool rows train by month', pd.Series(mo[pool&~te]).value_counts().sort_index().to_dict(), 'test', (pool&te).sum())
print('train pool labels', pd.Series(lab[pool&tr]).value_counts().head(5).to_dict())
print('test pool G1', pd.Series(G[pool&te]).value_counts().to_dict())
# rows by device-number range per month
bins=[0,420000,422978,473632,480000,10**7]
print(pd.crosstab(pd.cut(dn,bins,right=False),np.where(te,'test',mo)))
# October run phones: first ones
o=(dn>=473632)
d=pd.DataFrame({'dn':dn[o],'te':te[o],'G':G[o],'lab':lab[o],'seg':cls[o]})
g=d.groupby('dn').agg(n=('te','size'),G=('G',lambda s:s.value_counts().index[0]),seg=('seg','first'))
print(g.head(60).to_string())
