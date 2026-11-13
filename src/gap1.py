"""What were the missing fraud-block phone numbers allocated to? Compare allocated IDs with rows, phones and class totals."""
import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; lab=al.label.values; mo=al.timestamp.str[:7].values
dn=al.device_id.str[2:].astype(int).values
seg=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl'); st=seg.seg_state.values.astype(int); cls=seg.seg_class.values.astype(int)
m=(~te)&(dn>=423001)&(dn<=428454)
d=pd.DataFrame({'dn':dn[m],'st':st[m],'cls':cls[m]})
g=d.groupby('st').agg(cls=('cls','first'),lo=('dn','min'),hi=('dn','max'),phones=('dn','nunique'),rows=('dn','size'))
g['class']=[LABELS[c] for c in g.cls]; g['allocated']=g.hi-g.lo+1
tot=pd.Series(lab[~te]).value_counts(); g['class_rows']=[tot[c] for c in g['class']]
g['present/alloc']=(g.phones/g.allocated).round(3); g['alloc/class_rows']=(g.allocated/g.class_rows).round(4); g['phones/class_rows']=(g.phones/g.class_rows).round(4); g['rows/phone']=(g.rows/g.phones).round(2); g['alloc/block_rows']=(g.allocated/g.rows).round(2)
print(g[['class','allocated','phones','rows','class_rows','present/alloc','alloc/class_rows','phones/class_rows','rows/phone','alloc/block_rows']].to_string(index=False))
lg=(~te)&(dn>=428455)&(dn<473632); print('legit extra phones / legit rows:', round(len(np.unique(dn[lg]))/(lab[~te]=='legit').sum(),4))
