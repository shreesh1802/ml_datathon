import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
al=load_all(); te=al.is_test.values; lab=al.label.values
dn=al.device_id.str[2:].astype(int).values; sn=al.sender_id.str[2:].astype(int).values
idx=np.arange(len(al)); nt=(~te).sum()
# train: position of legit extra phones, sorted by id -> see step structure
m=(~te)&(dn>=428455)&(dn<473632)
d=pd.DataFrame({'pos':idx[m],'id':dn[m]}).sort_values('id')
# rolling min/max of pos over id windows of 500
d['b']=(d.id-428455)//1500
g=d.groupby('b').pos.agg(['min','max']); print(g.to_string())
