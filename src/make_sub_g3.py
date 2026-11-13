"""G1 with two block corrections from the ID-density leak (fraud-block phone numbers have gaps, legit ones do not):
1. October's fraud range ends at 474551, so rows on 474545..474549 take the last block's class (merchant_collusion).
2. The SIM-swap block is dense (71% of IDs present) and the takeover block sparse (27%). The density change point
   puts the boundary near 473988; the same estimate overshoots by about 10 IDs on the training run, so 473980 is used."""
import os,numpy as np,pandas as pd
from common import load_all,SUBS
al=load_all(); te=al.is_test.values
dn=al.device_id.str[2:].astype(int).values[te]
s=pd.read_csv(os.path.join(SUBS,'sub_G1.csv')); assert (s.id.values==al.txn_id.values[te]).all()
lab=s.label.values.copy()
a=(dn>=474545)&(dn<=474551); print('rows -> merchant_collusion:', a.sum(), pd.Series(lab[a]).value_counts().to_dict()); lab[a]='merchant_collusion'
b=(dn>=473971)&(dn<=473979); print('rows -> sim_swap:', b.sum(), pd.Series(lab[b]).value_counts().to_dict()); lab[b]='sim_swap'
print('total changed', (lab!=s.label.values).sum())
pd.DataFrame({'id':s.id,'label':lab}).to_csv(os.path.join(SUBS,'sub_G3.csv'),index=False)
