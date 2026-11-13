"""G3 with the SIM-swap/takeover boundary moved to the density maximum-likelihood point (473988)."""
import os,numpy as np,pandas as pd
from common import load_all,SUBS,CACHE,LABELS
al=load_all(); te=al.is_test.values
dn=al.device_id.str[2:].astype(int).values[te]
s=pd.read_csv(os.path.join(SUBS,'sub_G3.csv')); lab=s.label.values.copy()
b=(dn>=473980)&(dn<=473987); print('G4 rows -> sim_swap:', b.sum(), pd.Series(lab[b]).value_counts().to_dict()); lab[b]='sim_swap'
pd.DataFrame({'id':s.id,'label':lab}).to_csv(os.path.join(SUBS,'sub_G4.csv'),index=False)
# other boundary candidates: rows that would move
for nm,lo,hi in [('473988..473992 (upper end of SIM interval)',473988,473992),('bot->layering 473818..473838',473818,473838),('mule->bot 473750..473756',473750,473756),('qr->refund 474448..474461',474448,474461),('ato->phish 474276..474280',474276,474280)]:
    m=(dn>=lo)&(dn<=hi); print(nm,'rows',m.sum(),pd.Series(lab[m]).value_counts().to_dict())
