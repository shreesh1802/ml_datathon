import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
G=np.full(len(al),'',object); G[te]=pd.read_csv('../submissions/sub_G1.csv').label.values
a=al.amount.values; rt=al.receiver_type.values; tt=al.txn_type.values; qm=al.qr_vpa_mismatch.values
def cmp(name,mask):
    print(f'{name:45s} train/mo {(mask&tr).sum()/3:8.1f}  test {(mask&te).sum():6d}  ratio {(mask&te).sum()/max((mask&tr).sum()/3,1):.2f}')
for c in ['velocity_bot','layering_chain','qr_swap','refund_abuse','loan_app_extortion']:
    print('==',c); print(' train label tt/rt:',pd.crosstab(tt[tr&(lab==c)],rt[tr&(lab==c)]).to_dict())
    print(' test pred tt/rt:',pd.crosstab(tt[te&(G==c)],rt[te&(G==c)]).to_dict())
cmp('P2P QR',(rt=='P2P')&(tt=='QR'))
cmp('QR mismatch==1',qm==1); cmp('QR mismatch==0',qm==0); cmp('QR mismatch NaN & tt=QR',np.isnan(qm)&(tt=='QR'))
cmp('REFUND',tt=='REFUND'); cmp('AUTOPAY',tt=='AUTOPAY'); cmp('COLLECT',tt=='COLLECT')
cmp('lending merchant',al.merchant_category.values=='lending')
