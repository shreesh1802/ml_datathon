import numpy as np,pandas as pd,os
from common import load_all,CACHE,LABELS,DATA
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; lab=al.label.values
tr=(~te)&(mo<'2026-10'); oof=np.load(f'{CACHE}/oof_E1.npy'); pl=oof[:,0]
leg=tr&(lab=='legit'); ordf=tr&(lab!='legit')&(pl>0.9); pat=tr&(lab!='legit')&(pl<0.5)
ac=pd.read_csv(os.path.join(DATA,'accounts.csv')).set_index('account_id')
op=pd.to_datetime(ac.opened_on).reindex(al.sender_id.values).values
days=(al.ts.values.astype('datetime64[D]')-op.astype('datetime64[D]')).astype(int)
res=al.sender_account_age_days.values-days
hr=al.ts.dt.hour.values
print('resid distribution: legit',pd.Series(res[leg]).value_counts(normalize=True).round(4).to_dict()); print('  ordinary fraud',pd.Series(res[ordf]).value_counts(normalize=True).round(4).to_dict()); print('  patterned fraud',pd.Series(res[pat]).value_counts(normalize=True).round(4).to_dict()); print('  test',pd.Series(res[te]).value_counts(normalize=True).round(4).to_dict())
print('\nshare resid != 0 by hour: legit / ordinary fraud / patterned')
for h in range(24):
    print(h, round(np.mean(res[leg&(hr==h)]!=0),3), round(np.mean(res[ordf&(hr==h)]!=0),3), round(np.mean(res[pat&(hr==h)]!=0),3))
