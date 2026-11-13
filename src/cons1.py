import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
t=al.t.values.astype(float)
dbirth=t/86400-al.device_age_days.values          # days
sbirth=t/3600-al.sim_age_hours.values             # hours
abirth=t/86400-al.sender_account_age_days.values
d=pd.DataFrame({'dev':al.device_id.values,'snd':al.sender_id.values,'db':dbirth,'sb':sbirth,'ab':abirth})
d['db_med']=d.groupby('dev').db.transform('median'); d['sb_med']=d.groupby('snd').sb.transform('median'); d['ab_med']=d.groupby('snd').ab.transform('median')
d['n_dev']=d.groupby('dev').db.transform('size'); d['n_snd']=d.groupby('snd').sb.transform('size')
rd=(d.db-d.db_med).abs().values; rs=(d.sb-d.sb_med).abs().values; ra=(d.ab-d.ab_med).abs().values
print('dev resid quantiles', np.nanquantile(rd,[.1,.25,.5,.75,.9,.95,.99]).round(2))
print('sim resid quantiles', np.nanquantile(rs,[.1,.25,.5,.75,.9,.95,.99]).round(2))
print('acct resid quantiles', np.nanquantile(ra,[.1,.25,.5,.75,.9,.95,.99]).round(2))
oof=np.load(f'{CACHE}/oof_E1.npy'); ordf=tr&(lab!='legit')&(oof[:,0]>0.9); pat=tr&(lab!='legit')&(oof[:,0]<=0.5); leg=tr&(lab=='legit')
ok=(d.n_dev.values>=5)
for nm,r,th in [('dev',rd,1.0),('sim',rs,24.0),('acct',ra,1.5)]:
    for g,m in [('legit',leg),('ordinary fraud',ordf),('patterned fraud',pat)]:
        mm=m&ok; print(f'{nm} resid>{th}: {g:16s} {np.mean(r[mm]>th):.4f}  n={mm.sum()}')
