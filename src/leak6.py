import numpy as np,pandas as pd
from common import load_all,CACHE
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; lab=al.label.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); pl=np.load(f'{CACHE}/oof_E1.npy')[:,0]
dn=al.device_id.str[2:].astype(int).values; sn=al.sender_id.str[2:].astype(int).values
leg=tr&(lab=='legit')&(pl>0.9); ordf=tr&(lab!='legit')&(pl>0.9)
kind=np.where(dn==sn,'own',np.where(dn<420000,'other account phone',np.where(((dn>=428455)&(dn<473632))|(dn>=474552),'legit-range extra','fraud-range extra')))
print(pd.DataFrame({'legit':pd.Series(kind[leg]).value_counts(normalize=True),'ordinary fraud':pd.Series(kind[ordf]).value_counts(normalize=True),'test':pd.Series(kind[te]).value_counts(normalize=True)}).round(4).to_string())
dag=al.device_age_days.values; aag=al.sender_account_age_days.values.astype(float); sim=al.sim_age_hours.values
own=dn==sn
for nm,m in [('own-phone rows',own),('other-account-phone rows',(dn!=sn)&(dn<420000))]:
    print('\n',nm)
    for g,q in [('legit',leg&m),('ordinary fraud',ordf&m),('test',te&m)]:
        r=dag[q]/np.maximum(aag[q],1)
        print(f'  {g:15s} n={q.sum():7d} dag quantiles {np.nanquantile(dag[q],[.05,.25,.5,.75,.95]).round(1)}  dag/acct_age quantiles {np.nanquantile(r,[.05,.25,.5,.75,.95]).round(3)}  P(dag>acct_age+1) {np.mean(dag[q]>aag[q]+1):.4f}  P(|dag-acct_age|<1) {np.mean(np.abs(dag[q]-aag[q])<1):.4f}  sim median {np.nanmedian(sim[q]):.0f}')
