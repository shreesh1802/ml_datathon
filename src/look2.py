import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,Win
pd.set_option('display.width',260)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
pred=np.load(f'{CACHE}/pred_G9raw.npy'); dn=al.device_id.str[2:].astype(int).values
t=al.t.values.astype(np.int64); amt=al.amount.values
codes=pd.factorize(np.r_[al.sender_id.values,al.receiver_id.values])[0]; n=len(al); S=codes[:n].astype(np.int64); R=codes[n:].astype(np.int64)
cand=np.where((pred==L['layering_chain'])|(pred==L['mule_account']))[0]     # rows predicted layering or mule (train+test)
d=pd.DataFrame({'i':cand,'S':S[cand],'R':R[cand],'t':t[cand],'a':amt[cand],'run':te[cand]})
j=d.merge(d,left_on=['R','run'],right_on=['S','run'],suffixes=('','_n'))
j=j[(j.t_n>j.t)&(j.t_n-j.t<=3600)&(j.a_n/j.a>0.85)&(j.a_n/j.a<=1.0)]
# union-find components
par={i:i for i in cand}
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for a,b in zip(j.i.values,j.i_n.values): par[f(a)]=f(b)
comp=np.array([f(i) for i in cand]); size=pd.Series(comp).map(pd.Series(comp).value_counts()).values
csz=np.zeros(n,int); csz[cand]=size
keep=np.zeros(n); 
m=tr&(pred==L['layering_chain'])&(dn<420000)
print('rows predicted layering (train, ordinary phones) by chain length: label counts')
x=pd.crosstab(np.minimum(csz[m],8),np.array(LABELS)[y[m]]); x['mule share of mule+layering']=(x.get('mule_account',0)/(x.get('mule_account',0)+x.get('layering_chain',0))).round(2); print(x[['layering_chain','mule_account','legit','mule share of mule+layering']].to_string())
print('by first-hop amount of the chain (max amount in chain):')
mx=pd.Series(amt[cand]).groupby(comp).transform('max').values; cm=np.zeros(n); cm[cand]=mx
b=pd.cut(cm[m],[0,5000,20000,50000,80000,120000,1e9])
x=pd.crosstab(b,np.array(LABELS)[y[m]]); x['mule share']=(x.get('mule_account',0)/(x.get('mule_account',0)+x.get('layering_chain',0))).round(2); print(x[['layering_chain','mule_account','legit','mule share']].to_string())
# retention ratio per hop
r=np.full(n,np.nan); r[j.i_n.values]=(j.a_n/j.a).values
for nm,k in [('layering label',L['layering_chain']),('mule label',L['mule_account'])]:
    q=m&(y==k)&~np.isnan(r); print(nm,'hop retention quantiles',np.quantile(r[q],[.05,.25,.5,.75,.95]).round(4),'| amount is whole rupees',np.mean(amt[m&(y==k)]%1==0).round(3),'| gap to next hop median (s)')
mt=te&(pred==L['layering_chain'])&(dn<420000); print('TEST rows predicted layering by chain length:',pd.Series(np.minimum(csz[mt],8)).value_counts().sort_index().to_dict())
np.save(f'{CACHE}/chain_size.npy',csz); np.save(f'{CACHE}/chain_max.npy',cm)
