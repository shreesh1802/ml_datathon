import numpy as np,pandas as pd
src=open('scan1.py').read(); exec(src[:src.index("leg=pred==0")])
dn=al.device_id.str[2:].astype(int).values; tt=al.txn_type.values; rt=al.receiver_type.values
ex=np.isin(amt,[1950.,1990.,1999.,4950.,4990.,4999.,9950.,9990.,9999.,24950.,24990.,24999.])
kind=np.where(ex,'exact',np.where(~ai,'decimal',np.where(amt%10!=0,'whole_nd10',np.where(amt%500==0,'x500','x10'))))
newp=np.where(al.new_payee.values==1,'new','old')
ordp=dn<420000
rows=[]
for k in range(1,K):
    m=(pred==k)&ordp&tr
    for nm,keys in [('tt+rt+kind',[tt,rt,kind]),('tt+kind+newp',[tt,kind,newp])]:
        d=pd.DataFrame({f'k{i}':a[m] for i,a in enumerate(keys)}); d['ok']=(y[m]==k); d['lg']=(y[m]==0)
        g=d.groupby([c for c in d.columns if c.startswith('k')]).agg(rows=('ok','size'),correct=('ok','mean'),legit=('lg','mean'))
        for idx,r in g[(g.rows>=30)&(g.correct<tau[k]*0.6)].iterrows():
            mt=(pred==k)&ordp&te
            for i,a in enumerate(keys): mt=mt&(a==idx[i])
            rows.append((LABELS[k],round(tau[k],3),nm,'/'.join(idx),int(r.rows),round(r.correct,2),round(r.legit,2),int(mt.sum())))
print(pd.DataFrame(rows,columns=['predicted','keep if above','cell type','cell','train rows','correct','legit','test rows']).to_string(index=False))
