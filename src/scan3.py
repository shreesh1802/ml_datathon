import numpy as np,pandas as pd
src=open('scan1.py').read(); exec(src[:src.index("leg=pred==0")])
dn=al.device_id.str[2:].astype(int).values; tt=al.txn_type.values; p2p=al.receiver_type.values=='P2P'
ex=np.isin(amt,[1950.,1990.,1999.,4950.,4990.,4999.,9950.,9990.,9999.,24950.,24990.,24999.])
kind=np.where(ex,'exact phishing amt',np.where(~ai,'decimal',np.where(amt%10!=0,'whole, not x10',np.where((amt%1000>=900),'x10, 900-990 band','x10, other'))))
for cname in ['phishing_collect','fake_cashback']:
    k=L[cname]; m=(pred==k)&(dn<420000)
    d=pd.DataFrame({'tt':tt[m&tr],'kind':kind[m&tr],'y':y[m&tr]})
    g=d.groupby(['tt','kind']).y.agg(rows='size',correct=lambda s:(s==k).mean(),legit=lambda s:(s==0).mean(),other=lambda s:((s>0)&(s!=k)).mean()).round(2)
    dt=pd.DataFrame({'tt':tt[m&te],'kind':kind[m&te]}).groupby(['tt','kind']).size().rename('test rows')
    print('\npredicted',cname,'(train, ordinary phones) | F/2 keep-threshold',round(tau[k],3)); print(g.join(dt).fillna(0).to_string())
# the x10 900-band cell in detail: amount values
m=(pred==L['phishing_collect'])&(dn<420000)&tr&(kind=='x10, 900-990 band')
print(pd.DataFrame({'last3':(amt[m]%1000).astype(int),'ok':y[m]==L['phishing_collect'],'legit':y[m]==0}).groupby('last3').agg(rows=('ok','size'),correct=('ok','mean'),legit=('legit','mean')).round(2).to_string())
