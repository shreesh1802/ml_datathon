import numpy as np,pandas as pd
src=open('scan5.py').read(); exec(src[:src.index("months=['2026-07'")])
k=L['qr_swap']; m=(pred==k)&ordp
d=pd.DataFrame({'rt':A['rt'][m&tr],'kind':A['kind'][m&tr],'size':A['size'][m&tr],'qm':A['qm'][m&tr],'ok':y[m&tr]==k,'lg':y[m&tr]==0})
print(d.groupby(['rt','kind']).agg(n=('ok','size'),correct=('ok','mean'),legit=('lg','mean')).round(2).to_string())
x=d[(d.rt=='MERCHANT')&(d.kind=='dec')]; print(x.groupby(['size','qm']).agg(n=('ok','size'),correct=('ok','mean'),legit=('lg','mean')).round(2).to_string())
print('test rows predicted qr_swap, MERCHANT, decimal:',pd.Series(A['size'][m&te&(A['rt']=='MERCHANT')&(A['kind']=='dec')]).value_counts().to_dict())
