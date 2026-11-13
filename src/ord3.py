import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; lab=al.label.values
dn=al.device_id.str[2:].astype(int).values; sn=al.sender_id.str[2:].astype(int).values
idx=np.arange(len(al)); nt=(~te).sum(); n=te.sum()
ch=np.where(te, 6+((idx-nt)>=n//2), np.minimum(idx*6//nt,5))
d=pd.DataFrame({'ch':ch,'t':al.timestamp.values,'sn':sn,'fraud':(lab!='legit')&~te,'pool':(dn>=420000)&(dn<422978),'blk':((dn>=422978)&(dn<=428454))|((dn>=473632)&(dn<=474551)),'lab':lab})
print(d.groupby('ch').agg(rows=('t','size'),tmin=('t','min'),tmax=('t','max'),smin=('sn','min'),smax=('sn','max'),fraud=('fraud','sum'),pool=('pool','sum'),blk=('blk','sum')).to_string())
print(pd.crosstab(d.lab,d.ch).to_string())
print(pd.crosstab(al.timestamp.str[:7].values,ch).to_string())
