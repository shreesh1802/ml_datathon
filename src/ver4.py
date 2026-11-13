import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
al=load_all(); te=al.is_test.values; lab=al.label.values
dn=al.device_id.str[2:].astype(int).values
cls=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl').seg_class.values.astype(int)
# train-run densities per segment
m=(~te)&(dn>=422978)&(dn<=428454)
d=pd.DataFrame({'dn':dn[m],'seg':np.array(LABELS)[cls[m]]})
g=d.groupby('seg').dn.agg(['min','max','nunique']).sort_values('min'); g['density']=(g['nunique']/(g['max']-g['min']+1)).round(3); print(g.to_string())
# train: validate changepoint method on sim->ato boundary (true boundary from labels ~425022/425023)
def cp(lo,hi,p1,p2,ids):
    pres=np.zeros(hi-lo+1,bool); pres[ids[(ids>=lo)&(ids<=hi)]-lo]=True
    l1=np.where(pres,np.log(p1),np.log(1-p1)); l2=np.where(pres,np.log(p2),np.log(1-p2))
    c1=np.r_[0,np.cumsum(l1)]; c2=np.r_[0,np.cumsum(l2)]; ll=c1+(c2[-1]-c2)
    k=ll.argmax(); top=np.argsort(-ll)[:1]
    near=np.where(ll>ll.max()-2)[0]
    return lo+k, lo+near.min(), lo+near.max()
ut=np.unique(dn[~te]); uo=np.unique(dn[te])
print('TRAIN sim->ato changepoint (first ATO id), 2-loglik range:', cp(424630,426762,.712,.265,ut), ' seg boundary 425023')
print('OCT   sim->ato changepoint:', cp(473920,474279,.712,.265,uo), ' our boundary 473971')
# rows affected in October
for b in [473971,474001]:
    pass
o=te&(dn>=473971)&(dn<=474010)
x=pd.DataFrame({'dn':dn[o],'sim':al.sim_age_hours.values[o],'dag':al.device_age_days.values[o],'city':al.device_city.values[o],'amt':al.amount.values[o],'ts':al.timestamp.values[o],'snd':al.sender_id.values[o]}).sort_values(['dn','ts'])
print(x.to_string())
