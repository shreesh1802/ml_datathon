import numpy as np,pandas as pd,sys,os
sys.path.insert(0,r'C:\Users\0095\Downloads')
from common import load_all,CACHE,LABELS,DATA
import apply_winning_fixes as W
al=load_all(); te=al.is_test.values; lab=al.label.values
dn=al.device_id.str[2:].astype(int).values
cls=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl').seg_class.values.astype(int)
ut=np.unique(dn[~te]); uo=np.unique(dn[te])
def cp(lo,hi,p1,p2,ids):
    pres=np.zeros(hi-lo+1,bool); pres[ids[(ids>=lo)&(ids<=hi)]-lo]=True
    l1=np.where(pres,np.log(p1),np.log(1-p1)); l2=np.where(pres,np.log(p2),np.log(1-p2))
    c1=np.r_[0,np.cumsum(l1)]; c2=np.r_[0,np.cumsum(l2)]; ll=c1+(c2[-1]-c2)
    return ll,lo
def seginfo(mask):
    d=pd.DataFrame({'dn':dn[mask],'seg':cls[mask]}); d=d[d.seg>0]
    g=d.groupby('seg').dn.agg(['min','max','nunique']).sort_values('min'); g['dens']=g['nunique']/(g['max']-g['min']+1); return g
gt=seginfo((~te)&(dn>=422978)&(dn<=428454)); go=seginfo(te&(dn>=473632)&(dn<=474551))
ks=list(gt.index)
for a,b in zip(ks[:-1],ks[1:]):
    p1,p2=gt.dens[a],gt.dens[b]
    out=[]
    for g,ids in [(gt,ut),(go,uo)]:
        lo,hi=g['min'][a],g['max'][b]; ll,l0=cp(lo,hi,p1,p2,ids); k=ll.argmax(); cur=g['min'][b]-lo
        near=np.where(ll>ll.max()-2)[0]
        out.append(f"cur {g['min'][b]} best {l0+k} [{l0+near.min()},{l0+near.max()}] loglik gain {ll.max()-ll[cur]:.1f}")
    print(f'{LABELS[a][:8]:8s}->{LABELS[b][:8]:8s} dens {p1:.2f}->{p2:.2f} | TRAIN {out[0]} | OCT {out[1]}')
# cross-block claim on the training run
ac=pd.read_csv(os.path.join(DATA,'accounts.csv'))
m=(~te)&(al.timestamp.str[:7].values=='2026-09')
df=al[m].reset_index(drop=True); s=W.compute_pattern_overrides(df,ac)
c=cls[m]; l=df.label.map({x:i for i,x in enumerate(LABELS)}).values
x=(c>0)&(s['pat_cls']>0)&(s['pat_cls']!=c)
print('Sep block rows where audit pattern != block:', x.sum(), '| label==block', (l[x]==c[x]).sum(), '| label==pattern', (l[x]==s['pat_cls'][x]).sum(), '| label legit', (l[x]==0).sum())
x2=(c>0)&(s['pat_cls']>0)&(s['pat_cls']==c); print('pattern==block rows:', x2.sum(), 'label==block', (l[x2]==c[x2]).sum())
