import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS,macro_f1
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
dn=al.device_id.str[2:].astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
pred=decide(oof,tau); q=cls>0; pred[q]=cls[q]
pred[te]=pd.read_csv(os.path.join(SUBS,'sub_G7.csv')).label.map(L).values
mer=al.receiver_type.values=='MERCHANT'; t=al.t.values.astype(np.int64); R=al.receiver_id.values; a=al.amount.values
sh=np.array([l[:6] for l in LABELS])
seed=(cls>0)&(dn>=423001)&mer
sd=pd.DataFrame({'key':R[seed],'st':t[seed],'styp':cls[seed],'run':te[seed],'sr5':(a[seed]%500==0),'sint':(a[seed]%1==0)})
cand=np.where(~seed&(dn<420000)&mer&(pred>0))[0]
tg=pd.DataFrame({'i':cand,'key':R[cand],'t':t[cand],'cur':pred[cand],'y':y[cand],'run':te[cand],'r5':(a[cand]%500==0),'int':(a[cand]%1==0)})
j=sd.merge(tg,on=['key','run']); j['dt']=(j.t-j.st).abs()
j=j[j.cur!=j.styp]
for w,wn in [(86400,'1d'),(3*86400,'3d'),(7*86400,'7d'),(31*86400,'31d')]:
    g=j[j.dt<=w].sort_values('dt').drop_duplicates('i'); a_=g[~g.run]
    print(f'\nwindow {wn}: TRAIN rows {len(a_)} label==block {(a_.y==a_.styp).sum()} label==model {(a_.y==a_.cur).sum()} legit {(a_.y==0).sum()} | TEST rows {g.run.sum()}')
    x=a_.assign(blk=sh[a_.styp],mdl=sh[a_.cur],ob=a_.y==a_.styp,om=a_.y==a_.cur).groupby(['blk','mdl']).agg(n=('i','size'),label_block=('ob','sum'),label_model=('om','sum'))
    print(x.to_string())
    b=g[g.run]; print('TEST:',b.assign(blk=sh[b.styp],mdl=sh[b.cur]).groupby(['blk','mdl']).size().to_dict())
j.to_pickle(f'{CACHE}/prop4_links.pkl')
