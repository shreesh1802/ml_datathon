import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
dn=al.device_id.str[2:].astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2; pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
P=oof.copy(); P[te]=pt
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
raw=decide(P,tau)
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
pool=(dn>=420000)&(dn<=423000)
seed=(cls>0)&~pool
t=al.t.values.astype(np.int64); S=al.sender_id.values; R=al.receiver_id.values; p2p=al.receiver_type.values=='P2P'
d=pd.DataFrame({'i':np.arange(len(al)),'t':t,'S':S,'R':R,'seed':seed,'cls':cls,'raw':raw,'y':y,'te':te,'p2p':p2p})
sd=d[d.seed][['i','t','S','R','cls','te','p2p']].rename(columns={'i':'si','t':'st','cls':'scls'})
tgt=d[~d.seed & (dn<420000)]
out=[]
for nm,lk,rk in [('same sender','S','S'),('seed sender is receiver','S','R'),('seed receiver is sender','R','S'),('same receiver (P2P)','R','R')]:
    s2=sd if nm!='same receiver (P2P)' else sd[sd.p2p]
    j=s2[['si','st','scls','te',lk]].rename(columns={lk:'key'}).merge(tgt[['i','t','raw','y','te',rk]].rename(columns={rk:'key'}),on=['key','te'])
    j=j[(j.t-j.st).abs()<=86400]; j['link']=nm; out.append(j[['si','i','scls','raw','y','te','link','t','st']])
J=pd.concat(out)
for nm,g in J.groupby('link'):
    a=g[~g.te]
    fr=a[a.raw>0]
    dis=fr[fr.raw!=fr.scls]; agr=fr[fr.raw==fr.scls]
    lg=a[a.raw==0]
    print(f'\n{nm}: TRAIN linked rows {len(a)} | model says same fraud type as block: {len(agr)} (label==block {np.mean(agr.y==agr.scls):.2f})')
    print(f'   model says OTHER fraud type: {len(dis)} -> label==block {(dis.y==dis.scls).sum()}, label==model {(dis.y==dis.raw).sum()}, legit {(dis.y==0).sum()}, other {((dis.y>0)&(dis.y!=dis.scls)&(dis.y!=dis.raw)).sum()}')
    print(f'   model says legit: {len(lg)} -> label==block {(lg.y==lg.scls).sum()}, legit {(lg.y==0).sum()}, other fraud {((lg.y>0)&(lg.y!=lg.scls)).sum()}')
    b=g[g.te]; print(f'   TEST linked rows {len(b)}: model other fraud type {((b.raw>0)&(b.raw!=b.scls)).sum()}, model legit {(b.raw==0).sum()}, same {(b.raw==b.scls).sum()}')
J.to_pickle(f'{CACHE}/prop_links.pkl')
