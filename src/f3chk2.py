import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
pred=decide(oof,tau); q=cls>0; pred[q]=cls[q]; pred[te]=pd.read_csv(os.path.join(SUBS,'sub_G8.csv')).label.map(L).values
F=pd.read_pickle(f'{CACHE}/feats3.pkl'); print(F.shape)
tt=al.txn_type.values; p2p=al.receiver_type.values=='P2P'
print('rows the model calls LEGIT, by ring feature: train rows / fraud-labelled share | test rows')
for c in ['rcv_ring_other','snd_ring_other','rcv_ring_roles','snd_ring_roles','rcv_was_bot_snd','snd_was_qr_pc_rcv']:
    v=F[c].values
    for lo,hi in [(0,1),(1,2),(2,4),(4,99)]:
        m=(v>=lo)&(v<hi)&(pred==0)
        a=m&tr
        print(f'  {c:18s} [{lo},{hi}): train {a.sum():8d} fraud share {np.mean(y[a]>0) if a.any() else 0:.4f} | test {(m&te).sum():7d}')
print('\nall rows by rcv_ring_other (any prediction): train fraud share and how the model does')
for c,base in [('rcv_ring_other',np.ones(len(al),bool)),('rcv_ring_other',(tt=='COLLECT')&p2p),('snd_ring_other',np.ones(len(al),bool))]:
    v=F[c].values
    for lo,hi in [(0,1),(1,2),(2,4),(4,99)]:
        m=(v>=lo)&(v<hi)&base; a=m&tr
        print(f'  {c:16s} [{lo},{hi}) n={a.sum():8d} fraud {np.mean(y[a]>0):.3f} | fraud rows caught {np.mean(pred[a&(y>0)]>0) if (a&(y>0)).any() else 0:.3f} | test n={(m&te).sum()} test predicted fraud {np.mean(pred[m&te]>0):.3f}')
