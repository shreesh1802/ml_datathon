import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; Kn=len(LABELS)
tr=(~te)&(mo<'2026-10')
y=al.label.map({l:i for i,l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
# block-blind model for comparison
bo=np.load(f'{CACHE}/oof_C5.npy'); bt=np.load(f'{CACHE}/test_C5.npy'); B=bo.copy(); B[te]=bt
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
sh=np.array([l[:6] for l in LABELS])
bp=B[:,1:].argmax(1)+1; bc=B[:,1:].max(1)
for th in [0.5,0.8]:
    x=(cls>0)&(bc>th)&(bp!=cls)
    xt=x&tr; print(f'\n== block-blind model confident (> {th}) in another fraud type')
    print('TRAIN rows',xt.sum(),'| label==block',(y[xt]==cls[xt]).sum(),'| label==model',(y[xt]==bp[xt]).sum(),'| label legit',(y[xt]==0).sum(),'| other',((y[xt]!=cls[xt])&(y[xt]!=bp[xt])&(y[xt]!=0)).sum())
    print(pd.crosstab(sh[cls[xt]],sh[bp[xt]],rownames=['block'],colnames=['model']).to_string())
    xe=x&te; print('TEST rows',xe.sum()); print(pd.crosstab(sh[cls[xe]],sh[bp[xe]],rownames=['block'],colnames=['model']).to_string())
# reference: how often does a confident pattern carry another fraud label OUTSIDE blocks (train)?
nb=tr&(cls<=0)&(bc>0.8)
print('\nnon-block confident rows',nb.sum(),'label==model',(y[nb]==bp[nb]).mean().round(3),'legit',(y[nb]==0).mean().round(3),'other fraud',((y[nb]!=bp[nb])&(y[nb]>0)).mean().round(3))
blk=tr&(cls>0)&(bc>0.8)
print('block confident rows',blk.sum(),'label==model',(y[blk]==bp[blk]).mean().round(3),'legit',(y[blk]==0).mean().round(3),'other fraud',((y[blk]!=bp[blk])&(y[blk]>0)).mean().round(3))
print('\n== excluding SIM/takeover blocks; block-blind model confident > 0.5, training run')
ex=tr&(cls>0)&~np.isin(cls,[4,5])&(bc>0.5)
a=ex&(bp==cls); b=ex&(bp!=cls)
print('pattern == block:',a.sum(),'| label==block',(y[a]==cls[a]).sum(),'| label legit',(y[a]==0).sum(),'| label OTHER fraud',((y[a]>0)&(y[a]!=cls[a])).sum())
print('pattern != block:',b.sum(),'| label==block',(y[b]==cls[b]).sum(),'| label==pattern',(y[b]==bp[b]).sum(),'| label legit',(y[b]==0).sum(),'| other',((y[b]>0)&(y[b]!=cls[b])&(y[b]!=bp[b])).sum())
nb=tr&(cls<=0)&(bc>0.5)&~np.isin(bp,[4,5])
print('non-block, same classes:',nb.sum(),'label==pattern',(y[nb]==bp[nb]).sum(),'legit',(y[nb]==0).sum(),'other fraud',((y[nb]>0)&(y[nb]!=bp[nb])).sum(), 'rate', (((y[nb]>0)&(y[nb]!=bp[nb])).sum()/nb.sum()).round(3))
