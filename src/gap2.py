import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
pred=np.load(f'{CACHE}/pred_G9raw.npy'); oof=(np.load(f'{CACHE}/oof_E2n.npy')+np.load(f'{CACHE}/oof_M2.npy'))/2
pl=oof[:,0]; pmax=oof[:,1:].max(1)
rows=[]
for k in range(1,K):
    m=tr&(y==k); T=m.sum()
    caught=(m&(pred==k)).sum(); wrong=(m&(pred>0)&(pred!=k)).sum()
    miss=m&(pred==0); rnd=(miss&(pl>0.9)).sum(); mid=(miss&(pl<=0.9)).sum()
    P=(tr&(pred==k)).sum(); fpl=(tr&(pred==k)&(y==0)).sum(); fpo=(tr&(pred==k)&(y>0)&(y!=k)).sum()
    f=2*caught/(P+T)
    # ceiling under the rules: random rows undetectable, lookalikes unresolved; everything else caught, no false alarms beyond swaps received
    f_if_mid=2*(caught+mid)/(P+mid+T)
    rows.append(dict(cls=LABELS[k][:14],true=T,caught=caught,wrong_type=wrong,random_missed=rnd,detectable_missed=mid,pred=P,fp_legit_label=fpl,fp_other_fraud=fpo,F1=round(f,3),F1_if_missed_caught=round(f_if_mid,3)))
d=pd.DataFrame(rows); print(d.to_string(index=False))
print('macro now (train labels):',round((0.9872+d.F1.sum())/12,4),'| if every detectable-missed row were caught:',round((0.9872+d.F1_if_missed_caught.sum())/12,4))
# how sure is the model on the detectable-missed rows, and what would it cost to take them
mid=tr&(y>0)&(pred==0)&(pl<=0.9)
print('detectable-but-missed rows:',mid.sum(),'| their best fraud probability quantiles',np.quantile(pmax[mid],[.1,.25,.5,.75,.9]).round(3))
for lo,hi in [(0.02,0.05),(0.05,0.1),(0.1,0.15),(0.15,0.25)]:
    b=tr&(pred==0)&(pmax>=lo)&(pmax<hi); print(f'  rows called legit with best fraud prob in [{lo},{hi}): {b.sum():6d}  fraud-labelled {int((y[b]>0).sum()):5d} ({np.mean(y[b]>0):.3f})  same type as the top class {np.mean((y[b]==oof[b,1:].argmax(1)+1)):.3f}')
