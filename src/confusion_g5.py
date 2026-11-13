"""Out-of-fold confusion matrix and per-class scores for the G5 recipe (E1+M1 blend, F/2 rule, s2 block override),
on Jul-Sep given labels. Writes OOF_CONFUSION_G5.md in the project root."""
import os,numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,ROOT,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS)
tr=(~te)&(mo<'2026-10')
y=al.label.map({l:i for i,l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
pt=(np.load(f'{CACHE}/test_E1.npy')+np.load(f'{CACHE}/test_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
p=decide(oof,tau); q=cls>0; p[q]=cls[q]
yt,pp=y[tr],p[tr]
cm=pd.crosstab(pd.Categorical.from_codes(yt,LABELS),pd.Categorical.from_codes(pp,LABELS),dropna=False)
sh=[l[:9] for l in LABELS]; cm.index=sh; cm.columns=sh
tp=np.diag(cm.values); prec=tp/np.maximum(cm.values.sum(0),1); rec=tp/np.maximum(cm.values.sum(1),1); f1=2*prec*rec/np.maximum(prec+rec,1e-9)
G=pd.read_csv(os.path.join(ROOT,'submissions','sub_G5.csv')).label.value_counts()
probe={'mule_account':.541,'account_takeover':.511,'sim_swap':.524,'phishing_collect':.518,'fake_cashback':.517,'merchant_collusion':.300}
per=pd.DataFrame({'class':LABELS,'true rows (3 months)':cm.values.sum(1),'predicted rows':cm.values.sum(0),'precision':prec.round(3),'recall':rec.round(3),'F1':f1.round(3),
   'threshold':np.r_[np.nan,tau[1:]].round(3),'October predictions (G5)':[int(G.get(l,0)) for l in LABELS],'October F1 from public probes (G1)':[probe.get(l,'') for l in LABELS]})
def md(d,index=False):
    d=d.reset_index() if index else d
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','|'+'---|'*len(d.columns)]+['| '+' | '.join(map(str,r))+' |' for r in d.values])
kc=LABELS.index('merchant_collusion'); mer=al.receiver_type.values=='MERCHANT'; a=al.amount.values
c=tr&(y==kc); r5=mer&(a%500==0); isint=(a%1==0)
grp=np.where(r5,'merchant, multiple of 500',np.where(mer&isint,'merchant, other integer',np.where(mer,'merchant, decimal','P2P')))
col=pd.crosstab(grp[c],pd.Categorical.from_codes(p[c],LABELS),dropna=False); col=col.loc[:,col.sum()>0]
out=f"""# Out-of-fold confusion matrix for the G5 recipe

Recipe: average of runs E1 and M1, F/2 decision rule, block override from the density-aware segmentation (`s2`).
Scored on July-September rows against the given (noisy) training labels. Macro-F1 = {f1.mean():.4f}.
Public score of the same recipe on October: 0.5405.

Note: training labels are noisy (about 12% of fraud is labelled legit; about 61% of fraud-labelled rows look like legit rows and are random). Test labels are clean.

## Per-class scores

{md(per)}

## Confusion matrix (rows = given label, columns = prediction)

{md(cm.rename_axis('given \\ predicted'),index=True)}

## Where given `merchant_collusion` rows go, by kind of payment

{md(col.rename_axis('payment kind'),index=True)}
"""
open(os.path.join(ROOT,'OOF_CONFUSION_G5.md'),'w').write(out); print(out)
