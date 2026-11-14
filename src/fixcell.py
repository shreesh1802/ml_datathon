"""Remove one miscalibrated cell: phishing predicted on a P2P collect request whose amount is a whole multiple of 10
(but not of 500, and not an exact phishing amount such as 1999 or 9990). On training data these are 8% phishing and
89% legit, far below the keep threshold of the decision rule. usage: python src/fixcell.py IN_SUB OUT_SUB"""
import sys,os,numpy as np,pandas as pd
from common import load_all,CACHE,SUBS,LABELS,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
amt=al.amount.values; dn=al.device_id.str[2:].astype(int).values
ex=np.isin(amt,[1950.,1990.,1999.,4950.,4990.,4999.,9950.,9990.,9999.,24950.,24990.,24999.])
cell=(al.txn_type.values=='COLLECT')&(al.receiver_type.values=='P2P')&(amt%10==0)&(amt%500!=0)&~ex&(dn<420000)
pred=np.load(f'{CACHE}/pred_G9raw.npy').copy()                       # training-side check uses the G9 blend before event corrections
k=L['phishing_collect']; m=cell&(pred==k)&tr
f0=macro_f1(y[tr],pred[tr],per_class=True); p1=pred.copy(); p1[cell&(pred==k)]=0; f1=macro_f1(y[tr],p1[tr],per_class=True)
print(f'training: {m.sum()} rows changed, {int((y[m]==k).sum())} were phishing, {int((y[m]==0).sum())} legit | macro-F1 {f0.mean():.4f} -> {f1.mean():.4f} | phishing F1 {f0[k]:.4f} -> {f1[k]:.4f}')
for h in ['2026-07','2026-08','2026-09']:
    q=tr&(mo==h); print('  ',h,round(macro_f1(y[q],pred[q]),4),'->',round(macro_f1(y[q],p1[q]),4),'rows',(m&q).sum())
s=pd.read_csv(os.path.join(SUBS,sys.argv[1]+'.csv')); lab=s.label.values.copy(); c=cell[te]&(lab=='phishing_collect')
lab[c]='legit'; print('test rows changed:',c.sum()); pd.DataFrame({'id':s.id,'label':lab}).to_csv(os.path.join(SUBS,sys.argv[2]+'.csv'),index=False); print('wrote',sys.argv[2])
