import numpy as np,pandas as pd,os
from common import load_all,CACHE,LABELS,macro_f1,DATA
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
dn=al.device_id.str[2:].astype(int).values; ordp=dn<420000
def rule(tags):
    oof=sum(np.load(f'{CACHE}/oof_{t}.npy') for t in tags)/len(tags)
    def decide(p,tau):
        r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
    tau=np.full(K,.25)
    for it in range(6):
        f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
    return decide(oof,tau),oof
pE,oE=rule(['E1','M1']); pT,oT=rule(['T1','MT1'])
m=tr&ordp
print('ordinary phones, given labels: macro F1  E1+M1 %.4f   T1+MT1 %.4f'%(macro_f1(y[m],pE[m]),macro_f1(y[m],pT[m])))
fE=macro_f1(y[m],pE[m],per_class=True); fT=macro_f1(y[m],pT[m],per_class=True)
print(pd.DataFrame({'E1+M1':fE.round(3),'T1+MT1':fT.round(3)},index=LABELS).T.to_string())
pairs=[('account_takeover','sim_swap'),('sim_swap','account_takeover'),('mule_account','layering_chain'),('velocity_bot','mule_account'),('fake_cashback','phishing_collect'),('phishing_collect','fake_cashback'),('loan_app_extortion','phishing_collect'),('refund_abuse','qr_swap'),('merchant_collusion','qr_swap')]
print('\nlabel -> lookalike: rows (ordinary phones) predicted as lookalike / as label, by each model')
for a,b in pairs:
    la,lb=L[a],L[b]; q=m&(y==la)
    print(f'{a[:12]:12s} -> {b[:12]:12s} | E1+M1: as {b[:6]} {(q&(pE==lb)).sum():4d}, as {a[:6]} {(q&(pE==la)).sum():4d} | T1+MT1: as {b[:6]} {(q&(pT==lb)).sum():4d}, as {a[:6]} {(q&(pT==la)).sum():4d}')
# features of the takeover/SIM groups
ac=pd.read_csv(os.path.join(DATA,'accounts.csv')).set_index('account_id').home_city
away=(al.device_city.values!=ac.reindex(al.sender_id.values).values); hr=al.ts.dt.hour.values; night=(hr>=22)|(hr<5)
own=(al.device_id.str[2:].astype(int).values==al.sender_id.str[2:].astype(int).values)
def prof(nm,q): print(f'{nm:28s} n={q.sum():5d} sim<72h {np.mean(al.sim_age_hours.values[q]<72):.2f} dag<2 {np.mean(al.device_age_days.values[q]<2):.2f} away {away[q].mean():.2f} night {night[q].mean():.2f} own-phone {own[q].mean():.2f} p2p {np.mean(al.receiver_type.values[q]=="P2P"):.2f} med amt {np.median(al.amount.values[q]):.0f}')
print()
prof('SIM label, pred SIM',m&(y==4)&(pE==4)); prof('ATO label, pred SIM',m&(y==5)&(pE==4)); prof('ATO label, pred ATO',m&(y==5)&(pE==5)); prof('SIM label, pred ATO',m&(y==4)&(pE==5))
blk=tr&~ordp
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
prof('SIM block rows (train)',blk&(cls==4)); prof('ATO block rows (train)',blk&(cls==5))
prof('SIM block rows (Oct)',te&(cls==4)); prof('ATO block rows (Oct)',te&(cls==5))
