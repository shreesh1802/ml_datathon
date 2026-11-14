import sys,numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,macro_f1
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
    return decide(oof,tau)
pairs=[('account_takeover','sim_swap'),('sim_swap','account_takeover'),('mule_account','layering_chain'),('velocity_bot','mule_account'),('fake_cashback','phishing_collect'),('phishing_collect','fake_cashback'),('loan_app_extortion','phishing_collect'),('refund_abuse','qr_swap'),('merchant_collusion','qr_swap')]
m=tr&ordp; out={}
for tags in sys.argv[1:]:
    p=rule(tags.split(',')); f=macro_f1(y[m],p[m],per_class=True)
    out[tags]=[f.mean().round(4)]+[int((m&(y==L[a])&(p==L[b])).sum()) for a,b in pairs]+[int((m&(y==L[a])&(p==L[a])).sum()) for a,b in pairs]
idx=['macro-F1 (ordinary phones)']+[f'{a[:8]} called {b[:8]}' for a,b in pairs]+[f'{a[:8]} called right' for a,b in pairs]
print(pd.DataFrame(out,index=idx).to_string())
