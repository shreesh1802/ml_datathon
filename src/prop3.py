"""Event-level type correction from block phones.
Each fraud type has a minority variant that behaves like a lookalike type. A row of the event that happens to sit on a
block phone reveals the event's true type, so linked rows predicted as the lookalike take the block's type.
usage: python src/prop3.py IN_SUB OUT_SUB SEG   (prints the training check, writes the file)"""
import sys,os,numpy as np,pandas as pd
from common import load_all,CACHE,SUBS,LABELS,macro_f1
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; K=len(LABELS); L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
insub,outsub,segname=sys.argv[1],sys.argv[2],sys.argv[3]
dn=al.device_id.str[2:].astype(int).values
oof=(np.load(f'{CACHE}/oof_E1.npy')+np.load(f'{CACHE}/oof_M1.npy'))/2
def decide(p,tau):
    r=p[:,1:]/tau[1:]; k=r.argmax(1)+1; return np.where(r.max(1)>1,k,0)
tau=np.full(K,.25)
for it in range(6):
    f=macro_f1(y[tr],decide(oof[tr],tau),per_class=True); tau=np.r_[1,np.clip(f[1:]/2,.1,.45)]
cls=pd.read_pickle(f'{CACHE}/feats_seg_{segname}.pkl').seg_class.values.astype(int)
pred=decide(oof,tau); q=cls>0; pred[q]=cls[q]
sub=pd.read_csv(os.path.join(SUBS,insub+'.csv')); assert (sub.id.values==al.txn_id.values[te]).all()
pred[te]=sub.label.map(L).values
# (true type, lookalike it is predicted as): the directed pairs seen in the out-of-fold confusion matrix
PAIRS={('account_takeover','sim_swap'),('sim_swap','account_takeover'),('mule_account','layering_chain'),('velocity_bot','mule_account'),
       ('fake_cashback','phishing_collect'),('phishing_collect','fake_cashback'),('loan_app_extortion','phishing_collect'),
       ('refund_abuse','qr_swap'),('merchant_collusion','qr_swap')}
PAIRS={(L[a],L[b]) for a,b in PAIRS}
WIN=int(os.environ.get('WIN',6*3600)); HOPS=int(os.environ.get("HOPS",2))
t=al.t.values.astype(np.int64); S=al.sender_id.values; R=al.receiver_id.values; p2p=al.receiver_type.values=='P2P'
run=te.astype(int)
seedmask=(cls>0)&(dn>=423001)          # block phones outside the shared bot pool
new=pred.copy(); src_t=np.where(seedmask,t,0); typ=np.where(seedmask,cls,0); is_src=seedmask.copy()
for hop in range(HOPS):
    sd=pd.DataFrame({'st':t[is_src],'styp':typ[is_src],'S':S[is_src],'R':R[is_src],'run':run[is_src],'p2p':p2p[is_src]})
    cand=np.where(~is_src&(dn<420000)&(new>0))[0]
    tg=pd.DataFrame({'i':cand,'t':t[cand],'cur':new[cand],'S':S[cand],'R':R[cand],'run':run[cand],'p2p':p2p[cand]})
    hits=[]
    for lk,rk,sp,tp in [('S','S',False,False),('S','R',False,True),('R','S',True,False),('R','R',True,True)]:
        a=sd[sd.p2p] if sp else sd; b=tg[tg.p2p] if tp else tg
        j=a[['st','styp','run',lk]].rename(columns={lk:'key'}).merge(b[['i','t','cur','run',rk]].rename(columns={rk:'key'}),on=['key','run'])
        j=j[(j.t-j.st).abs()<=WIN]; hits.append(j[['i','styp','cur','t','st']])
    j=pd.concat(hits); j=j[[(a,b) in PAIRS for a,b in zip(j.styp,j.cur)]]
    if len(j)==0: break
    j['dt']=(j.t-j.st).abs(); j=j.sort_values('dt').drop_duplicates('i')
    new[j.i.values]=j.styp.values; typ[j.i.values]=j.styp.values; is_src[:]=False; is_src[j.i.values]=True
    a=j[~te[j.i.values]]; print(f'hop {hop+1}: train rows changed {len(a)}: label==new {(y[a.i.values]==a.styp.values).sum()}, label==old {(y[a.i.values]==a.cur.values).sum()} | test rows changed {te[j.i.values].sum()}')
f0=macro_f1(y[tr],pred[tr],per_class=True); f1=macro_f1(y[tr],new[tr],per_class=True)
print('training macro-F1 %.4f -> %.4f'%(f0.mean(),f1.mean())); print(pd.DataFrame({'before':f0.round(4),'after':f1.round(4),'delta':(f1-f0).round(4)},index=LABELS).query('delta!=0').to_string())
ch=te&(new!=pred); print('test rows changed:',ch.sum()); print(pd.crosstab(np.array(LABELS)[pred[ch]],np.array(LABELS)[new[ch]],rownames=['from'],colnames=['to']).to_string())
pd.DataFrame({'id':sub.id,'label':np.array(LABELS)[new[te]]}).to_csv(os.path.join(SUBS,outsub+'.csv'),index=False); print('wrote',outsub)
