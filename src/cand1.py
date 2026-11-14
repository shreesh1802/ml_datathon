"""Agent candidates A (phone owner link), B (balance continuity), C (narrow-window link to patterned fraud)."""
import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS,Win
al=load_all(); te=al.is_test.values; lab=al.label.values; mo=al.timestamp.str[:7].values; L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
oof=np.load(f'{CACHE}/oof_E1.npy'); pl=oof[:,0]
leg=tr&(lab=='legit')&(pl>0.9); ordf=tr&(lab!='legit')&(pl>0.9); patt=tr&(lab!='legit')&(pl<0.5)
dn=al.device_id.str[2:].astype(int).values; sn=al.sender_id.str[2:].astype(int).values
print('=== A. rows on another account\'s primary phone ===')
oth=(dn!=sn)&(dn<420000)
# is the phone owner (account y) involved in patterned fraud / fraud-labelled rows?
rn=np.where(al.receiver_type.values=='P2P',al.receiver_id.str[2:].astype(int).values,-1)
fr_acc=set(sn[patt])|set(rn[patt]); anyf_acc=set(sn[tr&(lab!='legit')])|set(rn[tr&(lab!='legit')])
own_in_pat=np.fromiter((d in fr_acc for d in dn),bool,len(dn)); own_in_any=np.fromiter((d in anyf_acc for d in dn),bool,len(dn))
for nm,m in [('legit',leg&oth),('ordinary fraud',ordf&oth)]:
    print(f'  {nm:15s} n={m.sum():6d} | phone owner is in patterned fraud {own_in_pat[m].mean():.4f} | owner in any fraud-labelled row {own_in_any[m].mean():.4f} | owner is the receiver {np.mean(dn[m]==rn[m]):.4f}')
print('  fraud share among other-phone rows the model calls legit:', round(ordf[oth&tr&(pl>0.9)].mean(),4),' vs own-phone rows', round(ordf[(dn==sn)&tr&(pl>0.9)].mean(),4))
# owner's own rows: do they predict?
d=pd.DataFrame({'own':dn,'f':np.where(tr,(lab!='legit').astype(float),np.nan)})
owner_rate=pd.Series(np.where(tr&(dn==sn),(lab!='legit').astype(float),np.nan)).groupby(sn).mean()
r=pd.Series(dn).map(owner_rate).values
for nm,m in [('legit',leg&oth),('ordinary fraud',ordf&oth)]: print(f'  {nm:15s} mean fraud rate of the phone owner\'s own rows {np.nanmean(r[m]):.4f}')

print('\n=== B. balance continuity along the sender ===')
o=np.lexsort((al.t.values,sn)); s_=sn[o]; b=al.balance_before.values[o]; a=al.amount.values[o]; same=np.r_[False,s_[1:]==s_[:-1]]
resid=np.full(len(o),np.nan); resid[1:]=b[1:]-(b[:-1]-a[:-1]); resid[~same]=np.nan
R=np.empty(len(o)); R[o]=resid
nxt=np.full(len(o),np.nan); nxt[:-1]=b[1:]-(b[:-1]-a[:-1]); same2=np.r_[s_[1:]==s_[:-1],False]; nxt[~same2]=np.nan; N=np.empty(len(o)); N[o]=nxt
for nm,m in [('legit',leg),('ordinary fraud',ordf),('patterned fraud',patt)]:
    v=R[m]; v=v[~np.isnan(v)]; w=N[m]; w=w[~np.isnan(w)]
    print(f'  {nm:15s} |resid to previous row| <0.01: {np.mean(np.abs(v)<0.01):.4f}  <1: {np.mean(np.abs(v)<1):.4f}  median |resid| {np.median(np.abs(v)):.0f} | to next row <0.01: {np.mean(np.abs(w)<0.01):.4f} median {np.median(np.abs(w)):.0f}')

print('\n=== C. ordinary-looking rows near patterned fraud of the same class (shared account) ===')
t=al.t.values.astype(np.int64)
codes=pd.factorize(np.r_[al.sender_id.values,al.receiver_id.values])[0]; S=codes[:len(al)].astype(np.int64); Rr=codes[len(al):].astype(np.int64)
for k in [1,3,4,5,6,7,8,9,10,11]:
    p=np.where(tr&(y==k)&(oof[:,k]>0.5))[0]
    keys=np.r_[S[p],Rr[p]]; tt=np.r_[t[p],t[p]]; W=Win(keys,tt)
    out=[]
    for wn,w in [('1h',3600),('6h',6*3600),('24h',86400)]:
        c=np.maximum(W.count(S,t,-w,w),W.count(Rr,t,-w,w))>0
        ok=tr&(y==k)&(pl>0.9); lg=leg
        out.append(f'{wn}: ord {c[ok].mean():.4f} legit {c[lg].mean():.4f} lift {c[ok].mean()/max(c[lg].mean(),1e-9):.2f} (ord rows hit {c[ok].sum()}, legit rows hit {c[lg].sum()})')
    print(f'  {LABELS[k][:14]:14s} '+' | '.join(out))
