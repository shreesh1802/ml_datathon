import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; L={l:i for i,l in enumerate(LABELS)}
tr=(~te)&(mo<'2026-10'); y=al.label.map(L).fillna(-1).astype(int).values
pred=np.load(f'{CACHE}/pred_G9raw.npy').copy(); cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
G=np.full(len(al),'',object); G[te]=pd.read_csv(os.path.join(SUBS,'sub_G12.csv')).label.values
dn=al.device_id.str[2:].astype(int).values; t=al.t.values.astype(np.int64); S=al.sender_id.values; R=al.receiver_id.values; p2p=al.receiver_type.values=='P2P'
ordp=dn<420000
def link(seedmask,keyseed,tgtmask,keys_t,win):
    sd=pd.DataFrame({'key':keyseed[seedmask],'st':t[seedmask],'run':te[seedmask]})
    hit=np.zeros(len(al),bool)
    for kt in keys_t:
        tg=pd.DataFrame({'i':np.where(tgtmask)[0],'key':kt[tgtmask],'t':t[tgtmask],'run':te[tgtmask]})
        j=sd.merge(tg,on=['key','run']); j=j[(j.t-j.st).abs()<=win]; hit[j.i.unique()]=True
    return hit
def report(nm,hit,newk,oldk):
    a=hit&tr; print(f'  {nm:14s} train rows {a.sum():4d} | label==new {(y[a]==newk).sum():4d} | label==old {(y[a]==oldk).sum():3d} | legit {(y[a]==0).sum():3d} | by month {[int((a&(mo==h)).sum()) for h in ["2026-07","2026-08","2026-09"]]} | TEST rows {int((hit&te).sum())}, of which G12 still says old type: {int((hit&te&(G==LABELS[oldk])).sum())}')
B,M,A,Sg=L['velocity_bot'],L['mule_account'],L['account_takeover'],L['sim_swap']
print('bot block (P2P rows, outside pool) -> rows predicted mule sharing the hub account')
seed=(cls==B)&(dn>=423001)&p2p; tgt=ordp&(pred==M)
for wn,w in [('12h',43200),('24h',86400),('7d',7*86400),('any time',10**9)]: report(wn,link(seed,R,tgt,[R,S],w),B,M)
print('takeover block -> rows predicted SIM swap by the same sender')
seed=(cls==A); tgt=ordp&(pred==Sg)
for wn,w in [('12h',43200),('24h',86400),('7d',7*86400),('any time',10**9)]: report(wn,link(seed,S,tgt,[S],w),A,Sg)
print('other pairs, same-account link at any distance vs 12h (label==new / label==old):')
for bk,pk in [(L['mule_account'],L['layering_chain']),(L['fake_cashback'],L['phishing_collect']),(L['phishing_collect'],L['fake_cashback']),(Sg,A),(L['loan_app_extortion'],L['phishing_collect'])]:
    seed=(cls==bk)&(dn>=423001); tgt=ordp&(pred==pk)
    out=[]
    for wn,w in [('12h',43200),('24h',86400),('7d',7*86400)]:
        h=link(seed,S,tgt,[S,R],w)|link(seed&p2p,R,tgt,[S,R],w); a=h&tr; out.append(f'{wn}: {(y[a]==bk).sum()}/{(y[a]==pk).sum()} (test {int((h&te&(G==LABELS[pk])).sum())})')
    print(f'  block {LABELS[bk][:10]} -> pred {LABELS[pk][:10]}: '+' | '.join(out))
