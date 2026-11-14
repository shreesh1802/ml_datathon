import numpy as np,pandas as pd,os
from common import load_all,CACHE,SUBS,LABELS
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values; lab=al.label.values
tr=(~te)&(mo<'2026-10')
dn=al.device_id.str[2:].astype(int).values; pool=(dn>=420000)&(dn<=423000)
amt=al.amount.values; lg=al.login_to_txn_sec.values; cat=al.merchant_category.values; tt=al.txn_type.values; mer=al.receiver_type.values=='MERCHANT'
G=np.full(len(al),'',object); G[te]=pd.read_csv(os.path.join(SUBS,'sub_G13.csv')).label.values
print('pool rows (train): login seconds',pd.Series(lg[pool&tr]).value_counts().sort_index().head(9).to_dict())
print('pool rows (train): category',pd.Series(cat[pool&tr]).value_counts().head(4).to_dict(),'| txn',pd.Series(tt[pool&tr]).value_counts().head(3).to_dict(),'| whole rupees',np.mean(amt[pool&tr]%1==0).round(3),'| amount range',amt[pool&tr].min(),np.quantile(amt[pool&tr],.99))
sig=mer&np.isin(cat,['ecommerce','gaming'])&(amt%1==0)&(amt>=10)&(amt<=499)
for nm,s in [('signature, any login',sig),('signature + login 1-6s',sig&(lg>=1)&(lg<=6)),('signature + login 1-3s',sig&(lg>=1)&(lg<=3)),('login 1-6s, any payment',(lg>=1)&(lg<=6))]:
    a=s&~pool&tr; b=s&~pool&te
    print(f'{nm:26s} off-pool: train/6 {a.sum()/6:8.1f} (bot-labelled {np.mean(lab[a]=="velocity_bot"):.3f}, legit {np.mean(lab[a]=="legit"):.3f}) | test {b.sum():6d} | G13 predicts fraud on {int((G[b]!="legit").sum())}')
# October bot-block rows (non-pool): what do they look like
cls=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl').seg_class.values.astype(int)
for nm,m in [('TRAIN bot block (non-pool)',tr&(cls==2)&~pool),('OCT bot block (non-pool)',te&(cls==2)&~pool)]:
    print(nm,'rows',m.sum(),'| signature share',np.mean(sig[m]).round(3),'| login<=6 share',np.mean(lg[m]<=6).round(3),'| merchant share',np.mean(mer[m]).round(3),'| ecommerce/gaming share',np.mean(np.isin(cat[m],['ecommerce','gaming'])).round(3))
print(al[te&(cls==2)&~pool][['timestamp','sender_id','receiver_id','receiver_type','amount','txn_type','merchant_category','login_to_txn_sec']].sort_values('timestamp').to_string(index=False))
