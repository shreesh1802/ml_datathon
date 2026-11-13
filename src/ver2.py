import numpy as np,pandas as pd,sys
sys.path.insert(0,r'C:\Users\0095\Downloads')
from common import load_all,CACHE,LABELS,macro_f1
import apply_winning_fixes as W
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
dn=al.device_id.str[2:].astype(int).values
seg=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl'); cls=seg.seg_class.values.astype(int)
G=np.full(len(al),'',object); G[te]=pd.read_csv('../submissions/sub_G1.csv').label.values
# 1. ID density, both runs
for nm,lo,hi in [('train fraud blocks?',422978,426000),('train later',430000,473631),('oct fraud',473632,474551),('oct legit',474552,482013)]:
    u=np.unique(dn[(dn>=lo)&(dn<=hi)]); print(f'{nm} {lo}-{hi}: present {len(u)} of {hi-lo+1} = {len(u)/(hi-lo+1):.3f}')
# find train-run boundary where density becomes 1: rolling
u=np.unique(dn[(dn>=422978)&(dn<473632)]); pres=np.zeros(473632-422978,bool); pres[u-422978]=True
miss=np.where(~pres)[0]+422978; print('train run: last missing id', miss.max(), 'n missing', len(miss))
# train-run: label by device in the fraud range, compare our seg class to labels, and where seg says legit but before last-missing
m=tr&(dn>=422978)&(dn<=miss.max())
print('train run rows before last-missing id:', m.sum(), ' labels:', pd.Series(lab[m]).value_counts().to_dict())
print('  our seg==0 there:', (m&(cls==0)).sum(), 'labels of those:', pd.Series(lab[m&(cls==0)]).value_counts().to_dict())
m2=tr&(dn>miss.max())&(dn<473632); print('train run rows after last-missing:', m2.sum(), pd.Series(lab[m2]).value_counts().head(4).to_dict(), 'seg>0:', (m2&(cls>0)).sum())
# seg accuracy on train-run fraud range
print('train: seg class vs label on seg>0 rows: acc', np.mean(np.array(LABELS)[cls[m&(cls>0)]]==lab[m&(cls>0)]))
# 2. October: our seg vs their block table
ob=W.get_oct_block_label(dn); o=te&(dn>=473632)&(dn<=474551)
print('Oct fraud range rows', o.sum(), 'our seg legit there', (o&(cls==0)).sum())
print(pd.crosstab(np.array(LABELS)[ob[o]],np.array(LABELS)[cls[o]]))
print('Oct rows >=474552 with our seg>0:', (te&(dn>=474552)&(cls>0)).sum())
