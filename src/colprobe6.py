import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te)&(mo<'2026-10'); lab=al.label.values
seg=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl'); cls=seg.seg_class.values.astype(int)
G=pd.read_csv('../submissions/sub_G1.csv').label.values
rows=[]
for k in range(1,12):
    c=LABELS[k]
    rows.append(dict(cls=c, train_block_rows_pm=((cls==k)&tr).sum()/3, train_block_true=((cls==k)&tr&(lab==c)).sum()/3,
        train_label_pm=((lab==c)&tr).sum()/3, test_block_rows=((cls==k)&te).sum(), G1_pred=(G==c).sum()))
d=pd.DataFrame(rows); d['blk_share_train']=d.train_block_true/d.train_label_pm; d['test_blk/pred']=d.test_block_rows/d.G1_pred
d['train_blk_ratio']=d.test_block_rows/d.train_block_rows_pm
print(d.round(2).to_string())
