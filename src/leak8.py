import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
pd.set_option('display.width',250)
al=load_all(); te=al.is_test.values; lab=al.label.values
dn=al.device_id.str[2:].astype(int).values; t=al.t.values.astype(float); sn=al.sender_id.str[2:].astype(int).values
seg=pd.read_pickle(f'{CACHE}/feats_seg_s2.pkl'); st=seg.seg_state.values.astype(int); cls=seg.seg_class.values.astype(int)
bo=np.load(f'{CACHE}/oof_C5.npy'); bt=np.load(f'{CACHE}/test_C5.npy'); B=bo.copy(); B[te]=bt   # block-blind model
pat=B[:,0]<0.5
from scipy.stats import spearmanr
for run,m0 in [('train',(~te)&(dn>=423001)&(dn<=428454)),('october',te&(dn>=473632)&(dn<=474551))]:
    print('==',run)
    for s in sorted(np.unique(st[m0])):
        m=m0&(st==s)
        if m.sum()<20: continue
        d=pd.DataFrame({'dn':dn[m],'t':t[m],'pat':pat[m],'sn':sn[m]}).groupby('dn').agg(t=('t','min'),pat=('pat','max'),sn=('sn','first')).reset_index()
        pos=(d.dn-d.dn.min())/(d.dn.max()-d.dn.min())
        print(f'{LABELS[cls[m][0]]:20s} phones {len(d):4d} | corr(id,time) {spearmanr(d.dn,d.t)[0]:+.2f} | corr(id,sender no.) {spearmanr(d.dn,d.sn)[0]:+.2f} | patterned share by block quarter {[round(d.pat[(pos>=a)&(pos<=a+.25)].mean(),2) for a in (0,.25,.5,.75)]}')
