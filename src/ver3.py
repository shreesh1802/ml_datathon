import numpy as np,pandas as pd,sys
sys.path.insert(0,r'C:\Users\0095\Downloads')
from common import load_all,CACHE,LABELS
import apply_winning_fixes as W
pd.set_option('display.width',300); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',30)
al=load_all(); te=al.is_test.values; mo=al.timestamp.str[:7].values
tr=(~te); lab=al.label.values
dn=al.device_id.str[2:].astype(int).values
cls=pd.read_pickle(f'{CACHE}/feats_seg_s1.pkl').seg_class.values.astype(int)
sh=[l[:6] for l in LABELS]
ob=W.get_oct_block_label(dn); o=te&(dn>=473632)&(dn<=474551)
ct=pd.crosstab(np.array(sh)[ob[o]],np.array(sh)[cls[o]]); print('rows: THEIR block, cols: OUR seg'); print(ct.to_string())
# training run: label ranges by device number
m=tr&(dn>=422978)&(dn<=428454)
d=pd.DataFrame({'dn':dn[m],'lab':lab[m],'seg':np.array(LABELS)[cls[m]]})
print(d.groupby('seg').dn.agg(['min','max','count','nunique']).sort_values('min').to_string())
# in train-run segments: label mix of sim and ato segments, and rows per device
d['n']=d.groupby('dn').dn.transform('size')
for s in ['sim_swap','account_takeover']:
    x=d[d.seg==s]; print(s,'segment: labels',x.lab.value_counts().head(4).to_dict(),'| single-row phones',(x.groupby('dn').size()==1).mean().round(2),' rows/phones',len(x),x.dn.nunique(), 'id density', round(x.dn.nunique()/(x.dn.max()-x.dn.min()+1),3))
# October segments ours
o2=te&(dn>=473632)&(dn<=474551)
d2=pd.DataFrame({'dn':dn[o2],'seg':np.array(LABELS)[cls[o2]],'sim':al.sim_age_hours.values[o2],'dag':al.device_age_days.values[o2]})
g=d2.groupby('seg').dn.agg(['min','max','count','nunique']).sort_values('min'); g['density']=(g['nunique']/(g['max']-g['min']+1)).round(3); print(g.to_string())
x=d2[(d2.dn>=473899)&(d2.dn<=474279)].copy(); x['n']=x.groupby('dn').dn.transform('size')
x['bin']=pd.cut(x.dn,[473898,473921,473960,474000,474040,474100,474160,474220,474279])
print(x.groupby('bin').agg(rows=('dn','size'),phones=('dn','nunique'),single=('n',lambda s:(s==1).mean()),sim_lt72=('sim',lambda s:(s<72).mean()),dag_lt2=('dag',lambda s:(s<2).mean())).round(2).to_string())
# same profile in train run for sim/ato segments
m=tr&(dn>=422978)&(dn<=428454)
t=pd.DataFrame({'dn':dn[m],'seg':np.array(LABELS)[cls[m]],'sim':al.sim_age_hours.values[m],'dag':al.device_age_days.values[m]}); t['n']=t.groupby('dn').dn.transform('size')
print(t[t.seg.isin(['sim_swap','account_takeover','layering_chain','phishing_collect'])].groupby('seg').agg(rows=('dn','size'),single=('n',lambda s:(s==1).mean()),sim_lt72=('sim',lambda s:(s<72).mean()),dag_lt2=('dag',lambda s:(s<2).mean())).round(2).to_string())
