import numpy as np,pandas as pd
from common import load_all,CACHE,LABELS
al=load_all(); te=al.is_test.values; lab=al.label.values
dn=al.device_id.str[2:].astype(int).values
idx=np.arange(len(al)); nt=(~te).sum(); print('train rows',nt,'test rows',te.sum(), 'train first?', (~te[:nt]).all())
ti=idx[te]-nt; n=te.sum()
# test: legit extra phones vs file position
m=te&(dn>=474552); pos=(idx[m]-nt)/n; d=dn[m]
print('test legit extra phones: corr(pos, id)=',np.corrcoef(pos,d)[0,1].round(3))
for k in [2,4,8]:
    print(k,'chunks of file -> id range per chunk:',[(d[(pos>=i/k)&(pos<(i+1)/k)].min(),d[(pos>=i/k)&(pos<(i+1)/k)].max()) for i in range(k)])
# train: same
m=(~te)&(dn>=428455)&(dn<473632); pos=idx[m]/nt; d=dn[m]
print('train legit extra phones: corr=',np.corrcoef(pos,d)[0,1].round(3))
for k in [2,6,12]:
    print(k,'chunks:',[(d[(pos>=i/k)&(pos<(i+1)/k)].min(),d[(pos>=i/k)&(pos<(i+1)/k)].max()) for i in range(k)])
# what differs between halves of test: time? fraud blocks?
t=al.ts.values
h=(idx-nt)>=n//2
print('test half1 time range',al.timestamp[te&~h].min(),al.timestamp[te&~h].max(),' half2',al.timestamp[te&h].min(),al.timestamp[te&h].max())
print('fraud-range phones by half:', (te&~h&(dn>=473632)&(dn<474552)).sum(), (te&h&(dn>=473632)&(dn<474552)).sum(), ' pool', (te&~h&(dn>=420000)&(dn<422978)).sum(), (te&h&(dn>=420000)&(dn<422978)).sum())
