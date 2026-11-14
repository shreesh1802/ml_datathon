import numpy as np,pandas as pd,re
src=open('scan1.py').read()
exec(src[:src.index("leg=pred==0")])
dn=al.device_id.str[2:].astype(int).values
fr=(pred>0)&(dn<420000)&tr
print('rows predicted fraud on ordinary phones (train):',fr.sum(),'| label==pred',np.mean(y[fr]==pred[fr]).round(3),'| legit-labelled',np.mean(y[fr]==0).round(3))
rows=[]
for nm,v in D.items():
    d=pd.DataFrame({'p':pred[fr],'v':v[fr],'y':y[fr]})
    for (p,val),g in d.groupby(['p','v']):
        if len(g)<25: continue
        vc=g.y.value_counts(); top=vc.index[0]; base=d[d.p==p]
        ok=(g.y==p).mean(); okb=(base.y==p).mean(); lg=(g.y==0).mean(); lgb=(base.y==0).mean()
        if top!=p and top!=0 and vc.iloc[0]/len(g)>0.5: rows.append(('WRONG TYPE',nm,val,LABELS[p],len(g),LABELS[top],round(vc.iloc[0]/len(g),2),round(ok,2)))
        elif lg>0.5 and lg>lgb+0.25: rows.append(('MOSTLY LEGIT',nm,val,LABELS[p],len(g),'legit',round(lg,2),round(ok,2)))
out=pd.DataFrame(rows,columns=['kind','feature','value','predicted','rows','majority label','majority share','pred correct'])
print(out.sort_values(['kind','rows'],ascending=[True,False]).head(60).to_string(index=False))
# how many test rows fall in the listed cells
tfr=(pred>0)&(dn<420000)&te
tot=0
for _,r in out.iterrows():
    m=tfr&(pred==L[r.predicted])&(D[r.feature]==r.value); tot+=m.sum()
print('test rows in listed cells (with overlap):',tot)
