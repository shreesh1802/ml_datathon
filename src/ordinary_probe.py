"""Decisive test: can the ordinary-looking fraud-labelled rows be told apart from legit rows at all?
Train on Jul-Aug, score September. AUC near 0.5 means no."""
import os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import load_all, CACHE, LABELS
al = load_all(); F = pd.read_pickle(os.path.join(CACHE, 'feats.pkl'))
F['ip_raw'] = al.ip_risk_score.values; F['app_version'] = al.app_version.values
y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
yc = np.load(os.path.join(CACHE, 'y_c2.npy')); month = al.timestamp.str[:7].values; te = al.is_test.values
ordn = (y > 0) & (yc == 0); leg = (y == 0) & (yc == 0)
rng = np.random.default_rng(0); pool = (ordn | (leg & (rng.random(len(al)) < 0.2))) & ~te & (month < '2026-10')
tr = pool & (month < '2026-09'); va = pool & (month == '2026-09')
X = F.values.astype(np.float32); t = ordn.astype(int)
m = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=31, feature_fraction=0.5, verbose=-1, min_data_in_leaf=100),
              lgb.Dataset(X[tr], t[tr]), 200)
p = m.predict(X[va]); print('ordinary-fraud vs legit, Sep AUC:', round(roc_auc_score(t[va], p), 4), ' rows', va.sum(), ' positives', t[va].sum())
imp = pd.Series(m.feature_importance('gain'), index=F.columns).sort_values(ascending=False).head(8); print(imp.round(0).to_dict())
# which class is an ordinary-fraud row given? 11-way, chance = majority share
tr2 = ordn & ~te & (month < '2026-09'); va2 = ordn & ~te & (month == '2026-09')
m2 = lgb.train(dict(objective='multiclass', num_class=12, learning_rate=0.05, num_leaves=31, feature_fraction=0.5, verbose=-1, min_data_in_leaf=50),
               lgb.Dataset(X[tr2], y[tr2]), 150)
acc = (m2.predict(X[va2]).argmax(1) == y[va2]).mean(); maj = pd.Series(y[va2]).value_counts(normalize=True).iloc[0]
print('type of ordinary-fraud row: accuracy', round(acc, 4), ' majority-class baseline', round(maj, 4))
