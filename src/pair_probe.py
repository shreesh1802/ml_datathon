"""Within rows the model assigns to class B, can rows whose given label is A be told apart from rows labelled B?
Train on two months, score the third. AUC near 0.5 = irreducible (or label noise)."""
import os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import load_all, CACHE, LABELS
al = load_all(); te = al.is_test.values; month = al.timestamp.str[:7].values; tr = ~te & (month < '2026-10')
y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof = np.load(os.path.join(CACHE, 'oof_E1.npy')); pred = oof.argmax(1)
F = pd.concat([pd.read_pickle(os.path.join(CACHE, 'feats.pkl')), pd.read_pickle(os.path.join(CACHE, 'feats2.pkl'))], axis=1)
F['ip_raw'] = al.ip_risk_score.values; F['app_version'] = al.app_version.values; F['dev_num'] = al.device_id.str[2:].astype(int).values
F['snd_num'] = al.sender_id.str[2:].astype(int).values; F['rcv_num'] = al.receiver_id.str[2:].astype(int).values; F['second'] = al.ts.dt.second.values
L = {l: i for i, l in enumerate(LABELS)}
pairs = [('sim_swap', ['account_takeover']), ('account_takeover', ['sim_swap']), ('layering_chain', ['mule_account']), ('mule_account', ['velocity_bot']),
         ('phishing_collect', ['fake_cashback']), ('phishing_collect', ['loan_app_extortion']), ('fake_cashback', ['phishing_collect']), ('qr_swap', ['refund_abuse', 'merchant_collusion'])]
for B, As in pairs:
    reg = tr & (pred == L[B]) & np.isin(y, [L[B]] + [L[a] for a in As]) & (F.dev_num.values < 420000)
    t = (y != L[B]).astype(int); aucs = []; imp = np.zeros(F.shape[1])
    for m in ['2026-07', '2026-08', '2026-09']:
        a = reg & (month != m); b = reg & (month == m)
        if t[b].sum() < 5: continue
        mdl = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=15, feature_fraction=0.5, verbose=-1), lgb.Dataset(F[a], t[a]), 150)
        aucs.append(roc_auc_score(t[b], mdl.predict(F[b]))); imp += mdl.feature_importance('gain')
    top = pd.Series(imp, index=F.columns).sort_values(ascending=False).head(6)
    print(f'predicted {B:17s} given {"/".join(a[:8] for a in As):22s} rows {reg.sum():5d}  minority share {t[reg].mean():.3f}  AUC by month {np.round(aucs, 3)}  top: {list(top.index)}')
