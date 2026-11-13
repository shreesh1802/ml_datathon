import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 200)
al = load_all(); F = pd.read_pickle(os.path.join(CACHE, 'feats.pkl'))
y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof = np.load(os.path.join(CACHE, 'oof_C2.npy')); pt = np.load(os.path.join(CACHE, 'test_C2.npy'))
te = al.is_test.values; month = al.timestamp.str[:7].values; tr = ~te & (month < '2026-10')
P = oof.copy(); P[te] = pt; top = P.argmax(1); conf = P.max(1)
L = {l: i for i, l in enumerate(LABELS)}
cols = ['amount', 'login_to_txn_sec', 'own_dev', 'new_payee', 'dev_cnt_10m', 'dev_n_acc', 'rcv_cnt_1h', 'rcv_cnt_6h', 'rcv_n_snd_6h', 'rcv_out_cnt_2h', 'rcv_out_ratio_2h', 'snd_cnt_1h', 'amt_int', 'device_age_days', 'r_acc_age', 'rcv_month_cnt', 'rcv_newp_6h', 'rcv_fast_6h', 'rcv_own_6h']
def desc(m, name):
    d = F.loc[m, cols].median().round(2).to_dict()
    print(f'\n{name}: n={m.sum()}  P2P share {(al.receiver_type.values[m] == "P2P").mean():.2f}  types {pd.Series(al.txn_type.values[m]).value_counts(normalize=True).round(2).to_dict()}')
    print('  cat', pd.Series(al.merchant_category.fillna('p2p').values[m]).value_counts(normalize=True).round(2).head(4).to_dict()); print('  ', d)
desc(tr & (top == L['mule_account']) & (conf > .5) & (y == L['mule_account']), 'mule (label mule)')
desc(tr & (top == L['mule_account']) & (conf > .5) & (y == L['velocity_bot']), 'model mule, label velocity_bot')
desc(tr & (top == L['mule_account']) & (conf > .5) & (y == L['layering_chain']), 'model mule, label layering')
desc(tr & (top == L['layering_chain']) & (conf > .5) & (y == L['mule_account']), 'model layering, label mule')
desc(tr & (top == L['velocity_bot']) & (conf > .5) & (y == L['velocity_bot']), 'bot (label bot)')
desc(te & (top == L['mule_account']) & (conf > .5), 'OCT model mule')
desc(te & (top == L['velocity_bot']) & (conf > .5), 'OCT model bot')
# mule bursts: per receiver-burst label composition in train
m = tr & (top == L['mule_account']) & (conf > .5)
d = pd.DataFrame({'r': al.receiver_id.values[m], 'day': al.t.values[m] // 86400, 'lab': np.array(LABELS)[y[m]], 'snd': al.sender_id.values[m]})
g = d.groupby(['r', 'day']); s = pd.DataFrame({'n': g.size(), 'n_vb': g.lab.apply(lambda x: (x == 'velocity_bot').sum()), 'n_mule': g.lab.apply(lambda x: (x == 'mule_account').sum()), 'n_legit': g.lab.apply(lambda x: (x == 'legit').sum())})
print('\nmule-predicted receiver-days: label composition'); print(pd.crosstab(pd.cut(s.n_vb / s.n, [-.01, 0, .3, .7, .99, 1]), pd.cut(s.n, [0, 2, 5, 10, 100])).to_string())
# mule-predicted burst counts per month: receivers
for nm, mm in [('Jul', tr & (month == '2026-07')), ('Aug', tr & (month == '2026-08')), ('Sep', tr & (month == '2026-09')), ('OCT', te)]:
    q = mm & (top == L['mule_account']) & (conf > .5)
    rb = pd.Series(al.receiver_id.values[q] + '_' + (al.t.values[q] // 86400).astype(str))
    vc = rb.value_counts()
    print(nm, 'mule rows', q.sum(), ' receiver-days', len(vc), ' rows/burst median', vc.median(), ' P2P share', round((al.receiver_type.values[q] == 'P2P').mean(), 2),
          ' median amount', np.median(al.amount.values[q]).round(0), ' login med', np.median(al.login_to_txn_sec.values[q]), ' own', F.own_dev.values[q].mean().round(2),
          ' burst sizes', vc.value_counts().sort_index().head(8).to_dict())
# Oct: QR mismatch clusters
qm = (al.qr_vpa_mismatch.values == 1)
for nm, mm in [('Sep', tr & (month == '2026-09')), ('OCT', te)]:
    q = mm & qm
    vc = pd.Series(al.receiver_id.values[q] + '_' + (al.t.values[q] // (86400)).astype(str)).value_counts()
    print(nm, 'mismatch rows', q.sum(), ' receiver-days', len(vc), ' size dist', vc.value_counts().sort_index().head(10).to_dict(), ' predicted qr_swap share of mismatch rows', round((top[q] == L['qr_swap']).mean(), 3),
          ' receiver type', pd.Series(al.receiver_type.values[q]).value_counts().to_dict())
    one = pd.Series(al.receiver_id.values[q] + '_' + (al.t.values[q] // (86400)).astype(str)).map(vc).values
    print('   predicted-qr share by cluster size:', pd.Series(top[q] == L['qr_swap']).groupby(np.minimum(one, 6)).mean().round(2).to_dict())
