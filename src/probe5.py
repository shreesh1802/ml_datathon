"""Find adapted fraud in October: among rows the model calls legit, which signature cells have more rows than
September (halved) would predict?"""
import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
al = load_all(); F = pd.read_pickle(os.path.join(CACHE, 'feats.pkl'))
oof = np.load(os.path.join(CACHE, 'oof_C2.npy')); pt = np.load(os.path.join(CACHE, 'test_C2.npy'))
te = al.is_test.values; month = al.timestamp.str[:7].values
P = oof.copy(); P[te] = pt
L = {l: i for i, l in enumerate(LABELS)}
y = np.load(os.path.join(CACHE, 'y_c1.npy'))
d = pd.DataFrame({'rt': al.receiver_type.values, 'tt': al.txn_type.values, 'cat': al.merchant_category.fillna('-').values,
                  'int': F.amt_int.values, 'own': F.own_dev.values, 'fast': (al.login_to_txn_sec.values < 8).astype(int), 'newp': al.new_payee.values,
                  'qr': np.where(np.isnan(al.qr_vpa_mismatch.values), -1, al.qr_vpa_mismatch.values).astype(int),
                  'r500': F.round500.values, 'tiny': F.tiny.values, 'away': F.away.values, 'devn': (al.device_age_days.values < 2).astype(int),
                  'simf': (al.sim_age_hours.values < 72).astype(int)})
per = np.where(te, 'OCT', np.where(np.isin(month, ['2026-08', '2026-09']), 'REF', 'x'))
legit_pred = P[:, 0] > 0.5
print('share of integer amounts among predicted-legit rows by receiver type and period:')
print(d[legit_pred & (per != 'x')].assign(per=per[legit_pred & (per != 'x')]).groupby(['per', 'rt', 'tt'])['int'].agg(['mean', 'sum', 'size']).round(4).to_string())
def scan(keys, mask, title, top=25):
    m = mask & (per != 'x')
    g = d[m].assign(per=per[m]).groupby(keys + ['per']).size().unstack(fill_value=0)
    g['exp'] = g.get('REF', 0) / 4; g['oct'] = g.get('OCT', 0); g['z'] = (g.oct - g.exp) / np.sqrt(g.exp + 1)
    print(f'\n== {title}'); print(g[(g.oct >= 15) | (g.exp >= 15)].sort_values('z', ascending=False)[['exp', 'oct', 'z']].head(top).round(1).to_string())
    print('  ...most negative:'); print(g[(g.exp >= 15)].sort_values('z')[['exp', 'oct', 'z']].head(8).round(1).to_string())
scan(['rt', 'tt', 'int', 'own', 'fast'], legit_pred, 'predicted-legit rows: type x integer x own-device x fast-login')
scan(['rt', 'tt', 'qr', 'int', 'newp'], legit_pred, 'predicted-legit rows: type x qr flag x integer x new payee')
scan(['cat', 'tt', 'int', 'r500'], legit_pred & (d.rt.values == 'MERCHANT'), 'predicted-legit merchant rows: category x type x integer x round500')
scan(['rt', 'tt', 'devn', 'simf', 'away', 'own'], legit_pred, 'predicted-legit: takeover signals')
scan(['rt', 'tt', 'int', 'tiny', 'newp'], legit_pred & (d.rt.values == 'P2P'), 'predicted-legit P2P: integer/tiny/new payee')
# all rows regardless of prediction, for the fraud-heavy cells
scan(['rt', 'tt', 'int', 'own', 'fast'], np.ones(len(d), bool), 'ALL rows: type x integer x own-device x fast-login', top=30)
# what fraction of integer-amount rows are fraud (clean labels) in train, by cell
tr = ~te & (month < '2026-10')
g = d[tr].assign(fr=(y[tr] > 0)).groupby(['rt', 'tt', 'int']).fr.agg(['mean', 'size']); print('\nfraud share (clean labels) by receiver type x txn type x integer amount'); print(g.round(4).to_string())
