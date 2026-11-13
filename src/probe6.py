"""Targeted October checks per fraud type: have amount shapes, chain cuts or bait sizes changed?"""
import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
al = load_all(); F = pd.read_pickle(os.path.join(CACHE, 'feats.pkl'))
oof = np.load(os.path.join(CACHE, 'oof_C4.npy')); pt = np.load(os.path.join(CACHE, 'test_C4.npy'))
te = al.is_test.values; month = al.timestamp.str[:7].values; P = oof.copy(); P[te] = pt
sep = np.isin(month, ['2026-08', '2026-09']) & ~te
amt = al.amount.values; p2p = al.receiver_type.values == 'P2P'; tt = al.txn_type.values; isint = F.amt_int.values == 1
print('== class probability buckets: (Aug+Sep)/4 vs Oct')
bk = [0.02, 0.1, 0.2, 0.5, 0.8, 1.01]; out = {}
for k in range(1, len(LABELS)):
    a = np.histogram(P[sep, k], bk)[0] / 4; b = np.histogram(P[te, k], bk)[0]; out[LABELS[k]] = [f'{x:.0f}/{z}' for x, z in zip(a, b)]
print(pd.DataFrame(out, index=['.02-.1', '.1-.2', '.2-.5', '.5-.8', '.8-1']).T.to_string())

def cmp(mask, vals, bins, title):
    a = np.histogram(vals[mask & sep], bins)[0] / 4; b = np.histogram(vals[mask & te], bins)[0]
    print(f'\n== {title}'); print(pd.DataFrame({'exp': a.round(1), 'oct': b, 'z': ((b - a) / np.sqrt(a + 1)).round(1)}, index=[f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)]).T.to_string())
cmp(p2p & (tt == 'PAY') & isint, amt, [1, 2, 6, 11, 21, 51, 101, 201], 'P2P PAY integer small amounts (cashback bait?)')
cmp(p2p & (tt == 'PAY') & ~isint, amt, [0, 2, 6, 11, 21, 51, 101], 'P2P PAY non-integer small amounts')
cmp(p2p & (tt == 'COLLECT') & isint, amt % 1000, [0, 1, 100, 400, 500, 501, 900, 950, 990, 999, 1000], 'P2P COLLECT integer: amount mod 1000')
cmp(p2p & (tt == 'COLLECT') & isint, amt, [0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 1e7], 'P2P COLLECT integer: amount')
cmp(p2p & (tt == 'COLLECT') & isint & (al.new_payee.values == 1), amt % 1000, [0, 1, 100, 400, 500, 501, 900, 950, 990, 999, 1000], 'P2P COLLECT integer new payee: mod 1000')
cmp(p2p & (tt == 'PAY') & isint & (amt >= 5000), amt % 1000, [0, 1, 100, 400, 500, 501, 900, 950, 990, 999, 1000], 'P2P PAY integer >=5000: mod 1000')
mer = ~p2p
cmp(mer & isint, amt % 500, [0, 1, 100, 250, 400, 499, 500], 'MERCHANT integer: amount mod 500')
cmp(mer & isint & (amt % 500 == 0), amt, [0, 501, 1001, 2001, 5001, 10001, 1e7], 'MERCHANT round500: amount')
cmp(mer & isint & (amt % 100 == 0) & (amt % 500 != 0), amt, [0, 501, 1001, 2001, 5001, 10001, 1e7], 'MERCHANT round100 not 500: amount')
cmp(mer & isint & (al.merchant_category.values == 'lending'), F.hour.values.astype(float), [0, 6, 12, 17, 19, 21, 24], 'lending integer payments by hour')
cmp(mer & (al.merchant_category.values == 'lending'), F.hour.values.astype(float), [0, 6, 12, 17, 19, 21, 24], 'lending all payments by hour')
cmp(tt == 'REFUND', F.snd_ref_30d.values.astype(float), [1, 2, 3, 4, 6, 10, 100], 'REFUND rows by sender refund count in +-30d')
# chains: large P2P payment whose receiver pays on within 2h
r = F.next_out_ratio.values.astype(float); gap = F.next_out_gap.values.astype(float)
big = p2p & (amt >= 5000) & isint & (gap <= 7200)
cmp(big, r, [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0, 1.0001, 1.2, 100], 'big P2P with onward payment <2h: onward/this amount ratio')
cmp(big & (r > 0.7) & (r <= 1.0), gap, [0, 60, 300, 900, 1800, 3600, 7200], 'chain-like links: onward gap seconds')
cmp(p2p & isint & (amt >= 20000), amt, [20000, 30000, 50000, 80000, 120000, 200000, 1e8], 'P2P integer large amounts')
# takeover-like sessions: new device rows
cmp((al.device_age_days.values < 2) & p2p, amt, [0, 500, 1000, 2000, 5000, 10000, 1e7], 'new-device P2P rows: amount')
cmp((al.sim_age_hours.values < 72) & p2p, F.hour.values.astype(float), [0, 5, 9, 18, 22, 24], 'fresh-SIM P2P rows by hour')
cmp(p2p, al.device_age_days.values, [0, 0.5, 1, 2, 3, 5, 10], 'P2P rows by device_age_days (small)')
cmp(p2p, al.sim_age_hours.values, [0, 12, 24, 48, 72, 120, 240], 'P2P rows by sim_age_hours (small)')
cmp(np.ones(len(al), bool), al.login_to_txn_sec.values.astype(float), [0, 2, 3, 5, 8, 12, 20], 'all rows by login seconds (small)')
cmp(np.ones(len(al), bool), al.pin_retries.values.astype(float), [0, 1, 2, 3, 4, 10], 'all rows by pin retries')
