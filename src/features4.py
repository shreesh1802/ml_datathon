"""Upgraded value-set, balance-clamp, and exact simulator cutoff features (no cross-month ring features).
Writes cache/feats4.pkl (or cache/feats4_m.pkl when MONTHLY=1).
Usage:
    python src/features4.py
    MONTHLY=1 python src/features4.py
"""
import os, time
import numpy as np
import pandas as pd
from common import load_all, Win, CACHE

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

al = load_all()
n = len(al)
t = al.t.values.astype(np.int64)
amt = al.amount.values.astype(np.float64)
bal = al.balance_before.values.astype(np.float64)
p2p = (al.receiver_type.values == 'P2P')
mer = (al.receiver_type.values == 'MERCHANT')
tt = al.txn_type.values
newp = (al.new_payee.values == 1)
sim = al.sim_age_hours.values
dag = al.device_age_days.values
lgn = al.login_to_txn_sec.values
hr  = al.ts.dt.hour.values

MONTHLY = os.environ.get('MONTHLY') == '1'
codes, uniq = pd.factorize(pd.concat([al.sender_id, al.receiver_id], ignore_index=True))
S = codes[:n].astype(np.int64)
R = codes[n:].astype(np.int64)
NE = len(uniq)
Dv = pd.factorize(al.device_id)[0].astype(np.int64)

if MONTHLY:
    _m = al.timestamp.str[:7].values
    mc = pd.factorize(np.where(_m >= '2026-10', '2026-10', _m))[0].astype(np.int64)
    NM = mc.max() + 1
    S = S * NM + mc
    R = R * NM + mc
    Dv = Dv * NM + mc
    NE = NE * NM

PAIR = S * NE + R
RPAIR = R * NE + S
p_code = pd.factorize(np.concatenate([PAIR, RPAIR]))[0]
P_c = p_code[:n].astype(np.int64)
RP_c = p_code[n:].astype(np.int64)

H, Dd = 3600, 86400
WS = Win(S, t)
WR = Win(R, t)
WD = Win(Dv, t)
WP = Win(P_c, t)

G = {}

# 1. Amount modulo & exact phishing-limit features (kept from features3.py, with orthogonal bait split)
amt_int = (amt == np.floor(amt))
int_div10 = amt_int & (amt % 10 == 0)
int_nd10 = amt_int & (amt % 10 != 0)
int_nd10_gt10 = int_nd10 & (amt > 10)
int_nd10_ge500 = int_nd10 & (amt >= 500)

exact_phish_amt = np.isin(amt, [1950., 1990., 1999., 4950., 4990., 4999., 9950., 9990., 9999., 24950., 24990., 24999.])
lend_int_non500 = (al.merchant_category.values == 'lending') & amt_int & (~np.isin(amt, [1000., 2000., 2500., 5000., 10000.]))

tiny_f = (amt <= 10).astype(float)
rev_tiny_7d = WP.sum(tiny_f, RP_c, t, -7 * Dd, 7 * Dd, 'rt7')

G['amt_div10'] = int_div10.astype(np.int8)
G['amt_nd10'] = int_nd10.astype(np.int8)
G['amt_nd10_gt10'] = int_nd10_gt10.astype(np.int8)
G['amt_nd10_ge500'] = int_nd10_ge500.astype(np.int8)
G['p2p_nd10_ge500'] = (p2p & int_nd10_ge500).astype(np.int8)
G['exact_phish_amt'] = exact_phish_amt.astype(np.int8)
G['p2p_qr_nd10'] = (p2p & (tt == 'QR') & int_nd10_ge500).astype(np.int8)
G['lend_int_non500'] = lend_int_non500.astype(np.int8)

# Orthogonal phishing_collect vs fake_cashback features:
# - col_nd10_nobait: 150 phishing vs 1 fake_cashback on train
# - repay_nd10_hasbait: 237 fake_cashback vs 119 phishing on train
# - repay_hasbait_oldp & rev_oldpayee_repay_7d: 43 fake_cashback vs 0 phishing on train (100% fake_cashback)
G['col_nd10_nobait'] = (p2p & (tt == 'COLLECT') & (~exact_phish_amt) & int_nd10_ge500 & (rev_tiny_7d == 0)).astype(np.int8)
G['col_exact_phish_nobait'] = (p2p & (tt == 'COLLECT') & exact_phish_amt & (rev_tiny_7d == 0)).astype(np.int8)
G['repay_nd10_hasbait'] = (p2p & (~exact_phish_amt) & int_nd10_ge500 & (rev_tiny_7d >= 1)).astype(np.int8)
G['repay_div10_hasbait'] = (p2p & int_div10 & (amt >= 500) & (rev_tiny_7d >= 1)).astype(np.int8)
G['repay_hasbait_oldp'] = (p2p & (amt >= 500) & (rev_tiny_7d >= 1) & (~newp)).astype(np.int8)
G['rev_oldpayee_repay_7d'] = np.minimum(WP.sum(((amt >= 500) & (~newp)).astype(float), RP_c, t, 1, 7 * Dd, 'rold'), 10).astype(np.int8)

# 2. Exact balance_before vs amount clamps (hidden by max_bin=127 on amt_share_bal and bal_after)
bal_eq_110 = np.abs(bal - np.round(amt * 1.10, 2)) <= 0.015
bal_eq_102 = np.abs(bal - np.round(amt * 1.02, 2)) <= 0.015
bal_lt_amt = bal < amt
bal_ratio = bal / np.maximum(amt, 0.01)
bal_ratio_100_110 = (bal_ratio >= 1.000) & (bal_ratio < 1.0995) & (~bal_eq_102)

G['bal_eq_110'] = bal_eq_110.astype(np.int8)
G['bal_eq_102'] = bal_eq_102.astype(np.int8)
G['bal_lt_amt'] = bal_lt_amt.astype(np.int8)
G['bal_ratio_100_110'] = bal_ratio_100_110.astype(np.int8)

# Window propagations of exact balance clamps
b110_f = bal_eq_110.astype(float)
b102_f = bal_eq_102.astype(float)
blt_f  = bal_lt_amt.astype(float)

G['snd_bal110_1d'] = np.minimum(WS.sum(b110_f, S, t, -Dd, Dd, 'b110'), 20)
G['rcv_bal110_2d'] = np.minimum(WR.sum(b110_f, R, t, -2 * Dd, 2 * Dd, 'b110'), 20)
G['snd_bal102_1d'] = np.minimum(WS.sum(b102_f, S, t, -Dd, Dd, 'b102'), 20)
G['rcv_bal102_1d'] = np.minimum(WR.sum(b102_f, R, t, -Dd, Dd, 'b102'), 20)
G['snd_ballt_1d']  = np.minimum(WS.sum(blt_f, S, t, -Dd, Dd, 'blt'), 20)
G['rcv_ballt_1d']  = np.minimum(WR.sum(blt_f, R, t, -Dd, Dd, 'blt'), 20)

# 3. Exact simulator step-function boundaries (sim<=30, dag<=1.5, lgn<=6, lgn 3..25, hr 0..4, hr 20..23)
sim_le_30 = (sim <= 30.0)
dag_le_1p5 = (dag <= 1.5)
lgn_le_6 = (lgn <= 6)
lgn_3_to_25 = (lgn >= 3) & (lgn <= 25)
hr_0_to_4 = (hr <= 4)
hr_20_to_23 = (hr >= 20)

vb_exact_box = mer & np.isin(al.merchant_category.values, ['ecommerce', 'gaming']) & amt_int & (amt >= 10) & (amt <= 499) & lgn_le_6
loan_exact_box = (al.merchant_category.values == 'lending') & amt_int & (amt >= 1500) & (amt <= 15000) & hr_20_to_23

G['sim_le_30'] = sim_le_30.astype(np.int8)
G['dag_le_1p5'] = dag_le_1p5.astype(np.int8)
G['lgn_le_6'] = lgn_le_6.astype(np.int8)
G['lgn_3_to_25'] = lgn_3_to_25.astype(np.int8)
G['hr_0_to_4'] = hr_0_to_4.astype(np.int8)
G['hr_20_to_23'] = hr_20_to_23.astype(np.int8)
G['vb_exact_box'] = vb_exact_box.astype(np.int8)
G['loan_exact_box'] = loan_exact_box.astype(np.int8)

# Session & window counts with the exact simulator cutoffs
nd10_f = int_nd10_gt10.astype(float)
phish_f = (p2p & (tt == 'COLLECT') & newp & (exact_phish_amt | int_nd10_ge500)).astype(float)
lend_f = lend_int_non500.astype(float)
loan_box_f = loan_exact_box.astype(float)
vb_box_f = vb_exact_box.astype(float)
sim30_f = sim_le_30.astype(float)
dag15_f = dag_le_1p5.astype(float)
hr04_f  = hr_0_to_4.astype(float)

for nm, w in [('1h', H), ('6h', 6 * H), ('1d', Dd), ('7d', 7 * Dd), ('30d', 30 * Dd)]:
    G[f'snd_nd10_{nm}'] = np.minimum(WS.sum(nd10_f, S, t, -w, w, 'nd10'), 30)
    G[f'rcv_nd10_{nm}'] = np.minimum(WR.sum(nd10_f, R, t, -w, w, 'nd10'), 30)
    G[f'rcv_phish_exact_{nm}'] = np.minimum(WR.sum(phish_f, R, t, -w, w, 'phish'), 30)
    G[f'snd_lend_int_{nm}'] = np.minimum(WS.sum(lend_f, S, t, -w, w, 'lend_i'), 30)
    G[f'snd_loan_box_{nm}'] = np.minimum(WS.sum(loan_box_f, S, t, -w, w, 'lbox'), 30)

G['dev_nd10_1d'] = np.minimum(WD.sum(nd10_f, Dv, t, -Dd, Dd, 'nd10'), 30)
G['dev_vb_box_10m'] = np.minimum(WD.sum(vb_box_f, Dv, t, -600, 600, 'vbox'), 30)
G['snd_vb_box_1d'] = np.minimum(WS.sum(vb_box_f, S, t, -Dd, Dd, 'vbox'), 30)
G['ses_sim_le30_1d'] = np.minimum(WS.sum(sim30_f, S, t, -Dd, Dd, 's30'), 15)
G['ses_dag_le1p5_1d'] = np.minimum(WS.sum(dag15_f, S, t, -Dd, Dd, 'd15'), 15)
G['ses_hr04_1d'] = WS.sum(hr04_f, S, t, -Dd, Dd, 'h04') / np.maximum(WS.count(S, t, -Dd, Dd), 1)

# Non-500 integer merchant burst count (merchant_collusion minority variant: MR005718, MR009316, MR007988)
mer_int_1k10k = (mer & (al.merchant_category.values != 'lending') & amt_int & (amt >= 900) & (amt <= 10000)).astype(float)
G['rcv_mer_int1k_2d'] = np.minimum(WR.sum(mer_int_1k10k, R, t, -2 * Dd, 2 * Dd, 'mi1k'), 40)
G['pair_mer_int1k_2d'] = np.minimum(WP.sum(mer_int_1k10k, P_c, t, -2 * Dd, 2 * Dd, 'pmi1k'), 20)

out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in G.items()})
fname = 'feats4_m.pkl' if MONTHLY else 'feats4.pkl'
out.to_pickle(os.path.join(CACHE, fname))
log('saved', fname, out.shape)
