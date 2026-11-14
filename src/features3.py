"""Build amount-modulo (int_not_div10), exact phishing-limit, and cross-role Fraud Ring features.
Writes cache/feats3.pkl (or cache/feats3_m.pkl when MONTHLY=1).
Usage:
    python src/features3.py
    MONTHLY=1 python src/features3.py
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
p2p = (al.receiver_type.values == 'P2P')
mer = (al.receiver_type.values == 'MERCHANT')
tt = al.txn_type.values
newp = (al.new_payee.values == 1)
dn = al.device_id.str[2:].astype(np.int64).values

MONTHLY = os.environ.get('MONTHLY') == '1'
codes, uniq = pd.factorize(pd.concat([al.sender_id, al.receiver_id], ignore_index=True))
S0 = codes[:n].astype(np.int64)
R0 = codes[n:].astype(np.int64)
NE0 = len(uniq)
Dv0 = pd.factorize(al.device_id)[0].astype(np.int64)

S, R, Dv, NE = S0.copy(), R0.copy(), Dv0.copy(), NE0
if MONTHLY:
    _m = al.timestamp.str[:7].values
    mc = pd.factorize(np.where(_m >= '2026-10', '2026-10', _m))[0].astype(np.int64)
    NM = mc.max() + 1
    S = S * NM + mc
    R = R * NM + mc
    Dv = Dv * NM + mc
    NE = NE * NM

G = {}

# 1. Exact amount divisibility & phishing limit tiers
amt_int = (amt == np.floor(amt))
int_div10 = amt_int & (amt % 10 == 0)
int_nd10 = amt_int & (amt % 10 != 0)
int_nd10_gt10 = int_nd10 & (amt > 10)
int_nd10_ge500 = int_nd10 & (amt >= 500)

exact_phish_amt = np.isin(amt, [1950., 1990., 1999., 4950., 4990., 4999., 9950., 9990., 9999., 24950., 24990., 24999.])
legit_900s = np.isin(amt, [900., 910., 920., 930., 940., 950., 960., 970., 980.])
lend_int_non500 = (al.merchant_category.values == 'lending') & amt_int & (~np.isin(amt, [1000., 2000., 2500., 5000., 10000.]))

G['amt_div10'] = int_div10.astype(np.int8)
G['amt_nd10'] = int_nd10.astype(np.int8)
G['amt_nd10_gt10'] = int_nd10_gt10.astype(np.int8)
G['amt_nd10_ge500'] = int_nd10_ge500.astype(np.int8)
G['p2p_nd10_ge500'] = (p2p & int_nd10_ge500).astype(np.int8)
G['p2p_col_nd10_ge500'] = (p2p & (tt == 'COLLECT') & int_nd10_ge500).astype(np.int8)
G['p2p_qr_nd10'] = (p2p & (tt == 'QR') & int_nd10_ge500).astype(np.int8)
G['exact_phish_amt'] = exact_phish_amt.astype(np.int8)
G['legit_900s'] = legit_900s.astype(np.int8)
G['lend_int_non500'] = lend_int_non500.astype(np.int8)

# 2. Window counts of non-divisible-by-10 integers and lending integers
H, Dd = 3600, 86400
WS = Win(S, t)
WR = Win(R, t)
WD = Win(Dv, t)

nd10_f = int_nd10_gt10.astype(float)
phish_f = (p2p & (tt == 'COLLECT') & newp & (exact_phish_amt | int_nd10_ge500)).astype(float)
lend_f = lend_int_non500.astype(float)

for nm, w in [('1h', H), ('6h', 6 * H), ('1d', Dd), ('7d', 7 * Dd), ('30d', 30 * Dd)]:
    G[f'snd_nd10_{nm}'] = np.minimum(WS.sum(nd10_f, S, t, -w, w, 'nd10'), 30)
    G[f'rcv_nd10_{nm}'] = np.minimum(WR.sum(nd10_f, R, t, -w, w, 'nd10'), 30)
    G[f'rcv_phish_exact_{nm}'] = np.minimum(WR.sum(phish_f, R, t, -w, w, 'phish'), 30)
    G[f'snd_lend_int_{nm}'] = np.minimum(WS.sum(lend_f, S, t, -w, w, 'lend_i'), 30)

G['dev_nd10_1d'] = np.minimum(WD.sum(nd10_f, Dv, t, -Dd, Dd, 'nd10'), 30)

# 3. Unsupervised Cross-Role Fraud Ring Account Pool counts (within graph scope S, R)
is_bot_snd = (dn >= 420000) & (dn <= 422977)
is_p2p_qr_rcv = (tt == 'QR') & p2p
is_col_phish_rcv = (tt == 'COLLECT') & newp & (exact_phish_amt | int_nd10_ge500) & p2p

ref_cnt_scope = pd.Series((tt == 'REFUND').astype(int)).groupby(S).transform('sum').values
is_ref_snd = (tt == 'REFUND') & (ref_cnt_scope >= 2)

r500_mer = mer & (al.merchant_category.values != 'lending') & np.isin(amt, [1000., 2000., 2500., 5000., 10000.])
r500_cnt_scope = pd.Series(r500_mer.astype(int)).groupby(S).transform('sum').values
is_r500_snd = r500_mer & (r500_cnt_scope >= 2)

j_nxt = WS.nxt(R, t)
jj_nxt = np.maximum(j_nxt, 0)
is_chain_link = p2p & (j_nxt >= 0) & (t[jj_nxt] - t <= H) & (amt[jj_nxt] / amt > 0.8) & (amt[jj_nxt] / amt <= 1.0) & (amt > 500)

PAIR = S * NE + R
RPAIR = R * NE + S
p_code = pd.factorize(np.concatenate([PAIR, RPAIR]))[0]
P_c = p_code[:n].astype(np.int64)
RP_c = p_code[n:].astype(np.int64)
WP = Win(P_c, t)
rev_amt_7d = WP.sum(amt, RP_c, t, 1, 7 * Dd, 'amt')
is_bait_snd = p2p & (amt <= 10) & (rev_amt_7d >= 100)

# Per-account role counts (indexed by unified entity code S / R)
n_ent = max(S.max(), R.max()) + 1
c_bot = np.bincount(S[is_bot_snd], minlength=n_ent)
c_qr  = np.bincount(R[is_p2p_qr_rcv], minlength=n_ent)
c_pc  = np.bincount(R[is_col_phish_rcv], minlength=n_ent)
c_ref = np.bincount(S[is_ref_snd], minlength=n_ent)
c_r50 = np.bincount(S[is_r500_snd], minlength=n_ent)
c_chn = np.bincount(S[is_chain_link], minlength=n_ent) + np.bincount(R[is_chain_link], minlength=n_ent)
c_bat = np.bincount(S[is_bait_snd], minlength=n_ent)

ring_tot = c_bot + c_qr + c_pc + c_ref + c_r50 + c_chn + c_bat
n_roles = (
    (c_bot >= 1).astype(int) + (c_qr >= 1).astype(int) + (c_pc >= 1).astype(int) +
    (c_ref >= 1).astype(int) + (c_r50 >= 1).astype(int) + (c_chn >= 1).astype(int) + (c_bat >= 1).astype(int)
)

own_rcv = is_p2p_qr_rcv.astype(int) + is_col_phish_rcv.astype(int) + is_chain_link.astype(int)
own_snd = is_bot_snd.astype(int) + is_ref_snd.astype(int) + is_r500_snd.astype(int) + is_chain_link.astype(int) + is_bait_snd.astype(int)

G['rcv_ring_other'] = np.minimum(np.maximum(ring_tot[R] - own_rcv, 0), 30).astype(np.float32)
G['snd_ring_other'] = np.minimum(np.maximum(ring_tot[S] - own_snd, 0), 30).astype(np.float32)
G['rcv_ring_roles'] = n_roles[R].astype(np.int8)
G['snd_ring_roles'] = n_roles[S].astype(np.int8)
G['rcv_was_bot_snd'] = np.minimum(c_bot[R], 10).astype(np.int8)
G['snd_was_qr_pc_rcv'] = np.minimum(c_qr[S] + c_pc[S], 10).astype(np.int8)

out = pd.DataFrame(G)
fname = 'feats3_m.pkl' if MONTHLY else 'feats3.pkl'
out.to_pickle(os.path.join(CACHE, fname))
log('saved', fname, out.shape)
