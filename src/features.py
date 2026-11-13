"""Build row + connection features for train and test together. Writes cache/feats.pkl.
No label is used here, so windows may look both backwards and forwards in time."""
import os, time, warnings
import numpy as np, pandas as pd
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
from common import load_all, Win, DATA, CACHE

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

al = load_all()
n = len(al)
acc = pd.read_csv(os.path.join(DATA, 'accounts.csv')).set_index('account_id')
mer = pd.read_csv(os.path.join(DATA, 'merchants.csv')).set_index('merchant_id')
t = al.t.values.astype(np.int64)
amt = al.amount.values.astype(np.float64)
month = al.timestamp.str[:7].values
period = np.where(month >= '2026-10', '2026-10', month)   # early-November tail joins October
F = pd.DataFrame(index=al.index)

# ---------- entity codes ----------
codes, uniq = pd.factorize(pd.concat([al.sender_id, al.receiver_id], ignore_index=True))
S = codes[:n].astype(np.int64); R = codes[n:].astype(np.int64); NE = len(uniq)
Dv = pd.factorize(al.device_id)[0].astype(np.int64)
MONTHLY = os.environ.get('MONTHLY') == '1'
if MONTHLY:
    # Each calendar month becomes its own graph. Fraud accounts recur across Jul-Sep but October's are fresh,
    # so features must not learn "this account was in a fraud pattern in an earlier month".
    mc = pd.factorize(period)[0].astype(np.int64); NM = mc.max() + 1
    S = S * NM + mc; R = R * NM + mc; Dv = Dv * NM + mc; NE = NE * NM
PAIR = S * NE + R; RPAIR = R * NE + S
pair_code = pd.factorize(np.concatenate([PAIR, RPAIR]))[0]; P = pair_code[:n].astype(np.int64); RP = pair_code[n:].astype(np.int64)
p2p = (al.receiver_type == 'P2P').values
ttype = al.txn_type.values

# ---------- row features ----------
hour = al.ts.dt.hour.values
F['hour'] = hour; F['minute_of_day'] = hour * 60 + al.ts.dt.minute.values; F['weekday'] = al.ts.dt.weekday.values
F['day_of_month'] = al.ts.dt.day.values
for c in ['sim_age_hours', 'device_age_days', 'sender_account_age_days', 'new_payee', 'pin_retries', 'login_to_txn_sec', 'qr_vpa_mismatch']:
    F[c] = al[c].values
F['qr_checked'] = al.qr_vpa_mismatch.notna().astype(np.int8).values
F['amount'] = amt; F['balance_before'] = al.balance_before.values
F['amt_share_bal'] = amt / np.maximum(al.balance_before.values, 1)
F['bal_after'] = al.balance_before.values - amt
g = pd.Series(period)
F['amount_rk'] = pd.Series(amt).groupby(g).rank(pct=True).values
F['balance_rk'] = al.balance_before.groupby(g.values).rank(pct=True).values
F['ip_risk_rk'] = al.ip_risk_score.groupby(g.values).rank(pct=True).values
tt_key = pd.Series(period + '_' + al.receiver_type.values)
F['amount_rk_rtype'] = pd.Series(amt).groupby(tt_key).rank(pct=True).values
F['amt_int'] = (amt == np.floor(amt)).astype(np.int8)
for k in [100, 500, 1000]:
    F[f'round{k}'] = (amt % k == 0).astype(np.int8)
F['mod1000'] = amt % 1000; F['mod100'] = amt % 100
F['under1000'] = ((amt % 1000) >= 900).astype(np.int8)
F['under_x99'] = (np.floor(amt) % 100 == 99).astype(np.int8)
F['tiny'] = (amt <= 10).astype(np.int8)
F['cents'] = np.round((amt - np.floor(amt)) * 100)
for c in ['receiver_type', 'txn_type', 'merchant_category', 'app']:
    F[c] = al[c].fillna('none').astype('category').cat.codes.values
# account / merchant joins
home_s = al.sender_id.map(acc.home_city).values
F['away'] = (al.device_city.values != home_s).astype(np.int8)
F['own_dev'] = (al.device_id.str[3:].values == al.sender_id.str[2:].values).astype(np.int8)
F['s_kyc'] = al.sender_id.map(acc.kyc_level).astype('category').cat.codes.values
F['s_os'] = al.sender_id.map(acc.os).astype('category').cat.codes.values
F['s_bank'] = al.sender_id.map(acc.bank_code).astype('category').cat.codes.values
opened = pd.to_datetime(acc.opened_on)
F['r_acc_age'] = (al.ts - al.receiver_id.map(opened)).dt.days.values
F['r_kyc'] = al.receiver_id.map(acc.kyc_level).astype('category').cat.codes.values
F['r_bank_same'] = (al.receiver_id.map(acc.bank_code).values == al.sender_id.map(acc.bank_code).values).astype(np.int8)
home_r = al.receiver_id.map(acc.home_city).values
F['r_home_same'] = np.where(p2p, (home_r == home_s), np.nan)
F['m_age'] = (al.ts - al.receiver_id.map(pd.to_datetime(mer.onboarded_on))).dt.days.values
mcity = al.receiver_id.map(mer.city).values
F['m_city_dev_same'] = np.where(~p2p, (mcity == al.device_city.values), np.nan)
F['m_city_home_same'] = np.where(~p2p, (mcity == home_s), np.nan)
F['m_cat_mismatch'] = np.where(~p2p, al.receiver_id.map(mer.category).values != al.merchant_category.values, np.nan)
F['city'] = al.device_city.astype('category').cat.codes.values
log('row features', F.shape)

# signals reused in window means
sig = {
    'newp': al.new_payee.values.astype(float), 'int': F.amt_int.values.astype(float), 'r500': F.round500.values.astype(float),
    'r100': F.round100.values.astype(float), 'under': F.under1000.values.astype(float), 'tiny': F.tiny.values.astype(float),
    'fast': (al.login_to_txn_sec.values < 10).astype(float), 'own': F.own_dev.values.astype(float), 'away': F.away.values.astype(float),
    'simf': (al.sim_age_hours.values < 72).astype(float), 'devn': (al.device_age_days.values < 2).astype(float),
    'qrm': (al.qr_vpa_mismatch.values == 1).astype(float), 'ref': (ttype == 'REFUND').astype(float),
    'col': (ttype == 'COLLECT').astype(float), 'amt': amt, 'lamt': np.log1p(amt), 'pin': al.pin_retries.values.astype(float),
    'lend': (al.merchant_category.values == 'lending').astype(float), 'eve': ((hour >= 19)).astype(float),
    'night': ((hour >= 22) | (hour < 5)).astype(float), 'p2p': p2p.astype(float),
    'login': al.login_to_txn_sec.values.astype(float),
}
M, H, Dd = 60, 3600, 86400
BIG = 1 << 24

def per_period_count(code):
    return pd.Series(np.ones(n)).groupby([code, period]).transform('sum').values

# ---------- device ----------
W = Win(Dv, t)
for nm, w in [('2m', 2 * M), ('10m', 10 * M), ('1h', H), ('6h', 6 * H), ('1d', Dd)]:
    F[f'dev_cnt_{nm}'] = W.count(Dv, t, -w, w)
F['dev_cnt_past10m'] = W.count(Dv, t, -10 * M, -1); F['dev_cnt_fut10m'] = W.count(Dv, t, 1, 10 * M)
for nm, w in [('10m', 10 * M), ('1d', Dd)]:
    c = np.maximum(W.count(Dv, t, -w, w), 1)
    for s in ['fast', 'own', 'amt', 'newp', 'login', 'away']:
        v = W.sum(sig[s], Dv, t, -w, w, s)
        F[f'dev_{s}_{nm}'] = v if s == 'amt' else v / c
j = W.prv(Dv, t); F['dev_gap_prev'] = np.where(j >= 0, t - t[np.maximum(j, 0)], np.nan)
j = W.nxt(Dv, t); F['dev_gap_next'] = np.where(j >= 0, t[np.maximum(j, 0)] - t, np.nan)
F['dev_gap_min'] = np.fmin(F.dev_gap_prev, F.dev_gap_next)
F['dev_n_acc'] = pd.Series(S).groupby(Dv).transform('nunique').values
F['dev_n_rows'] = per_period_count(Dv)
F['snd_n_dev'] = pd.Series(Dv).groupby(S).transform('nunique').values
dd = pd.DataFrame({'d': Dv, 's': S, 'b': t // (6 * H)})
F['dev_n_acc_6h'] = dd.groupby(['d', 'b']).s.transform('nunique').values
F['dev_n_rcv_6h'] = pd.DataFrame({'d': Dv, 'r': R, 'b': t // (6 * H)}).groupby(['d', 'b']).r.transform('nunique').values
# is this the sender's usual device / city
sd = pd.Series(np.ones(n)).groupby([S, Dv]).transform('sum').values
s_tot = pd.Series(np.ones(n)).groupby(S).transform('sum').values
F['snd_dev_share'] = sd / s_tot; F['snd_tot'] = s_tot
city_c = al.device_city.astype('category').cat.codes.values
F['snd_city_share'] = pd.Series(np.ones(n)).groupby([S, city_c]).transform('sum').values / s_tot
log('device')

# ---------- sender (outgoing) ----------
WS = Win(S, t)
for nm, w in [('10m', 10 * M), ('1h', H), ('6h', 6 * H), ('1d', Dd), ('7d', 7 * Dd)]:
    c = WS.count(S, t, -w, w); F[f'snd_cnt_{nm}'] = c
    F[f'snd_amt_{nm}'] = WS.sum(amt, S, t, -w, w, 'amt')
    cc = np.maximum(c, 1)
    for s in ['newp', 'simf', 'devn', 'away', 'own', 'fast', 'pin', 'night', 'int', 'p2p']:
        F[f'snd_{s}_{nm}'] = WS.sum(sig[s], S, t, -w, w, s) / cc
F['snd_cnt_past1h'] = WS.count(S, t, -H, -1); F['snd_cnt_fut1h'] = WS.count(S, t, 1, H)
F['snd_amt_fut1d'] = WS.sum(amt, S, t, 1, Dd, 'amt'); F['snd_amt_past1d'] = WS.sum(amt, S, t, -Dd, -1, 'amt')
F['snd_drain_1h'] = F.snd_amt_1h / np.maximum(al.balance_before.values, 1)
F['snd_drain_fut1d'] = (F.snd_amt_fut1d + amt) / np.maximum(al.balance_before.values, 1)
j = WS.prv(S, t); jj = np.maximum(j, 0)
F['snd_gap_prev'] = np.where(j >= 0, t - t[jj], np.nan)
F['snd_bal_jump'] = np.where(j >= 0, al.balance_before.values - (al.balance_before.values[jj] - amt[jj]), np.nan)
F['snd_sim_drop'] = np.where(j >= 0, al.sim_age_hours.values - al.sim_age_hours.values[jj], np.nan)
j = WS.nxt(S, t); jj = np.maximum(j, 0)
F['snd_gap_next'] = np.where(j >= 0, t[jj] - t, np.nan)
for s, nm in [('ref', 'ref'), ('lend', 'lend'), ('col', 'col')]:
    F[f'snd_{nm}_7d'] = WS.sum(sig[s], S, t, -7 * Dd, 7 * Dd, s)
    F[f'snd_{nm}_30d'] = WS.sum(sig[s], S, t, -30 * Dd, 30 * Dd, s)
F['snd_ref_amt_30d'] = WS.sum(sig['ref'] * amt, S, t, -30 * Dd, 30 * Dd, 'refamt')
F['snd_ref_share'] = pd.Series(sig['ref']).groupby(S).transform('mean').values
sgrp = pd.DataFrame({'s': S, 'a': amt, 'sim': al.sim_age_hours.values, 'dag': al.device_age_days.values, 'lg': al.login_to_txn_sec.values}).groupby('s')
med = sgrp.a.transform('median').values
F['snd_amt_med'] = med; F['amt_vs_snd_med'] = amt / np.maximum(med, 1)
F['amt_vs_snd_max'] = amt / sgrp.a.transform('max').values
F['sim_vs_snd_med'] = al.sim_age_hours.values - sgrp.sim.transform('median').values
F['sim_vs_snd_max'] = al.sim_age_hours.values / np.maximum(sgrp.sim.transform('max').values, 1)
F['dag_vs_snd_med'] = al.device_age_days.values - sgrp.dag.transform('median').values
F['login_vs_snd_med'] = al.login_to_txn_sec.values - sgrp.lg.transform('median').values
F['snd_n_rcv'] = pd.Series(R).groupby(S).transform('nunique').values
F['snd_n_rcv_1d'] = pd.DataFrame({'s': S, 'r': R, 'b': t // Dd}).groupby(['s', 'b']).r.transform('nunique').values
log('sender')

# ---------- receiver (incoming) ----------
WR = Win(R, t)
r_month = per_period_count(R)
F['rcv_month_cnt'] = r_month
for nm, w in [('10m', 10 * M), ('1h', H), ('6h', 6 * H), ('1d', Dd), ('7d', 7 * Dd)]:
    c = WR.count(R, t, -w, w); F[f'rcv_cnt_{nm}'] = c
    F[f'rcv_cnt_{nm}_rel'] = c / r_month
    F[f'rcv_amt_{nm}'] = WR.sum(amt, R, t, -w, w, 'amt')
    cc = np.maximum(c, 1)
    for s in ['newp', 'int', 'r500', 'r100', 'under', 'tiny', 'qrm', 'col', 'ref', 'eve', 'lamt', 'simf', 'devn', 'own', 'fast']:
        F[f'rcv_{s}_{nm}'] = WR.sum(sig[s], R, t, -w, w, s) / cc
F['rcv_cnt_past1h'] = WR.count(R, t, -H, -1); F['rcv_cnt_fut1h'] = WR.count(R, t, 1, H)
j = WR.prv(R, t); F['rcv_gap_prev'] = np.where(j >= 0, t - t[np.maximum(j, 0)], np.nan)
j = WR.nxt(R, t); F['rcv_gap_next'] = np.where(j >= 0, t[np.maximum(j, 0)] - t, np.nan)
rg = pd.DataFrame({'r': R, 's': S, 'b6': t // (6 * H), 'bd': t // Dd, 'a': amt, 'p': period})
F['rcv_n_snd_6h'] = rg.groupby(['r', 'b6']).s.transform('nunique').values
F['rcv_n_snd_1d'] = rg.groupby(['r', 'bd']).s.transform('nunique').values
F['rcv_n_snd'] = rg.groupby(['r', 'p']).s.transform('nunique').values
F['rcv_snd_ratio'] = F.rcv_n_snd / r_month
for s in ['r500', 'r100', 'int', 'qrm', 'newp', 'ref', 'tiny', 'col', 'under']:
    F[f'rcv_{s}_month'] = pd.Series(sig[s]).groupby([R, period]).transform('mean').values
F['rcv_amt_med'] = rg.groupby('r').a.transform('median').values
F['amt_vs_rcv_med'] = amt / np.maximum(F.rcv_amt_med.values, 1)
# concentration of the receiver's income in its top sender (per month)
ps = rg.groupby(['r', 'p', 's']).a.transform('size').values
F['rcv_top_snd_share'] = pd.Series(ps).groupby([R, period]).transform('max').values / r_month
F['pair_share_of_rcv'] = ps / r_month
F['rcv_day_peak'] = pd.Series(np.ones(n)).groupby([R, t // Dd]).transform('sum').values / r_month
log('receiver')

# ---------- money moving on: the receiver's own outgoing payments ----------
for nm, w in [('30m', 30 * M), ('2h', 2 * H), ('1d', Dd)]:
    F[f'rcv_out_cnt_{nm}'] = WS.count(R, t, 1, w)
    F[f'rcv_out_amt_{nm}'] = WS.sum(amt, R, t, 1, w, 'amt')
    F[f'rcv_out_ratio_{nm}'] = F[f'rcv_out_amt_{nm}'] / amt
    F[f'rcv_in_amt_{nm}'] = WR.sum(amt, R, t, -w, w, 'amt')
    F[f'rcv_passthru_{nm}'] = WS.sum(amt, R, t, -w, w, 'amt') / np.maximum(F[f'rcv_in_amt_{nm}'], 1)
j = WS.nxt(R, t); jj = np.maximum(j, 0)
F['next_out_gap'] = np.where(j >= 0, t[jj] - t, np.nan)
F['next_out_ratio'] = np.where(j >= 0, amt[jj] / amt, np.nan)
F['next_out_p2p'] = np.where(j >= 0, p2p[jj], np.nan)
# the sender's own incoming payments (is this row a later hop?)
for nm, w in [('30m', 30 * M), ('2h', 2 * H), ('1d', Dd)]:
    F[f'snd_in_cnt_{nm}'] = WR.count(S, t, -w, -1)
    F[f'snd_in_amt_{nm}'] = WR.sum(amt, S, t, -w, -1, 'amt')
    F[f'snd_in_ratio_{nm}'] = amt / np.maximum(F[f'snd_in_amt_{nm}'], 1)
    F[f'snd_in_n_newp_{nm}'] = WR.sum(sig['newp'], S, t, -w, -1, 'newp')
j = WR.prv(S, t); jj = np.maximum(j, 0)
F['prev_in_gap'] = np.where(j >= 0, t - t[jj], np.nan)
F['prev_in_ratio'] = np.where(j >= 0, amt / amt[jj], np.nan)
# chain depth: follow best onward / inward match within 1 hour, amount ratio 0.80-1.0
def chain_link(forward=True):
    if forward:
        j = WS.nxt(R, t); jj = np.maximum(j, 0)
        ok = (j >= 0) & (t[jj] - t <= H) & (amt[jj] / amt > 0.8) & (amt[jj] / amt <= 1.0)
    else:
        j = WR.prv(S, t); jj = np.maximum(j, 0)
        ok = (j >= 0) & (t - t[jj] <= H) & (amt / amt[jj] > 0.8) & (amt / amt[jj] <= 1.0)
    return np.where(ok, j, -1)
for nm, fwd in [('fwd', True), ('bwd', False)]:
    link = chain_link(fwd); depth = np.zeros(n); cur = link.copy()
    for _ in range(8):
        alive = cur >= 0; depth += alive; cur = np.where(alive, link[np.maximum(cur, 0)], -1)
    F[f'chain_{nm}'] = depth
F['chain_len'] = F.chain_fwd + F.chain_bwd
log('flows')

# ---------- pair and reverse pair ----------
WP = Win(P, t)
F['pair_cnt'] = WP.count(P, t, -BIG, BIG)
for nm, w in [('1h', H), ('1d', Dd), ('7d', 7 * Dd)]:
    F[f'pair_cnt_{nm}'] = WP.count(P, t, -w, w); F[f'pair_amt_{nm}'] = WP.sum(amt, P, t, -w, w, 'amt')
F['pair_cnt_past'] = WP.count(P, t, -BIG, -1)
F['newp_but_seen'] = (al.new_payee.values == 1) & (F.pair_cnt_past.values > 0)
j = WP.prv(P, t); F['pair_gap_prev'] = np.where(j >= 0, t - t[np.maximum(j, 0)], np.nan)
j = WP.nxt(P, t); F['pair_gap_next'] = np.where(j >= 0, t[np.maximum(j, 0)] - t, np.nan)
F['rev_cnt'] = WP.count(RP, t, -BIG, BIG)
for nm, lo, hi in [('past1d', -Dd, -1), ('fut1d', 1, Dd), ('past7d', -7 * Dd, -1), ('fut7d', 1, 7 * Dd)]:
    F[f'rev_cnt_{nm}'] = WP.count(RP, t, lo, hi); F[f'rev_amt_{nm}'] = WP.sum(amt, RP, t, lo, hi, 'amt')
    F[f'rev_tiny_{nm}'] = WP.sum(sig['tiny'], RP, t, lo, hi, 'tiny')
F['rev_ratio_past'] = amt / np.maximum(F.rev_amt_past7d, 0.01) * (F.rev_cnt_past7d > 0)
F['rev_ratio_fut'] = F.rev_amt_fut7d / np.maximum(amt, 0.01)
F['pair_tiny_7d'] = WP.sum(sig['tiny'], P, t, -7 * Dd, 7 * Dd, 'tiny')
# sender's repeat payments to lending merchants / same payee in the evening
F['pair_eve_7d'] = WP.sum(sig['eve'], P, t, -7 * Dd, 7 * Dd, 'eve')
log('pairs')

F = F.astype({c: np.float32 for c in F.columns if F[c].dtype == np.float64})
F['newp_but_seen'] = F.newp_but_seen.astype(np.int8)
F.to_pickle(os.path.join(CACHE, 'feats_m.pkl' if MONTHLY else 'feats.pkl'))
log('saved', F.shape, f'{F.memory_usage().sum() / 1e9:.2f} GB')
