"""Extra features: session-level extremes (a victim's drained session shares one label), and count-style cluster signals
that survive October's half volume. Writes cache/feats2.pkl."""
import os, time
import numpy as np, pandas as pd
from common import load_all, Win, CACHE

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def win_min(W, vals, qkey, qt, lo, hi):
    """Minimum of vals over rows with the same key inside the window (NaN if the window is empty)."""
    a, b = W._bounds(qkey, qt, lo, hi)
    qo = W._kc[(id(qkey), id(qt))][0]
    v = np.concatenate([np.nan_to_num(np.asarray(vals, dtype=np.float64), nan=np.inf)[W.order], [np.inf]])
    aa = a[qo].astype(np.int64); bb = b[qo].astype(np.int64)
    idx = np.stack([aa, np.maximum(bb, aa)], 1).ravel()
    red = np.minimum.reduceat(v, idx)[::2]
    red = np.where(bb > aa, red, np.nan)
    out = np.empty(len(aa)); out[qo] = red
    return out.astype(np.float32)


al = load_all(); n = len(al)
F1 = pd.read_pickle(os.path.join(CACHE, 'feats_m.pkl' if os.environ.get('MONTHLY') == '1' else 'feats.pkl'))
t = al.t.values.astype(np.int64); amt = al.amount.values.astype(np.float64)
codes, uniq = pd.factorize(pd.concat([al.sender_id, al.receiver_id], ignore_index=True))
S = codes[:n].astype(np.int64); R = codes[n:].astype(np.int64)
Dv = pd.factorize(al.device_id)[0].astype(np.int64)
MONTHLY = os.environ.get('MONTHLY') == '1'
if MONTHLY:   # see features.py: one graph per calendar month
    _m = al.timestamp.str[:7].values; mc = pd.factorize(np.where(_m >= '2026-10', '2026-10', _m))[0].astype(np.int64); NM = mc.max() + 1
    S = S * NM + mc; R = R * NM + mc; Dv = Dv * NM + mc
p2p = (al.receiver_type == 'P2P').values
M, H, Dd = 60, 3600, 86400
G = {}
WS = Win(S, t); WR = Win(R, t); WD = Win(Dv, t)
sim = al.sim_age_hours.values; dag = al.device_age_days.values; login = al.login_to_txn_sec.values
away = F1.away.values.astype(float); own = F1.own_dev.values.astype(float); pin = al.pin_retries.values.astype(float)
for nm, w in [('2h', 2 * H), ('1d', Dd)]:
    G[f'ses_sim_min_{nm}'] = win_min(WS, sim, S, t, -w, w)
    G[f'ses_dag_min_{nm}'] = win_min(WS, dag, S, t, -w, w)
    G[f'ses_own_min_{nm}'] = win_min(WS, own, S, t, -w, w)
    G[f'ses_away_max_{nm}'] = -win_min(WS, -away, S, t, -w, w)
    G[f'ses_pin_max_{nm}'] = -win_min(WS, -pin, S, t, -w, w)
    G[f'ses_login_min_{nm}'] = win_min(WS, login, S, t, -w, w)
    G[f'ses_amt_max_{nm}'] = -win_min(WS, -amt, S, t, -w, w)
    c = WS.count(S, t, -w, w)
    G[f'ses_n_{nm}'] = np.minimum(c, 8)
    G[f'ses_pos_{nm}'] = np.minimum(WS.count(S, t, -w, -1), 8)
    G[f'ses_left_{nm}'] = np.minimum(WS.count(S, t, 1, w), 8)
    G[f'ses_amt_rel_{nm}'] = amt / G[f'ses_amt_max_{nm}']
    night = ((F1.hour.values >= 22) | (F1.hour.values < 5)).astype(float)
    G[f'ses_night_{nm}'] = WS.sum(night, S, t, -w, w, 'night') / np.maximum(c, 1)
    G[f'ses_sim_lt72_n_{nm}'] = WS.sum((sim < 72).astype(float), S, t, -w, w, 'simf')
    G[f'ses_dag_lt2_n_{nm}'] = WS.sum((dag < 2).astype(float), S, t, -w, w, 'devn')
# sender's history before the session: was the phone / city ever used before by this sender?
sd_first = pd.Series(t).groupby([S, Dv]).transform('min').values
G['dev_first_seen_gap'] = np.minimum(t - sd_first, 30 * Dd)
G['snd_first_seen_gap'] = np.minimum(t - pd.Series(t).groupby(S).transform('min').values, 30 * Dd)
city = al.device_city.astype('category').cat.codes.values
G['city_first_seen_gap'] = np.minimum(t - pd.Series(t).groupby([S, city]).transform('min').values, 30 * Dd)
G['dag_minus_sesmin'] = dag - G['ses_dag_min_1d']; G['sim_minus_sesmin'] = sim - G['ses_sim_min_1d']
# session drain: money leaving in the session relative to the first balance seen
G['ses_amt_sum_1d_bal'] = WS.sum(amt, S, t, -Dd, Dd, 'amt') / np.maximum(-win_min(WS, -al.balance_before.values, S, t, -Dd, Dd), 1)
log('session')

# count-style cluster signals at the receiver
sig = {'qrm': (al.qr_vpa_mismatch.values == 1).astype(float), 'r500': F1.round500.values.astype(float), 'tiny': F1.tiny.values.astype(float),
       'newp': al.new_payee.values.astype(float), 'int': F1.amt_int.values.astype(float), 'col': (al.txn_type.values == 'COLLECT').astype(float),
       'ref': (al.txn_type.values == 'REFUND').astype(float), 'under': F1.under1000.values.astype(float),
       'lend_eve': ((al.merchant_category.values == 'lending') & (F1.hour.values >= 19)).astype(float)}
for nm, w in [('1h', H), ('6h', 6 * H), ('1d', Dd), ('7d', 7 * Dd)]:
    for s in ['qrm', 'r500', 'tiny', 'newp', 'int', 'col', 'ref', 'under']:
        G[f'rcv_{s}_n_{nm}'] = np.minimum(WR.sum(sig[s], R, t, -w, w, s), 50)
    G[f'rcv_qrm_n_excl_{nm}'] = G[f'rcv_qrm_n_{nm}'] - sig['qrm']
# nearest mismatch at this receiver (seconds), a QR swap stays up for a while
qidx = np.where(sig['qrm'] == 1)[0]
WQ = Win(R[qidx], t[qidx])
tq = t[qidx]
j = WQ.prv(R, t, strict=True); G['rcv_qrm_gap_prev'] = np.where(j >= 0, np.minimum(t - tq[np.maximum(j, 0)], 7 * Dd), 7 * Dd)
j = WQ.nxt(R, t, strict=True); G['rcv_qrm_gap_next'] = np.where(j >= 0, np.minimum(tq[np.maximum(j, 0)] - t, 7 * Dd), 7 * Dd)
G['rcv_qrm_gap_min'] = np.minimum(G['rcv_qrm_gap_prev'], G['rcv_qrm_gap_next'])
G['rcv_qrm_between'] = ((G['rcv_qrm_gap_prev'] < 7 * Dd) & (G['rcv_qrm_gap_next'] < 7 * Dd)).astype(np.int8)
G['rcv_qrm_span'] = np.where(G['rcv_qrm_between'] == 1, G['rcv_qrm_gap_prev'] + G['rcv_qrm_gap_next'], 14 * Dd)
# same idea for refund claims by the sender, tiny credits on the pair, lending payments in the evening
for s, key, Wk, nm in [('ref', S, None, 'snd_ref'), ('lend_eve', S, None, 'snd_lendeve')]:
    qi = np.where(sig[s] == 1)[0]; Wq = Win(key[qi], t[qi]); tqq = t[qi]
    j = Wq.prv(key, t); a = np.where(j >= 0, np.minimum(t - tqq[np.maximum(j, 0)], 30 * Dd), 30 * Dd)
    j = Wq.nxt(key, t); b = np.where(j >= 0, np.minimum(tqq[np.maximum(j, 0)] - t, 30 * Dd), 30 * Dd)
    G[f'{nm}_gap_min'] = np.minimum(a, b); G[f'{nm}_gap_max'] = np.maximum(a, b)
log('receiver counts')

# device: clipped counts and the number of accounts in a short window
for nm, w in [('10m', 10 * M), ('1d', Dd)]:
    G[f'dev_cnt_c_{nm}'] = np.minimum(WD.count(Dv, t, -w, w), 20)
G['dev_own_min_1d'] = win_min(WD, own, Dv, t, -Dd, Dd)
G['dev_login_min_10m'] = win_min(WD, login, Dv, t, -10 * M, 10 * M)
# receiver P2P: incoming burst then outgoing (mule) -- counts relative to quiet periods
G['rcv_in_n_2h'] = np.minimum(WR.count(R, t, -2 * H, 2 * H), 30)
G['rcv_in_quiet_7d'] = WR.count(R, t, -7 * Dd, 7 * Dd) - WR.count(R, t, -2 * H, 2 * H)
G['rcv_out_n_after_2h'] = np.minimum(WS.count(R, t, 1, 2 * H), 30)
G['snd_out_is_after_burst'] = np.minimum(WR.count(S, t, -2 * H, -1), 30)
G['snd_in_newp_share_2h'] = WR.sum(sig['newp'], S, t, -2 * H, -1, 'newp') / np.maximum(WR.count(S, t, -2 * H, -1), 1)
out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in G.items()})
out.to_pickle(os.path.join(CACHE, 'feats2_m.pkl' if MONTHLY else 'feats2.pkl')); log('saved', out.shape)
