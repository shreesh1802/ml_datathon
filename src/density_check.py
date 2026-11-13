"""Is October a thinned (50% row sample) version of a normal month, or a normal month at half rate with bursts intact?
Compare burst statistics per day: full train, random half of train, test."""
import numpy as np, pandas as pd
from common import load_all, Win

al = load_all()
rng = np.random.default_rng(0)
half = rng.random(len(al)) < 0.5
day = al.t // 86400
aug = (al.timestamp.str[:7] == '2026-08').values
sep = (al.timestamp.str[:7] == '2026-09').values
octt = al.is_test.values & (al.timestamp.str[:7] == '2026-10').values
sets = {'aug_full': aug, 'aug_half': aug & half, 'sep_full': sep, 'sep_half': sep & half, 'oct_test': octt}
dev = pd.factorize(al.device_id)[0]; snd = pd.factorize(al.sender_id)[0]
allacc = pd.factorize(pd.concat([al.sender_id, al.receiver_id]))[0]; s_code = allacc[:len(al)]; r_code = allacc[len(al):]
p2p = (al.receiver_type == 'P2P').values
rows = []
for name, mask in sets.items():
    idx = np.where(mask)[0]; t = al.t.values[idx]; amt = al.amount.values[idx]; nd = 31 if 'sep' not in name else 30
    wd = Win(dev[idx], t); c_dev = wd.count(dev[idx], t, -300, 300)
    wr = Win(r_code[idx], t); c_rcv = wr.count(r_code[idx], t, -7200, 7200)
    ws = Win(s_code[idx], t)
    j = ws.nxt(r_code[idx], t)  # receiver's next outgoing
    ok = j >= 0; jj = np.where(ok, j, 0)
    gap = np.where(ok, t[jj] - t, 10**9); ratio = np.where(ok, amt[jj] / amt, 0)
    chain = (gap <= 3600) & (ratio > 0.85) & (ratio <= 1.0) & (amt > 500)
    c_snd = ws.count(s_code[idx], t, -3600, 3600)
    ref = (al.txn_type.values[idx] == 'REFUND'); c_ref = ws.sum(ref.astype(float), s_code[idx], t, -7 * 86400, 7 * 86400, 'ref')
    rows.append(dict(set=name, rows_day=len(idx) / nd, dev_ge3=(c_dev >= 3).sum() / nd, dev_ge5=(c_dev >= 5).sum() / nd, dev_ge8=(c_dev >= 8).sum() / nd,
                     p2p_rcv_ge3=((c_rcv >= 3) & p2p[idx]).sum() / nd, p2p_rcv_ge5=((c_rcv >= 5) & p2p[idx]).sum() / nd,
                     chain_links=chain.sum() / nd, snd_ge3_1h=(c_snd >= 3).sum() / nd, snd_ge5_1h=(c_snd >= 5).sum() / nd,
                     ref_ge3=((c_ref >= 3) & ref).sum() / nd, ref_rows=ref.sum() / nd, big_amt=(amt > 50000).sum() / nd,
                     tiny=(amt <= 10).sum() / nd, collect=(al.txn_type.values[idx] == 'COLLECT').sum() / nd,
                     qr_mis=(al.qr_vpa_mismatch.values[idx] == 1).sum() / nd, sim_fresh=(al.sim_age_hours.values[idx] < 72).sum() / nd,
                     dev_new=(al.device_age_days.values[idx] < 2).sum() / nd, fast=(al.login_to_txn_sec.values[idx] < 5).sum() / nd,
                     round500=(amt % 500 == 0).sum() / nd))
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).set_index('set').T.round(1))
