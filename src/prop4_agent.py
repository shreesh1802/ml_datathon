"""Event-level block propagation across both P2P (6h, 2-hop) and Merchant / Low-Volume Receiver & Sender bursts (48h).
Usage:
    python src/prop4.py sub_G9 E1,M1 --seg s2
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from common import load_all, CACHE, SUBS, LABELS, macro_f1

ap = argparse.ArgumentParser()
ap.add_argument('sub_name', default='sub_G9')
ap.add_argument('tags', default='E1,M1')
ap.add_argument('--seg', default='s2')
a = ap.parse_args()

al = load_all()
n = len(al)
te = al.is_test.values
mo = al.timestamp.str[:7].values
tr = (~te) & (mo < '2026-10')
K = len(LABELS)
L = {l: i for i, l in enumerate(LABELS)}

y = al.label.map(L).fillna(-1).astype(int).values
tags = a.tags.split(',')
oof = sum(np.load(os.path.join(CACHE, f'oof_{t}.npy')) for t in tags) / len(tags)
pt = sum(np.load(os.path.join(CACHE, f'test_{t}.npy')) for t in tags) / len(tags)
P = oof.copy()
P[te] = pt

# 1. Standard F/2 threshold tuning on OOF
def decide(p, tau):
    r = p[:, 1:] / tau[1:]
    k = r.argmax(1) + 1
    return np.where(r.max(1) > 1, k, 0)

tau = np.full(K, 0.25)
for _ in range(6):
    f = macro_f1(y[tr], decide(oof[tr], tau), per_class=True)
    tau = np.r_[1.0, np.clip(f[1:] / 2.0, 0.10, 0.45)]

cls = pd.read_pickle(os.path.join(CACHE, f'feats_seg_{a.seg}.pkl')).seg_class.values.astype(int)
pred = decide(P, tau)
pred[cls > 0] = cls[cls > 0]
base_pred = pred.copy()

print(f"Baseline G5 OOF Macro-F1: {macro_f1(y[tr], base_pred[tr]):.5f}")

# 2. Lookalike pair map: {true_block_class: [allowed_current_predicted_classes_to_flip]}
LOOKALIKE_PAIRS = {
    L['account_takeover']:   {L['sim_swap']},
    L['sim_swap']:           {L['account_takeover']},
    L['mule_account']:       {L['layering_chain'], L['velocity_bot']},
    L['velocity_bot']:       {L['mule_account'], L['layering_chain']},
    L['layering_chain']:     {L['mule_account'], L['velocity_bot']},
    L['fake_cashback']:      {L['phishing_collect']},
    L['phishing_collect']:   {L['fake_cashback'], L['loan_app_extortion']},
    L['loan_app_extortion']: {L['phishing_collect']},
    L['refund_abuse']:       {L['qr_swap'], L['merchant_collusion']},
    L['qr_swap']:            {L['refund_abuse'], L['merchant_collusion']},
    L['merchant_collusion']: {L['qr_swap'], L['refund_abuse']},
}

t_sec = al.t.values.astype(np.int64)
snd = al.sender_id.values
rcv = al.receiver_id.values
is_p2p = (al.receiver_type.values == 'P2P')
is_mer = (al.receiver_type.values == 'MERCHANT')
amt = al.amount.values
is_round_c5 = np.isin(amt, [1000.0, 2000.0, 2500.0, 5000.0, 10000.0])

# Rule A: 2-hop P2P account propagation within +-6 hours (from prop3.py)
# Build account -> row indices index for P2P endpoints
from collections import defaultdict
acc_rows = defaultdict(list)
for idx in range(n):
    acc_rows[snd[idx]].append(idx)
    if is_p2p[idx]:
        acc_rows[rcv[idx]].append(idx)
for k in acc_rows:
    acc_rows[k] = np.array(acc_rows[k], dtype=np.int32)

seed_type = np.zeros(n, dtype=int)
seed_type[cls > 0] = cls[cls > 0]

for hop in range(2):
    active_seeds = np.where(seed_type > 0)[0]
    new_seeds = {}
    for s_idx in active_seeds:
        b_k = seed_type[s_idx]
        allowed = LOOKALIKE_PAIRS.get(b_k, set())
        if not allowed:
            continue
        t0 = t_sec[s_idx]
        endpoints = [snd[s_idx]] + ([rcv[s_idx]] if is_p2p[s_idx] else [])
        for acc in endpoints:
            nbrs = acc_rows[acc]
            # Filter within +-6 hours (21600s)
            dt = np.abs(t_sec[nbrs] - t0)
            cand = nbrs[(dt <= 21600) & (cls[nbrs] <= 0)]
            for c_idx in cand:
                if pred[c_idx] in allowed and seed_type[c_idx] == 0:
                    pred[c_idx] = b_k
                    new_seeds[c_idx] = b_k
    for c_idx, b_k in new_seeds.items():
        seed_type[c_idx] = b_k
    print(f"After P2P hop {hop+1}: train changed={(pred[tr] != base_pred[tr]).sum()}, test changed={(pred[te] != base_pred[te]).sum()}, OOF Macro-F1={macro_f1(y[tr], pred[tr]):.5f}")

# Rule B: Merchant 48-hour burst propagation from Block 7 (qr_swap), Block 8 (refund_abuse), and Block 11 (merchant_collusion)
# When a block phone (cls in {7, 8, 11}) pays a merchant MR..., any row to that same merchant within +-48h
# currently predicted in LOOKALIKE_PAIRS[b_k] (e.g. merchant_collusion predicted on a refund_abuse or qr_swap burst)
# takes the block phone's class b_k.
mer_rows = defaultdict(list)
for idx in np.where(is_mer)[0]:
    mer_rows[rcv[idx]].append(idx)
for k in mer_rows:
    mer_rows[k] = np.array(mer_rows[k], dtype=np.int32)

blk_mer_seeds = np.where((cls > 0) & is_mer & np.isin(cls, [L['qr_swap'], L['refund_abuse'], L['merchant_collusion']]))[0]
for s_idx in blk_mer_seeds:
    b_k = cls[s_idx]
    allowed = LOOKALIKE_PAIRS[b_k]
    t0 = t_sec[s_idx]
    nbrs = mer_rows[rcv[s_idx]]
    # Only propagate on low/medium-volume burst merchants (<= 60 txns in train, <= 30 in test) within +-48h
    max_cnt = 30 if te[s_idx] else 60
    if len(nbrs) > max_cnt:
        continue
    dt = np.abs(t_sec[nbrs] - t0)
    cand = nbrs[(dt <= 48 * 3600) & (cls[nbrs] <= 0)]
    for c_idx in cand:
        if pred[c_idx] in allowed:
            pred[c_idx] = b_k

print(f"After Merchant 48h propagation: train changed={(pred[tr] != base_pred[tr]).sum()}, test changed={(pred[te] != base_pred[te]).sum()}, OOF Macro-F1={macro_f1(y[tr], pred[tr]):.5f}")

# Rule C: Same-sender 48-hour propagation for multi-day sender campaigns (refund_abuse, merchant_collusion, sim_swap, account_takeover, loan_app_extortion)
blk_snd_seeds = np.where((cls > 0) & np.isin(cls, [L['sim_swap'], L['account_takeover'], L['refund_abuse'], L['loan_app_extortion'], L['merchant_collusion']]))[0]
for s_idx in blk_snd_seeds:
    b_k = cls[s_idx]
    allowed = LOOKALIKE_PAIRS[b_k]
    t0 = t_sec[s_idx]
    nbrs = acc_rows[snd[s_idx]]
    dt = np.abs(t_sec[nbrs] - t0)
    cand = nbrs[(dt <= 48 * 3600) & (cls[nbrs] <= 0) & (snd[nbrs] == snd[s_idx])]
    for c_idx in cand:
        if pred[c_idx] in allowed:
            pred[c_idx] = b_k

print(f"After Sender 48h propagation:   train changed={(pred[tr] != base_pred[tr]).sum()}, test changed={(pred[te] != base_pred[te]).sum()}, OOF Macro-F1={macro_f1(y[tr], pred[tr]):.5f}")

# Audit exact accuracy of all changed rows on train.csv
ch_tr = tr & (pred != base_pred)
print(f"\nTrain rows changed: {ch_tr.sum()}")
print(f"  New prediction matches given label: {(pred[ch_tr] == y[ch_tr]).sum()} ({(pred[ch_tr] == y[ch_tr]).mean()*100:.1f}%)")
print(f"  Old prediction matched given label: {(base_pred[ch_tr] == y[ch_tr]).sum()} ({(base_pred[ch_tr] == y[ch_tr]).mean()*100:.1f}%)")
print(f"  Given label is legit (unreported):  {(y[ch_tr] == 0).sum()}")

ch_te = te & (pred != base_pred)
print(f"\nTest rows changed: {ch_te.sum()}")
if ch_te.sum() > 0:
    print(pd.crosstab(
        pd.Series(np.array(LABELS)[base_pred[ch_te]], name='old_pred'),
        pd.Series(np.array(LABELS)[pred[ch_te]], name='new_pred')
    ).to_string())

os.makedirs(SUBS, exist_ok=True)
out_path = os.path.join(SUBS, f'{a.sub_name}.csv')
pd.DataFrame({'id': al.txn_id[te].values, 'label': np.array(LABELS)[pred[te]]}).to_csv(out_path, index=False)
print(f"Saved {out_path}")
