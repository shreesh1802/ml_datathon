#!/usr/bin/env bash
# Reproduces submissions/sub_G1.csv (public 0.5332) and sub_H1.csv from the raw data in upi-fraud-rings/.
# Needs about 12 GB of free RAM; run the steps one at a time, never in parallel. Total time is roughly 60 minutes on a laptop CPU.
set -e
cd "$(dirname "$0")"
PY="python -W ignore"

# 1. connection features (all months in one graph)
$PY src/features.py
$PY src/features2.py

# 2. first model on the given labels, then two rounds of label cleaning
$PY src/train.py B2 --driftfix
$PY src/clean.py B2 c1
$PY src/train.py C4 --driftfix --labels c1 --extra feats2.pkl
$PY src/clean.py C4 c2
$PY src/train.py C5 --driftfix --labels c2 --extra feats2.pkl --full 4

# 3. device-number blocks: labels restored on extra phones (y_d1) and the ordered segmentation (feats_seg_s1)
$PY src/devblock.py C5 d1 c2
$PY src/segblock.py C5 s1

# 4. final model on the all-months graph
$PY src/train.py E1 --driftfix --labels d1 --extra feats2.pkl,feats_seg_s1.pkl --full 3

# 5. same model on features where each month is its own graph
MONTHLY=1 $PY src/features.py
MONTHLY=1 $PY src/features2.py
$PY src/train.py M1 --driftfix --labels d1 --base feats_m.pkl --extra feats2_m.pkl,feats_seg_s1.pkl --full 3

# 6. extra blend members: original fraud types (T1, MT1) and a smaller-tree model (V1)
$PY - <<'PYEOF'
import sys, os, numpy as np
sys.path.insert(0, 'src')
from common import load_all, CACHE, LABELS
al = load_all(); y0 = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
y = np.load(os.path.join(CACHE, 'y_d1.npy')).copy(); q = (y0 > 0) & (y > 0) & (y != y0); y[q] = y0[q]
np.save(os.path.join(CACHE, 'y_t1.npy'), y)
PYEOF
$PY src/train.py T1 --driftfix --labels t1 --extra feats2.pkl,feats_seg_s1.pkl --full 3
$PY src/train.py MT1 --driftfix --labels t1 --base feats_m.pkl --extra feats2_m.pkl,feats_seg_s1.pkl --full 3 --seed 1
$PY src/train.py V1 --driftfix --labels d1 --extra feats2.pkl,feats_seg_s1.pkl --full 3 --leaves 31 --lr 0.05 --seed 2

# 7. blend, apply the macro-F1 decision rule and the block override, write the file
$PY src/make_sub2.py sub_G1 E1,M1 --seg s1
$PY src/make_sub2.py sub_H1 E1,M1,V1,T1,MT1 --seg s1

# 8. best upload: same blend, blocks re-segmented with the phone-number gap signal (fraud blocks have gaps,
#    each block its own density; legit phones have none). Only the final block override uses it.
$PY src/segblock2.py C5 s2
$PY src/make_sub2.py sub_G5 E1,M1 --seg s2

# 9. event-level type correction: a row on a block phone reveals its event's true type; linked rows predicted as the
#    lookalike type take the block type (training check: 124 rows changed, 112 match the label, 1 was right before)
MERCH=0 $PY src/prop3.py sub_G5 sub_G6 s2

# 10. same correction with a 12-hour link window (training check: 147 rows changed, 125 match the label, 3 were right before)
MERCH=0 WIN=43200 $PY src/prop3.py sub_G5 sub_G7 s2

# 11. plus the merchant step: a round-amount merchant payment on a block phone reveals the type of the other
#     round-amount payments at that merchant within one day (collusion, QR swap or refund abuse)
WIN=43200 $PY src/prop3.py sub_G5 sub_G8 s2

# 12. amount features (whole-rupee amounts that are not multiples of 10, exact phishing amounts) and role counts;
#     the all-months model leaves out the cross-month role counts (feats3n), the monthly model keeps them per month
$PY src/features3.py
MONTHLY=1 $PY src/features3.py
$PY - <<'PYEOF'
import pandas as pd
d = pd.read_pickle('cache/feats3.pkl')
d[[c for c in d.columns if not c.startswith(('rcv_ring', 'snd_ring', 'rcv_was', 'snd_was'))]].to_pickle('cache/feats3n.pkl')
PYEOF
$PY src/train.py E2n --driftfix --labels d1 --extra feats2.pkl,feats3n.pkl,feats_seg_s1.pkl --full 3
$PY src/train.py M2 --driftfix --labels d1 --base feats_m.pkl --extra feats2_m.pkl,feats3_m.pkl,feats_seg_s1.pkl --full 3
$PY src/make_sub2.py sub_G9raw E2n,M2 --seg s2
TAGS=E2n,M2 WIN=43200 $PY src/prop3.py sub_G9raw sub_G9 s2
