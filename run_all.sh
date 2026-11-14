#!/usr/bin/env bash
# Reproduces submissions/sub_G13.csv (public macro-F1 0.5487) from the raw data in upi-fraud-rings/.
# Needs about 12 GB of memory available (RAM plus page file) and 15 GB of free disk. Steps run one at a time.
# Total time is about 30 minutes on a laptop CPU.
set -e
cd "$(dirname "$0")"
PY="python -W ignore"

# 1. connection features on train and test together (all months in one graph)
$PY src/features.py
$PY src/features2.py

# 2. first model on the given labels, then two rounds of label cleaning
$PY src/train.py B2 --driftfix
$PY src/clean.py B2 c1
$PY src/train.py C4 --driftfix --labels c1 --extra feats2.pkl
$PY src/clean.py C4 c2
$PY src/train.py C5 --driftfix --labels c2 --extra feats2.pkl --full 4

# 3. phone-number blocks: phones numbered 420000 and above are allocated in ordered blocks by fraud type.
#    devblock restores labels on those phones (y_d1); segblock segments the blocks from model predictions (s1)
$PY src/devblock.py C5 d1 c2
$PY src/segblock.py C5 s1

# 4. the same features with each month as its own graph, for the second model
MONTHLY=1 $PY src/features.py
MONTHLY=1 $PY src/features2.py

# 5. amount features: whole-rupee amounts that are not multiples of 10, exact phishing amounts, window counts.
#    The all-months model leaves out the cross-month role counts (feats3n); the monthly model keeps them per month
$PY src/features3.py
MONTHLY=1 $PY src/features3.py
$PY - <<'PYEOF'
import pandas as pd
d = pd.read_pickle('cache/feats3.pkl')
d[[c for c in d.columns if not c.startswith(('rcv_ring', 'snd_ring', 'rcv_was', 'snd_was'))]].to_pickle('cache/feats3n.pkl')
PYEOF

# 6. the two final models: all-months graph (E2n) and monthly graph (M2), leave-one-month-out plus a 3-seed full refit
$PY src/train.py E2n --driftfix --labels d1 --extra feats2.pkl,feats3n.pkl,feats_seg_s1.pkl --full 3
$PY src/train.py M2 --driftfix --labels d1 --base feats_m.pkl --extra feats2_m.pkl,feats3_m.pkl,feats_seg_s1.pkl --full 3

# 7. block segmentation again, now also using the gaps in phone numbering (fraud blocks have gaps, each block its
#    own density; legit phones have none). Used for the final block override (s2)
$PY src/segblock2.py C5 s2

# 8. blend the two models, apply the class-size-aware macro-F1 decision rule and the block override
$PY src/make_sub2.py sub_G12raw E2n,M2 --seg s2 --sizeaware

# 9. event-level type correction: a row on a block phone reveals its event's true type. Rows linked to it by a shared
#    account within 12 hours (two hops), or by a shared merchant within a day for round-amount bursts, take that type
TAGS=E2n,M2 WIN=43200 $PY src/prop3.py sub_G12raw sub_G12ev s2

# 10. three precision rules, each checked on the training months:
#     - phishing predicted on a P2P collect request for a whole multiple of 10 (8% phishing, 89% legit) -> legit
#     - any fraud predicted on a legit-range extra phone (11 of 11 legit) -> legit
#     - QR swap predicted on a merchant payment with a decimal amount (18% QR swap, 80% legit) -> legit
$PY src/fixcell.py sub_G12ev sub_G13 legitrange qrdec

echo "done: submissions/sub_G13.csv"
