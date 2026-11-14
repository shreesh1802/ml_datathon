# Note for the judges — team Vhagar

**Best public upload:** `sub_G13.csv`, macro-F1 0.5487.
**Code:** https://github.com/shreesh1802/ml_datathon

## To reproduce

1. Put `train.csv`, `test.csv`, `accounts.csv` and `merchants.csv` in a folder `upi-fraud-rings/` next to the README.
2. `pip install -r requirements.txt`
3. `bash run_all.sh`

This writes `submissions/sub_G13.csv`. It needs about 12 GB of memory and 15 GB of disk, uses no GPU, and takes about 30 minutes on a laptop CPU. Seeds are fixed. LightGBM is multi-threaded, so a rerun is not bit-exact: a from-scratch run on 15 Nov rebuilt the file with 24 of 260,582 rows different from the upload (0.01%).

## Approach

Two LightGBM models on about 460 features built from how rows connect (per phone, sender, receiver and pair, over windows from 10 minutes to 30 days), with features that stop working in October dropped and noisy training labels cleaned using out-of-fold predictions. Predictions come from a macro-F1 decision rule that weighs each class by its size.

Four things specific to this data, all computed in code:

1. **Phone-number blocks.** Phones numbered 420000 and above are allocated in ordered blocks by fraud type. We segment the blocks from model predictions, row counts and the gaps in the numbering, and label rows on those phones by their block.
2. **Event-level type correction.** About one fraud event in five carries the label of a lookalike type. A row on a block phone reveals the event's type, so linked rows take it.
3. **Amount digits.** Whole-rupee amounts that are not multiples of 10 mark fraud (99.4% of legit person-to-person whole-rupee payments are multiples of 10).
4. **Precision rules.** Three groups of predictions that are mostly legit in the training months are set to legit.

No test label and no block boundary is typed in by hand. Each step was validated on held-out training months before upload.

## What we found about the data

About 61% of fraud-labelled rows are indistinguishable from legit rows, and about one event in five is labelled as a lookalike type with nothing in the data separating it. We measured October F1 for all eleven fraud classes with probe uploads (0.34 to 0.60). `SCORE_IMPROVEMENTS.md` has the score after each step and what did not work.
