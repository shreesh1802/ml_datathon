# UPI Fraud Rings — team Vhagar (ML Gradient Wars 2026)

12-class macro-F1 on October 2026 UPI transactions. Best public score: **0.5453** (`sub_G9.csv`).
`run_all.sh` reproduces every submission from the raw data, ending with `sub_G9.csv` (step 12). See `SCORE_IMPROVEMENTS.md` for what each step added.

## How to run

1. Put the competition files (`train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`) in a folder `upi-fraud-rings/` next to this README. They are not in this repository.
2. `pip install -r requirements.txt` (developed on Python 3.14, Windows 11, 15 GB RAM).
3. `bash run_all.sh` (writes `submissions/sub_G1.csv` through `submissions/sub_G9.csv`)

Seeds are fixed. LightGBM runs multi-threaded, so a few dozen borderline rows can differ between runs: a from-scratch rerun on 14 Nov reproduced each submission to within 27–68 of 260,582 rows (about 0.02%).

## Approach

1. **Connection features** (`src/features.py`, `src/features2.py`). About 430 columns built on train and test together, with no labels: bursts per phone, receiver fan-in, money moving on within minutes, chain depth, reverse pairs, refund counts, amount shapes, session-level extremes.
2. **October drift handling** (`--driftfix` in `src/train.py`). `app_version` and `ip_risk_score` are dropped: both separate fraud in Jul–Sep and not in October. Long gaps are capped and long-window counts dropped because October has half the volume. A blank QR-mismatch flag is treated as "not flagged".
3. **Label cleaning** (`src/clean.py`). About 63% of fraud-labelled training rows look exactly like legit rows. Out-of-fold predictions are used to train on cleaned labels; thresholds are still tuned against the given labels because the test set contains the same rows.
4. **Device-number blocks** (`src/segblock.py`). Phones that do not belong to an account are numbered in the order the simulator created them: fraud-event phones first, grouped by fraud type, then ordinary second phones. A forward-only dynamic programme segments each run of numbers into blocks. Its emission probabilities are learned on the training run; no boundary or test label is typed in. This recovers ordinary-looking fraud rows on those phones and separates SIM swap from account takeover on new phones. It is an artefact of the synthetic data, not a real-world fraud signal.
5. **Model** (`src/train.py`). LightGBM, 12 classes, leave-one-month-out for out-of-fold probabilities, then a refit on all three months with several seeds.
6. **Decision rule** (`src/make_sub2.py`). A row is given class k when its probability exceeds half of that class's F1 (the macro-F1-optimal threshold), and every row on a fraud-block phone takes the block's class.

## Validation

Train on two of Jul/Aug/Sep, score the third, against the given labels. This check tracked the public leaderboard throughout (see `PLAN.md`, section 11, for every run, what was tried and what was dropped).

## Files

- `src/common.py` — loading, window helper, macro-F1, weight tuning
- `src/features.py`, `src/features2.py` — features (`MONTHLY=1` builds one graph per month)
- `src/train.py` — training and out-of-fold predictions
- `src/clean.py` — label cleaning rules
- `src/devblock.py`, `src/segblock.py` — device-number blocks
- `src/make_sub2.py` — blending, decision rule, submission
- `src/baseline.py` — the organisers' starter model
- `src/*probe*.py`, `src/explore2.py`, `src/drift.py`, `src/cond_drift.py`, `src/history.py`, `src/stage2.py` — analysis scripts and ideas that were tested and dropped
