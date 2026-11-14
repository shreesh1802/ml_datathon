# UPI Fraud Rings — team Vhagar (ML Gradient Wars 2026)

Twelve-class macro-F1 on October 2026 UPI transactions.

**Best public score: 0.5487, from `submissions/sub_G13.csv`.** `run_all.sh` rebuilds that file from the raw data.

## How to run

1. Put the four competition files (`train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`) in a folder named `upi-fraud-rings/` next to this README. They are not in the repository.
2. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline:

   ```bash
   bash run_all.sh
   ```

The result is `submissions/sub_G13.csv` (columns `id`, `label`; 260,582 rows). Intermediate tables go to `cache/`.

| Requirement | Value |
|---|---|
| Python | 3.14 (developed on Windows 11; any OS with bash works) |
| Libraries | numpy, pandas, scikit-learn, lightgbm |
| Memory | about 12 GB available (RAM plus page file) |
| Disk | about 15 GB free for `cache/` |
| GPU | not used |
| Time | about 30 minutes on a laptop CPU |

On Windows, run the command in Git Bash. The steps run one after another; do not run two at once, because each needs most of the memory.

## Reproducibility

Seeds are fixed. LightGBM trains on several threads, so a few dozen borderline rows can differ between runs. A from-scratch run of `run_all.sh` on 15 Nov (28 minutes, fresh folder, raw CSVs only) rebuilt `sub_G13.csv` with 24 of 260,582 rows different from the uploaded file (0.01%).

No test label and no phone-block boundary is typed in by hand. Every rule is computed from the data by the scripts in `src/`.

## Approach

1. **Connection features** (`features.py`, `features2.py`). About 430 features built on train and test together: counts and sums per phone, sender, receiver and pair over windows from 10 minutes to 30 days, onward flows and chain length, reverse payments, session extremes. The problem statement allows building connections from `test.csv`; no label enters any feature.
2. **Drift fixes** (`train.py --driftfix`). `app_version` and the `ip_risk_score` rank are dropped because they behave differently in October. QR payments to a personal account are flagged directly, because October QR swaps arrive with a blank mismatch flag.
3. **Label cleaning** (`clean.py`). Training labels are noisy, so a first model's out-of-fold probabilities are used to relabel rows before retraining: fraud labels the model is over 90% sure are legit become legit, and legit labels it is over 50% sure are fraud take that type.
4. **Phone-number blocks** (`devblock.py`, `segblock.py`, `segblock2.py`). Phones numbered 420000 and above are allocated in ordered blocks, one fraud type after another, then legit second phones. A forward-only Viterbi pass recovers the blocks from model predictions, row counts and the gaps in the numbering (fraud blocks have gaps, each with its own density; legit phones have none). Every row on a fraud-block phone takes the block's type.
5. **Amount features** (`features3.py`). 99.4% of legit person-to-person payments in whole rupees are multiples of 10, so whole-rupee amounts that are not mark fraud. A tree cannot learn divisibility from the raw amount, so these are added explicitly, with exact phishing amounts and window counts.
6. **Two LightGBM models** (`train.py`). One on features where all months form one graph (E2n), one where each month is its own graph (M2). Each is trained leave-one-month-out for validation, then refitted on all three months with three seeds.
7. **Decision rule** (`make_sub2.py --sizeaware`). A row is assigned to the class with the largest positive (p − F/2) divided by class size, which is the first-order gain in macro-F1.
8. **Event-level type correction** (`prop3.py`). About one fraud event in five carries the label of a lookalike type. A row on a block phone reveals the event's true type, so linked rows predicted as the lookalike take that type.
9. **Precision rules** (`fixcell.py`). Three groups of predictions that are mostly legit in the training months are set to legit.

`SCORE_IMPROVEMENTS.md` gives the public score after each step, what did not work, the measured October F1 per class, and why the score is capped: about 61% of fraud-labelled rows are indistinguishable from legit rows.

## Validation

Every step was checked on held-out training months (leave-one-month-out) before it was uploaded. Training macro-F1 of the final recipe is 0.5045 on the given labels; the public score is higher because test labels are clean while about 12% of training fraud is labelled legit.

## Repository layout

| Path | Contents |
|---|---|
| `run_all.sh` | The whole pipeline, in order, with a comment on each step |
| `src/common.py` | Data loading, windowed-count helper, macro-F1 |
| `src/features.py`, `features2.py`, `features3.py` | Feature tables |
| `src/train.py` | LightGBM training and validation |
| `src/clean.py` | Label cleaning |
| `src/devblock.py`, `segblock.py`, `segblock2.py` | Phone-number blocks |
| `src/make_sub2.py` | Blend, decision rule, block override |
| `src/prop3.py` | Event-level type correction |
| `src/fixcell.py` | Precision rules |
| `SCORE_IMPROVEMENTS.md` | What each step added, and what did not help |
| `requirements.txt` | Library list |
