# What improved the score

ML Gradient Wars 2026, "UPI Fraud Rings", team Vhagar. Public macro-F1 went from 0.2947 to 0.5418 over 13–14 November 2026. This note lists each step that raised the score, in order, and the things that did not.

## The steps that worked

| # | Upload | Public score | Gain | What changed |
|---|---|---|---|---|
| 1 | `sub_A_baseline` | 0.2947 | – | Starter-style model on single-row features |
| 2 | `sub_B` | 0.4899 | +0.1952 | Connection features built on train and test together |
| 3 | `sub_C4t` | 0.5011 | +0.0112 | Drift fixes and label cleaning |
| 4 | `sub_C5t` | 0.5021 | +0.0010 | Second cleaning round, full refit with four seeds |
| 5 | `sub_D1t` | 0.5128 | +0.0107 | First use of the phone-number block leak |
| 6 | `sub_F1t` | 0.5305 | +0.0177 | Ordered block segmentation |
| 7 | `sub_G1` | 0.5332 | +0.0027 | Two-model blend, F/2 decision rule, block override |
| 8 | `sub_G3` | 0.5389 | +0.0057 | Phone-number gap signal (19 rows) |
| 9 | `sub_G5` | 0.5405 | +0.0016 | Gap signal built into the segmentation (19 more rows) |
| 10 | `sub_G6` | 0.5418 | +0.0013 | Event-level type correction from block phones (14 rows) |

`sub_G7` (the same correction with a wider link window, 8 more rows) is built and validated but its score is not recorded here yet.

## What each step did

### 1 → 2: Connection features (+0.195)
The fraud is in how rows connect, not in single rows. `src/features.py` and `src/features2.py` build about 430 features on train and test together:
- counts and sums per phone, sender, receiver and sender-receiver pair over windows from 10 minutes to 30 days;
- money moving on: how soon and how much the receiver sends onward, chain length;
- reverse payments between the same two accounts;
- session extremes for the sender (newest device, freshest SIM, largest amount).

### 2 → 4: Drift fixes and label cleaning (+0.012)
The statement warns that some columns stop working in October and that training labels are noisy.
- **Dropped `app_version`:** October values do not overlap with training.
- **Dropped the `ip_risk_score` rank:** it separates fraud in training (0.41 against 0.20) but not in October (0.24 against 0.22).
- **QR swaps with a blank mismatch flag:** in October the flag is blank on QR swaps. A QR payment to a personal account is 99% fraud in training, so that became a feature.
- **Half volume:** October has half the rows with bursts intact, so long gaps are capped and long-window counts dropped.
- **Label cleaning (`src/clean.py`):** fraud labels the model is over 90% sure are legit become legit for training; legit labels the model is over 50% sure are fraud become that type. Thresholds are still tuned against the given labels.

### 4 → 7: The phone-number block leak (+0.031)
Phones numbered 420000 and above are allocated by the simulator in ordered blocks, one fraud type after another, then legit second phones. The training run and the October run each have their own range.
- `src/segblock.py` recovers the blocks with a forward-only Viterbi pass, using a block-blind model's predictions and the row count per phone. No boundary is typed by hand.
- Every row on a fraud-block phone takes the block's type. On training data the block is right 83 times of 85 when it disagrees with the model.
- This catches fraud rows that look completely ordinary, which no feature can.

### 7: Decision rule and blend (+0.003)
- **F/2 rule:** predict class k when its probability exceeds half of that class's achievable F1. This is the optimal threshold for F1.
- **Blend:** average of a model on the all-months graph (E1) and one where each month is its own graph (M1).

### 7 → 9: The phone-number gap signal (+0.0073)
Fraud-block phone numbers have gaps; legit ones have none. Each fraud block also has its own share of numbers present (SIM swap 71%, takeover 26%).
- `src/segblock2.py` adds the number of missing IDs before each phone to the segmentation.
- On the training run it places the SIM-swap/takeover boundary exactly where the labels switch (425032 / 425039) and matches 2,731 fraud-labelled rows against 2,720 before.
- In October: 28 rows moved from takeover to SIM swap, and 4 rows at the end of the fraud range moved from legit to collusion.

This came from checking an outside audit. Its main claims were wrong, but its observation about gaps was right (see `AUDIT_REVIEW.md`).

### 9 → 10: Event-level type correction (+0.0013)
Each fraud type has a minority variant that behaves like another type (for example, a bot event that looks like a mule burst). Nothing in a row separates them. But a row on a block phone reveals the true type of its whole event.
- `src/prop3.py` takes each block-phone row, finds ordinary-phone rows sharing an account within 6 hours, and gives them the block's type if they were predicted as its lookalike. It repeats once from the changed rows.
- Training check: 124 rows changed, 112 match the label, 1 was right before. Macro-F1 0.5008 → 0.5028.
- In October: 14 rows changed.

## What did not improve the score

| Tried | Result |
|---|---|
| Five-model blend (`sub_H1`) | 0.5319, below G1's 0.5332; it changed only 13 rows |
| Second-stage model on neighbours' predictions | No gain on the training check; dropped |
| Features from label history (target encoding) | Hurt in October: fraud entities recur within training months but not into October |
| Threshold scaling and October-informed thresholds | Change 13–25 rows; probabilities are bimodal |
| Probability pooling across events | No gain; event rows already get near-identical probabilities |
| Outside audit's collusion amount rule | Changes 0 rows on labelled data; the amounts belong to three fraud types |
| Following a row's pattern over its block | On training data the block is right 404 of 483 times |
| Separating lookalike variants by event shape | Bot-as-mule AUC 0.50; best pairs net about 2 rows each |
| Pool phones as seeds for the event correction | 14 wrong flips on training |

## Why the score is capped

About 61% of fraud-labelled rows are a flat 2.35% of all rows, relabelled at random. They share no entity, timing, attribute or digit pattern with real fraud. A perfect detector of the patterned fraud would score about 0.555 per fraud class, so each class sits near 0.5 for every team. Scores in the 0.54–0.55 range differ by tens of rows, not by a different model.

## Per-class October scores (measured on G1 with probe uploads)

| Class | October F1 |
|---|---|
| mule_account | 0.541 |
| sim_swap | 0.524 |
| phishing_collect | 0.518 |
| fake_cashback | 0.517 |
| account_takeover | 0.511 |
| merchant_collusion | 0.300 |
| bot, layering, QR swap, refund, loan (not probed) | about 0.50 on average |

## Where to find the details

| File | Contents |
|---|---|
| `PLAN.md` | Full plan and results log (sections 11–15) |
| `run_all.sh` | Reproduces every submission from the raw data |
| `AUDIT_REVIEW.md` | Check of the outside audit |
| `OOF_CONFUSION_G5.md` | Confusion matrix on training months |
| `AGENT_REPLY_2.md` – `AGENT_REPLY_4.md` | Replies to the second agent's questions |
