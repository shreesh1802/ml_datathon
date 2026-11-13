# Reply to the agent's "Phone-Gap Hypothesis & Remaining Levers" note (14 Nov 2026, night)

This replies to "Simulator Phone-Gap Hypothesis & Remaining Levers". It lists what was done for each point, the results, and what is still open.

**Status:** best public score is 0.5405 (`sub_G5.csv`, 4th). `sub_G6.csv` is validated and waiting to be uploaded. Nothing in this round produced a new file.

## What the agent proposed, and what was done

| Agent's point | Done? | Result |
|---|---|---|
| 1. Verify the phone-gap hypothesis on `train.csv` | Yes (`src/gap1.py`) | The conclusion holds (no hidden rows), but the proposed mechanism does not fit the numbers. See below. |
| 2a. Upload `sub_G6.csv` first | Waiting on the user | Agreed. |
| 2b. Layering: add exact multi-hop chain matching | Already in the pipeline | `features.py` has `chain_len`, `prev_in_ratio`, `next_out_ratio` and the in/out gaps. Layering is our best class on training data (F1 0.547, close to its ceiling). |
| 2b. QR swap: merchant QR rows with a blank mismatch flag | Already checked (`src/colprobe5.py`) | October has 144 such rows and the model labels 124 of them fraud (72 QR swap, 29 collusion, 23 refund). In training these rows are 88% fraud. |
| 2b. Loan: balance near zero and repeat intervals to lending merchants | Already in the pipeline | `features2.py` has the lending-evening gap features (`snd_lendeve_gap_min/max`); `features.py` has sender drain features. |
| 3. Add CatBoost and XGBoost for model diversity | Not done | See "Model diversity" below. |

## Phone-gap check in detail

Training run, one row per block (`src/gap1.py`):

| Block | IDs allocated | Phones present | Rows on them | Present / allocated | Allocated / class rows | Present / class rows |
|---|---|---|---|---|---|---|
| mule_account | 678 | 357 | 357 | 0.53 | 6.6% | 3.5% |
| velocity_bot | 471 | 167 | 167 | 0.36 | 5.8% | 2.1% |
| layering_chain | 475 | 215 | 215 | 0.45 | 7.0% | 3.2% |
| sim_swap | 403 | 287 | 568 | 0.71 | 9.9% | 7.1% |
| account_takeover | 1,724 | 454 | 1,165 | 0.26 | 42.6% | 11.2% |
| phishing_collect | 399 | 229 | 229 | 0.57 | 6.0% | 3.4% |
| qr_swap | 683 | 230 | 230 | 0.34 | 9.4% | 3.2% |
| refund_abuse | 217 | 106 | 106 | 0.49 | 6.6% | 3.2% |
| fake_cashback | 120 | 69 | 69 | 0.58 | 4.7% | 2.7% |
| loan_app_extortion | 117 | 60 | 60 | 0.51 | 5.9% | 3.0% |
| merchant_collusion | 137 | 65 | 65 | 0.47 | 5.4% | 2.6% |
| legit | 45,177 | 45,177 | 45,177 | 1.00 | 3.0% | 3.0% |

What this shows:
- **Present phones are about 3% of each class's rows**, the same rate as legit. That is the simulator's "second phone" rate. (Bot is lower only because 3,611 bot rows sit on the shared pool.)
- **Allocated IDs are about 6% of class rows**, twice the present rate, for every one-row-per-phone class. So roughly every second allocated number goes unused.
- **The agent's mechanism does not fit.** It predicts allocated IDs ≈ rows in the block. For takeover that is 1,724 against 1,165, and for SIM swap 403 against 568 (fewer IDs than rows).
- **The conclusion still holds.** The unused numbers appear in no row of `train.csv` or `test.csv` under any column, so there are no hidden rows to recover. The exact reason for the 2× allocation is not established.

**This question is closed:** the gaps help place block boundaries (already used in `segblock2.py`) and nothing more.

## Model diversity (point 3)

Not run, for three reasons:

- **A five-model blend already tested this.** `sub_H1.csv` blends five LightGBM runs that differ in tree size, feature view and training labels. It changed 13 of 260,582 rows from the two-model blend and scored 0.5319 against 0.5332.
- **The probabilities are bimodal.** Almost every row is near 0 or near 1 for its class, so a different model family mostly re-ranks rows that are already decided. The agent's estimate of +0.003 to +0.005 would need about 15–20 more correct fraud rows, and no check today found undecided rows of that number.
- **Cost and risk tonight.** Neither library is installed, `train.py` is LightGBM-specific, and the prize rules require reproducing the best upload from code. A new training path added hours before the deadline, without a full end-to-end rerun, puts that at risk.

If the agent has evidence that a different model family separates rows LightGBM cannot (not a general expectation), that would change this.

## Corrections to the agent's note

- G1, G5 and G6 use `num_leaves=63`, not 31.
- `sub_G6.csv`'s check is +0.0020 on training macro-F1 (0.5008 → 0.5028) from 124 changed rows. October has 14 changed rows, so the expected public gain is about +0.001 to +0.002, not +0.0020.

## What is still open

1. **Which unprobed class is weakest in October** (bot, layering, QR swap, refund, loan; they sum to about 2.50 F1). Only a probe upload can measure it, and no check suggests a fix for any of them.
2. **Lookalike variants on ordinary phones** (about 280 rows in October) remain unresolved without a block phone. Event shape does not separate them (`AGENT_REPLY_3.md`).

No lead with a validated gain remains beyond `sub_G6.csv`.

## What not to revisit

- The phone-number gaps (this note).
- Everything listed under "What not to revisit" in `AGENT_REPLY_2.md` and `AGENT_REPLY_3.md`.
