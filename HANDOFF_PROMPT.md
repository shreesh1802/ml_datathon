You are taking over a fraud-classification competition entry that is stuck. We are 5th at 0.5332 public macro-F1; the top three are at 0.5497–0.5505. I need you to find what we are missing. Several earlier conclusions may be wrong, so treat them as claims to verify, not facts.

## The competition

ML Gradient Wars 2026, "UPI Fraud Rings". Predict one of 12 labels for each of 260,582 transactions in `test.csv` (October 2026), training on July–September (about 1.56M rows). Metric is macro-F1 over 12 labels: `legit` plus 11 fraud types (mule_account, velocity_bot, layering_chain, sim_swap, account_takeover, phishing_collect, qr_swap, refund_abuse, fake_cashback, loan_app_extortion, merchant_collusion).

Read `upi-fraud-rings/PROBLEM_STATEMENT.md` first. The lines that matter most:
- "Training labels are not perfect... some fraud was never reported (labelled legit) and some was filed under the wrong type. The test labels are clean."
- "Some fraud looks completely ordinary, and some fraud types look like each other."
- "Not every column is as useful as it looks in training."
- "October is festival season... Fraud rings adapt too."
- Rows are shuffled; using `test.csv` to build connections is allowed.

Rules: no hand-labelling test rows (rules written in code are fine); train only on the provided data. The public leaderboard is 30% of test; the final rank scores each team's best public upload on the other 70%. Deadline is 15 Nov 2026, 10:00 IST. We have very few uploads left, so everything must be validated offline first.

## Where things are

Project root: `C:\Users\0095\Downloads\ml gradient wars\` (repo: https://github.com/shreesh1802/ml_datathon).
- `upi-fraud-rings/`: train.csv, test.csv, accounts.csv, merchants.csv, problem statement.
- `src/`: pipeline. `common.py` (loading, `Win` windowed-count helper, `macro_f1`), `features.py` and `features2.py` (about 430 connection features built on train and test together), `clean.py` (label cleaning), `train.py` (LightGBM multiclass, leave-one-month-out plus full refit), `devblock.py` and `segblock.py` (the phone-number leak, below), `make_sub2.py` (decision rule and block override).
- `cache/`: feature tables, out-of-fold predictions (`oof_TAG.npy`) and test predictions (`test_TAG.npy`), all row-aligned with `load_all()`.
- `submissions/sub_G1.csv`: our best upload (0.5332).
- `PLAN.md`: full history. Section 11 is the results log; section 12 lists bottlenecks. `run_all.sh` reproduces G1.
- The machine has 15 GB RAM. Never run two feature-loading Python jobs at once. Do not `import features2` from a script: it reruns and rewrites the cache.

## What the current solution does

1. LightGBM multiclass on connection features (windowed counts and sums per device, sender, receiver and pair; chain depth; onward flows; session extremes), with legit rows subsampled.
2. Drift fixes: drops `app_version` (no overlap with October) and `ip_risk_score` rank (separates fraud in training, not in October); caps long gaps; treats P2P QR payments as fraud because October QR swaps arrive with a blank `qr_vpa_mismatch`.
3. Label cleaning from out-of-fold probabilities, then retraining.
4. A leak: phones numbered 420000 and above are allocated by the simulator in ordered blocks by fraud type, separately for the training run (420000–473631) and the October run (from 473632). `segblock.py` segments each run with a monotone Viterbi pass, and every row on a fraud-block phone takes the block's class. This took us from 0.5021 to 0.5305.
5. Decision rule: predict class k when p_k exceeds half of that class's achievable F1. G1 blends two models (E1 and M1) and applies the block override.

Upload history: baseline 0.2947, B 0.4899, C4t 0.5011, C5t 0.5021, D1t 0.5128, F1t 0.5305, G1 0.5332, H1 (5-model blend) 0.5319.

## What we believe, and how strongly

**Measured directly (public probes: G1 with one class switched to legit).** October F1 per class: mule 0.541, sim_swap 0.524, phishing 0.518, fake_cashback 0.517, account_takeover 0.511, merchant_collusion 0.300. The five unprobed classes (bot, layering, QR, refund, loan) sum to about 2.50.

**Our central claim, which you should attack first.** About 61.5% of fraud-labelled training rows look exactly like legit rows. We concluded they are random and unpredictable, which caps each fraud class near F1 0.555. Evidence:
- They are a flat 2.35% of all rows every day, about 1.6 per patterned row in every class.
- A model separating them from legit rows scores AUC 0.504–0.52 using the full feature set, raw digits, field ratios, ID numbers, and account and merchant attributes.
- They share senders, receivers and phones with patterned fraud of the same type no more often than legit rows do, and sit no closer to it in time.
- The test set appears to contain them too: test fraud share is about 4%, and our per-class scores are consistent with precision near 0.85 and half the training label counts.

If this claim is right, three teams being 0.017 above us means they catch more patterned fraud somewhere. If it is wrong, that is the whole gap. The problem statement says training noise is "unreported fraud" and "wrong type", which does not obviously describe random legit rows relabelled as fraud. That mismatch bothers me.

**Other conclusions to recheck:**
- Velocity bots: all patterned bots are 26–58 payment bursts from a shared phone pool (420000–422977). October has 269 pool rows, about 7 bursts against 13 expected. We found no slower or spread-out variant.
- Merchant collusion: the model catches only round-amount bursts to small merchants (about 88% precise, about 18% recall). Probabilities are bimodal, so thresholds do nothing. October repeat sender-to-merchant structure matches training.
- Features built from label history (target encoding of entities) hurt in October: fraud entities recur within July–September but not into October.
- A second-stage model on neighbours' predicted probabilities did not help.
- `txn_id` is random (not a hash of a row index). Account, merchant and other ID orderings showed no leak beyond the phone blocks.
- The train/test split follows each fraud event's start date: 147 training rows are dated October and 46 test rows are dated November.

## What I want from you

Find the flaw or the missing signal that explains a consistent 0.017 gap. In rough priority:

1. Try to break the "ordinary-looking fraud is random" claim. Look for structure we never tested: higher-order links (two hops, shared receivers of receivers, bank or city rings), event-level grouping, anything that only exists in how rows connect.
2. Look for a second simulator leak of the same kind as the phone blocks (any ID, counter or allocation order, in any of the four CSVs, including how accounts and merchants are numbered relative to fraud roles).
3. Check the October block segmentation in `segblock.py` for boundary errors, and whether block membership can be extended from phones to the accounts or receivers tied to them.
4. Re-examine what "fraud rings adapt" changed in October for each fraud type, especially bot, collusion and the five unprobed classes.
5. Review the pipeline for plain mistakes: leakage in validation, the label-cleaning rules, the decision rule, anything that makes the offline check (about 0.50) misleading.

Validate every idea offline before proposing an upload: leave-one-month-out on training, and a check that the feature behaves the same in October (compare training-month rates, halved, against test).

Report back with: what you verified, what you overturned, and for each proposed fix the evidence, the expected gain, and the exact commands that produce a submission file. Mark clearly what you confirmed by running code versus what you inferred. If you conclude the gap cannot be closed with this data, say so and show why.
