# UPI Fraud Rings: solution plan and fallbacks

ML Gradient Wars 2026. Round closes 15 Nov 2026, 10:00 AM IST. Written 13 Nov 2026.

**Status:** nothing is built yet. No features, model or submission exist. The advantages claimed below are arguments from the data profile and the papers, not measured scores.

## 1. The task in one paragraph

Label every October transaction in `test.csv` (about 260k rows) with one of 12 classes: `legit` plus 11 fraud types. Train on Jul–Sep (`train.csv`, about 1.56M labelled rows). The metric is macro-F1, so every class counts equally. All-`legit` scores about 0.08 and the organisers call 0.40 excellent. There are 10 uploads per day, and the final ranking takes each team's best public upload and scores it on the private 70%.

## 2. What the data profile showed

| Finding | Evidence | Consequence |
|---|---|---|
| `app_version` has no overlap | Train has 3.0–4.8 only, October has 5.0–5.2 only | Drop the column |
| October amounts are higher | Median ₹542 vs about ₹464 in Jul–Sep | Use within-month ranks, not raw rupees |
| `ip_risk_score` drifts | Median 0.18 → 0.21, while the fraud-vs-legit gap in train is about 0.08 | Rank within month, or drop if validation says so |
| Fraud labels look very noisy | Only 950 of 3,305 `refund_abuse` rows are refund claims; `qr_swap` rows carry the QR mismatch flag 30% of the time | Clean or downweight labels (not yet verified as noise) |
| History carries over | 91–95% of October rows involve a sender, receiver or phone seen in training | Entity history features will work on test |
| Device farms exist | A few phones are used by 20–28 accounts; nearly all others by 1–2 | Accounts-per-device is a strong feature |
| Class mix is stable Jul–Sep | Each fraud class has a similar count every month | October mix is unknown; estimate it |
| Test includes early November | 46 test rows are dated November, mostly refund claims | Month-based features must not assume October only |

## 3. Plan C (recommended): robust two-stage pipeline

1. **Validation.** Train on Jul–Aug, score September. Every feature uses past rows only.
2. **Drift fixes.** Drop `app_version`. Convert `amount`, `balance_before` and `ip_risk_score` to within-month ranks. Train a September-vs-October classifier and inspect any feature it relies on.
3. **Connection features**, built on train and test together:
   - Receiver: distinct senders in the last 1 h / 6 h / 24 h, total received, share of new payees.
   - Onward flow: did the receiver send a similar amount onward within minutes; hop position in a chain.
   - Device: accounts per device, payments per device in the last 1 / 5 / 60 minutes, device not the sender's usual one.
   - Pairs: has the receiver ever paid the sender; tiny credit followed by a large payment back.
   - Sender: refund claims this week and month, repeat payments to the same lending merchant, evening share.
   - Amount: round number, just under a limit (₹1,999, ₹4,999, …), share of balance.
   - Takeover signals: new phone, fresh SIM, device city differs from home city, night hour, PIN retries.
   - Merchant: round-number sales share, how few accounts make up its sales.
4. **Stage 1 model.** LightGBM, 12 classes, class weights, out-of-fold predictions for every training row.
5. **Label cleaning.** Downweight fraud-labelled rows the model confidently calls legit. Leave legit rows that look strongly like fraud out of the legit class. Retrain.
6. **Stage 2 model.** A second LightGBM that also sees the stage-1 predictions of neighbouring rows (same receiver, same device, same chain, close in time). Fraud comes in bursts that share a label, so neighbours should agree.
7. **October class mix.** Re-estimate class proportions on the test set with the EM method and adjust the predicted probabilities.
8. **Thresholds.** Tune per-class decision weights for macro-F1 on September, then apply to test.
9. **Final.** Retrain on all of Jul–Sep, average 3–5 seeds, write `submission.csv`.

**Keep rule:** each of steps 2, 5, 6 and 7 stays only if it improves the September score.

## 4. Plan B (first fallback): graph features plus LightGBM

This is the original plan. It is Plan C without steps 2 (beyond dropping `app_version`), 5, 6 and 7.

1. Out-of-time validation: Jul–Aug → Sep.
2. The same connection features as Plan C step 3.
3. One LightGBM with class weights.
4. Per-class thresholds for macro-F1.
5. Retrain on everything and submit.

**Fall back to this when:** label cleaning or the second stage lowers the September score, or time runs short. Plan B is also the first working milestone of Plan C, so it will exist either way.

## 5. Plan A (last resort): starter baseline

The organisers' `starter_notebook.ipynb`: single-row features and scikit-learn's HistGradientBoosting. It scores low on purpose because it cannot see connections between rows.

**Use this when:** the feature pipeline is broken close to the deadline and a valid scored upload is needed for the certificate. Upload it once early so a score is on the board.

## 6. Option not planned: graph neural network

GraphSAGE or a temporal GNN on the transaction graph. Not planned because:

- Published evidence is mixed on whether GNNs beat boosting with graph features.
- It needs a heavier install (PyTorch Geometric on Python 3.14 is a risk) and more tuning than 48 hours on a 15 GB laptop allows.
- Noisy labels hurt it at least as much as they hurt boosting.

**Revisit only if:** Plan C is finished with time to spare. Then add node embeddings as extra columns to the LightGBM, not as a replacement.

## 7. Why Plan C should beat the others

| Compared with | What Plan C adds | Why it should matter here |
|---|---|---|
| Plan A | Connection features | The statement says fraud is only visible across rows |
| Plan B | Drift fixes, label cleaning, second stage, class-mix correction | Plan B learns from noisy labels and raw amounts that shift in October |
| Graph features + boosting (paper) | Same four additions | That paper's benchmarks, as far as I read, have no noisy labels or shifted test month |
| GNNs | Cheaper, faster to iterate | Ten uploads a day and 48 hours reward fast iteration |

## 8. Risks and open questions

- **Validation is biased.** September labels are noisy while test labels are clean, so the September score understates the real score and can mislead on small differences.
- **The label-noise reading is unverified.** The odd rows may be genuine burst members that look ordinary, not noise. Check before downweighting.
- **Public leaderboard overfitting.** The best public upload is the one that counts, so upload only changes that also improve September.
- **Unverified research.** Searches for positive-unlabelled learning and the Kaggle IEEE-CIS winners' write-up returned nothing usable; neither is part of this plan.
- **No guarantee.** This plan targets the traps in this dataset. It does not promise a win.

## 9. Order of work

1. Upload Plan A once for a baseline score.
2. Build features and Plan B. Record the September score.
3. Add Plan C steps one at a time, keeping each only if September improves.
4. Final retrain, seed average, upload.
5. Push code to a repository and submit the link before the deadline.

## 10. Sources

- [Graph Feature Preprocessor (IBM, Caltech, TU Delft)](https://arxiv.org/pdf/2402.08593)
- [AMLworld: Realistic Synthetic Financial Transactions (Altman et al., NeurIPS 2023)](https://arxiv.org/pdf/2306.16424)
- [GraphSAGE on transaction networks for mule detection](https://arxiv.org/html/2509.12255v1)
- [Amatriciana: temporal GNN for laundering detection](https://arxiv.org/html/2506.00654v1)
- [Confident Learning (Northcutt et al., JAIR 2021)](https://www.jair.org/index.php/jair/article/view/12125)
- [Adjusting classifier outputs to new priors (Saerens et al., 2002)](https://mlanthology.org/neco/2002/saerens2002neco-adjusting)
- [Credit card fraud and concept drift with delayed labels (Dal Pozzolo et al., 2015)](https://boracchi.faculty.polimi.it/docs/2015_04_Credit_Card_Fraud_Detection_DalPozzolo_Boracchi_Caelen_Alippi_Bontempi.pdf)
- [Why tree-based models still outperform deep learning on tabular data (Grinsztajn et al., 2022)](https://arxiv.org/abs/2207.08815)
- [NVIDIA blog on the winning IEEE-CIS fraud solution](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)

Paper results are quoted from abstracts and search snippets. Check numbers against the originals before citing them.

## 11. Results log (13 Nov)

"Check" is macro-F1 on held-out months (train on two of Jul/Aug/Sep, score the third) against the given labels.

| Run | What changed | Check | Public score |
|---|---|---|---|
| A | Starter single-row model | 0.285 | 0.2947 |
| B | 341 connection features, LightGBM, tuned class weights | 0.467 | 0.4899 |
| C1 | Trained on cleaned labels | 0.473 | not uploaded |
| C2 | Dropped `ip_risk_score`, added session features | 0.473 | not uploaded |
| C3 | Second-stage model on neighbours' predictions | 0.470 | dropped |
| C4 | Blank QR flag treated as "not flagged", F/2 decision rule | 0.472 | 0.5011 |
| C5 | Second cleaning round, refit on all months with 4 seeds | 0.474 | 0.5021 |
| D1 | Device-number position shares (learned coordinates) | 0.500 | not confirmed |
| D2 | D1 iterated on its own predictions | 0.501 | dropped: October blocks got messier |
| E1 | Ordered segmentation of device numbers (`segblock.py`) | 0.498 | pending |

Device-number blocks (found 13 Nov evening, after the leaderboard showed three teams near 0.545):

- **Extra phones are numbered by fraud type.** Device numbers above the account range (420000+) were handed out run by run: fraud-event phones first, grouped by type in a fixed order (shared bot phones, mule, bot, layering, SIM swap, takeover, phishing, QR, refund, cashback, loan, collusion), then ordinary second phones. October has its own run starting right after the last training number.
- **It recovers some ordinary-looking fraud.** About 3% of the otherwise unpredictable fraud rows sit on these phones. Their block gives both "fraud" and the type.
- **It separates SIM swap from takeover on new phones**, which look identical row by row.
- **`segblock.py` finds the blocks without typed-in boundaries.** It segments the training run from the given labels, measures how a block-blind model labels each block, then segments both runs from those block-blind predictions alone. On training it agrees with the label-based blocks for 98.8% of fraud-block phones.
- **October blocks found:** mule 80 phones, bot 29, layering 39, SIM swap 37, takeover 88, phishing 43, QR 41, refund 22, cashback 11, loan 4, collusion 12, then 7,466 ordinary phones.
- **This is a simulator artefact.** It says nothing about real fraud; it is allowed as a coded feature.
- **Label history is a trap (run F1, dropped).** Fraud accounts recur across Jul–Sep (19.5% of September bot rows, 33% of phishing rows come from accounts with the same history), but almost none carry into October: 180 repeat-phishing rows expected, 9 found. October was generated with fresh fraud entities. History features gave +0.002 on the check and were dropped.
- **No adapted bots among known bot accounts.** Legit-predicted October rows from past bot accounts: 1,017 against 1,006 expected.
- **Linking from block phones adds nothing.** Rows sharing a receiver or sender with a fraud-block phone are fraud only when they already show a pattern.
- **Public score after the device-block work: 0.5128** (4th; top three 0.544–0.549).
- **E1 public score: 0.5305** (4th; third place 0.5443).
- **Block override (E2):** when the block and the model disagree on training rows, the block is right 83 times and the model 2. Applied to every row on a fraud-block phone; changes 20 October rows.
- **One graph per month (M1):** check 0.498 against 0.500 for E1; October phishing confidence partly recovers. Blended with E1 as `sub_G1.csv` (differs from E1 on 47 rows).
- **Checked and ruled out after 0.5305:** October columns on known block rows (ordinary-looking fraud matches legit on every column, AUC about 0.5); a phone-level SIM-swap/takeover classifier (AUC 0.63, boundary stays within 2 phones); separating look-alike types on own phones (mule vs bot-labelled 0.5, QR vs refund 0.55, others 0.6–0.78 with small minority shares); November tail rows (already all flagged).
- **Type confusion costs about 0.06 on the check** (0.500 against 0.563 if every detected fraud row had the right type). Most of it cannot be separated with the columns available.
- **Keeping the original fraud types (run T1) did not help:** check 0.498 against 0.500. Cleaning had retyped 1,358 patterned rows (velocity_bot→mule 295, mule→layering 220, cashback→phishing 157), but the model cannot learn those variants from the given types either.
- **Bots are genuinely rarer in October.** Shared bot phones come from one pool numbered for all four months; each is used for one burst. Bursts per month: 33, 26, 26, then 7 in October, on phones never seen in training.
- **G1 (E1+M1 blend, block override) public: 0.5332** (5th; 4th is 0.5400, top three 0.548–0.549).
- **No October excess at low probabilities.** Compared month by month, October counts sit inside the Jul–Sep range for every class. Only phishing shifts, from very to moderately confident, and those rows are already labelled phishing.
- **H1 = five-model blend** (E1, M1, V1 smaller trees, T1 and MT1 on original types): check 0.5011 against 0.5005 for G1; differs from G1 on 13 rows.
- **No other ID leaks.** Transaction IDs, row order, account and merchant numbers and value digits give AUC about 0.5 for the ordinary-looking fraud rows.

Later checks (13 Nov evening):

- **The ordinary-looking fraud rows are unpredictable.** A model trained to separate them from legit rows scores AUC 0.504 on September, and a model guessing their fraud type does worse than always picking the largest class. No feature work will recover them.
- **No further October changes found.** Amount shapes, chain cuts, bait sizes, refund counts and takeover signals match September once halved. Merchant collusion has 3 large clusters against about 6 expected, which is within chance.

What was learned, and how it changed the plan:

- **About 63% of fraud-labelled training rows are ordinary rows.** They share senders, receivers and phones with fraud no more often than legit rows do. The public score of B shows the test set contains them too, so they cap every class near F1 0.5. The ceiling on the check is about 0.53.
- **Label cleaning helps training, not scoring.** Training on cleaned labels lets the model train longer and sharper. Thresholds are still tuned against the given labels, because the test resembles them.
- **`ip_risk_score` is a trap.** Patterned fraud averages 0.41 against 0.20 for legit in Jul–Sep, but predicted fraud averages 0.24 against 0.22 in October. Dropped at no cost on the check.
- **October QR swaps have a blank mismatch flag.** QR payments to a personal account are 99% fraud in training. In October about 130 such rows had a blank flag and were called legit. Fixed in C4: predicted QR swaps went from about 380 to about 540, in line with the expected count.
- **October has half the volume with bursts intact.** Long gaps are capped at two hours and long-window counts are dropped.
- **Second-stage model did not help** and was dropped under the keep rule.
- **Velocity bots are about half as frequent in October** (7 phone bursts against about 13 expected). No adapted variant was found in a cell-by-cell comparison with September.
- **`sim_swap` and `account_takeover` swap labels at session level** in about a quarter of training sessions. Since test labels are clean, the check understates these two classes.
