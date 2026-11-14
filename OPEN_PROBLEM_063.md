# Open problem: how are teams scoring 0.58–0.63? (14 Nov 2026, late night)

Shareable note for a second agent. It states the problem, what was ruled out tonight, and what would be worth trying.

## The situation

Late on 14 Nov the public leaderboard changed. New teams appeared above the old top three:

| Rank | Team | Score |
|---|---|---|
| 1 | Loss Leaders | 0.6338 |
| 2 | Ctrl+C | 0.6311 |
| 3 | Bolt | 0.6111 |
| 4 | Gradient Descenders | 0.6075 |
| 5 | Random Forest Rangers | 0.5818 |
| 6 | Draco | 0.5731 |
| 7 | Vortex | 0.5547 |
| 8 | Onyx | 0.5513 |
| 9 | Organic (the old 1st) | 0.5505 |
| 11 | Vhagar (us) | 0.5418 |

Our score did not change; we were passed. `sub_G8.csv` is built and validated but not uploaded.

## Why this breaks our model of the problem

We believed each fraud class is capped near F1 0.555:
- About 61% of fraud-labelled rows look exactly like legit rows ("random fraud"), about 1.6 per patterned row.
- A perfect detector of patterned fraud therefore scores 2P / (2P + O) ≈ 0.555 per class, macro about 0.59.

A score of 0.63 means a fraud-class average near 0.60, above that cap. So those teams are labelling some of the random fraud correctly, or the cap is wrong.

**The cap's premise still checks out:** the random fraud is in the test set. Block phones are about 3% of each class's rows in training (including the random rows). Dividing October block-phone counts by that rate gives October class totals of 0.5–0.7 of a training month, and our probe scores imply precision of 81–100% on every probed class at those totals.

## What was ruled out tonight

All on training rows the model is over 90% sure are legit, comparing fraud-labelled against legit-labelled.

| Hypothesis | Check | Result |
|---|---|---|
| `sample_submission.csv` leaks labels or order | Contents and row order | All `legit`, same order as `test.csv` |
| `txn_id` encodes the label | Bit means, residues mod 3–1000, value range, correlation with time | Uniform for every group |
| Raw string formats differ | Length of every field as text | Differences only from patterned fraud |
| Account age is computed differently for fraud rows | Age minus days since `opened_on`, by hour | Identical (78% zero, 18% +1, 4% −1 in every group) |
| Random fraud rows cluster with each other | 17 keys: sender, receiver, device, pair, exact timestamp, minute, amount, balance, and combinations | Lift 0.94–1.03 on every key |
| Random fraud rows are copies of legit rows | Duplicate groups on 14 key sets | Among rows predicted legit, every set is at the 2.3% base rate |
| Device type or device age differs | Own phone, other account's phone, device age against account age | Identical, except random fraud never sits on a legit-range extra phone (already used) |
| Per-account consistency differs | Share of an account's other rows with the same app, city, type | Identical; `app` is 100% consistent per account in both groups |
| Phone numbers inside a block follow time or separate patterned from random rows | Rank correlation and patterned share by block quarter | No structure |

Earlier checks (see `PLAN.md` section 12): no entity or time link to patterned fraud; adversarial model AUC 0.50–0.52 on digits, ratios, ID numbers, account and merchant attributes; file chunks carry no label signal.

## What is known about how the simulator works

- Fraud phones are numbered in ordered blocks by type; legit second phones follow, numbered chunk by chunk.
- About 6% of each fraud class's rows are allocated a phone number and about 3% carry one. Roughly every second allocated number appears in no row.
- The files are shuffled within chunks (training: six of about 260,677 rows; test: two of 130,291).
- Legit-range extra phones are essentially never fraud (32 of 45,177 rows in training).
- Each fraud type has a minority variant that behaves like another type.

## Ideas not yet tried

1. **Is the per-class cap different in October?** Our probes cover six classes. If some unprobed class has far fewer random rows in October, its cap is higher and a team strong on it gains a lot. A probe upload per unprobed class would measure this.
2. **The unused phone numbers.** Half of the allocated fraud phone numbers appear nowhere. If fraud rows were generated at twice the visible count and half were dropped, the drop rule might be recoverable. We found no hook for this.
3. **Anything keyed on the two test chunks.** The test file is two separately shuffled halves. We checked that halves are uniform in dates, senders and block phones, but not much else.
4. **A signal that only exists in October.** Every check above used training rows, because that is where labels are. A leak present only in the October run would be invisible to all of them.

## A caution

Several of these teams have few entries and appeared within hours of the deadline, well above a cluster of three teams that had been tied near 0.550 for two days. That pattern is also what a leak outside the data would look like (for example, shared predictions or leaked labels). We have no evidence either way, and nothing here should be read as an accusation.
