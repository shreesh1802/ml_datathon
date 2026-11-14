# Task for the agent: find what tells lookalike fraud events apart

**One finding would move our score a lot, and this note is only about that finding.** Everything else has been squeezed to gains of about +0.001.

## Where we stand

| | Macro-F1 |
|---|---|
| Our best public score (`sub_G9.csv`) | 0.5453 |
| Estimated ceiling if lookalike events cannot be separated | about 0.56 |
| Estimated ceiling if they are separated perfectly | about 0.61 |
| Leaderboard top five | 0.6075–0.6338 |

Closing the gap between the two ceilings is worth about +0.05. Nothing else we know of is worth more than +0.002.

## The problem in one paragraph

Each fraud event is generated with a behaviour and a label. For about one event in five, the label is a different type from the one the behaviour matches. A complete layering chain carries the label `mule_account`; a fresh-SIM night drain carries `account_takeover`. The whole event gets the swapped label, in training and in October. Our model predicts by behaviour, so every such row is wrong twice: a miss for the true class and a false alarm for the other.

## The swaps we see (training, ordinary phones, three months)

| Label (true type) | Behaves like, and is predicted as | Rows swapped | Rows of that behaviour labelled normally |
|---|---|---|---|
| account_takeover | sim_swap | 361 | 1,500 |
| mule_account | layering_chain | 346 | 2,898 |
| velocity_bot | mule_account | 323 | 3,014 |
| fake_cashback | phishing_collect | 236 | 2,491 |
| phishing_collect | fake_cashback | 151 | 580 |
| loan_app_extortion | phishing_collect | 77 | 2,491 |
| sim_swap | account_takeover | 71 | 172 |
| refund_abuse | qr_swap | 52 | 2,740 |
| merchant_collusion | qr_swap | 57 | 2,740 |

About 1,700 rows in three months, so roughly 280 in October.

## What we know for certain

1. **The swap is per event, not per row.** In a mule-labelled layering chain, every hop has the mule label.
2. **The swap is present in the October test labels.** Rows on fraud-block phones carry the block's type even when they behave like another type, and three separate uploads that relied on this each raised the public score.
3. **A fraud-block phone reveals the true type.** Phones numbered 420000 and above are allocated in ordered blocks by label. About 3% of an event's rows land on one. When one does, the whole event's type is known. We already use this (`src/prop3.py`), and it resolves about 22 October rows.
4. **For full layering chains (three or more hops, over ₹20,000), 19–21% carry the mule label**, at every chain length from 3 to 6 and every amount band.

## What does NOT separate them (do not repeat)

| Tried | Result |
|---|---|
| Binary model per pair on 468 features (all pipeline features, event shape, account attributes), leave-one-month-out | AUC 0.50 for bot-as-mule, 0.64 for takeover-as-SIM, 0.73–0.88 for others, but never above 55–64% precision on confident picks |
| Retraining the main model on original labels with amount-digit features | Same rows mistyped (355 against 361 for takeover-as-SIM) |
| Chain length, first amount, hop retention ratio, whole-rupee amounts (mule against layering) | Identical distributions; retention median 0.980 in both |
| Fresh SIM, own phone, night, away from home (takeover against SIM swap) | Identical: 92% / 91% / 71% / 8% against 89% / 97% / 69% / 7% |
| Another account's phone | Small effect only: 9.1% for takeover-labelled against 2.3% for SIM-labelled, about 33 rows |
| Propagating a block phone's type beyond 12 hours, beyond 2 hops, or through pool phones | Wrong as often as right |
| Balance clamps, exact generator cut-offs, bait-payment split | No gain when retrained |
| `txn_id` bits and residues, file position, chunk membership | Uniform |

## What to look for

The swap is decided somewhere in the simulator. If it is a pure random draw, nothing can be found and the answer is "irreducible". If it depends on anything observable, that is the win. Hypotheses we have **not** tested:

1. **The swap follows an entity, not an event.** A given mule account, merchant, victim or ring might always produce swapped events. Test: for accounts or merchants with two or more separate events, is the label consistent across events far more often than chance? We tested links within 24 hours, not whole-month consistency per entity.
2. **The swap follows the calendar.** All events starting on certain days, weeks or hours might be swapped. Test: swap rate by start date, weekday, hour, and by position within the month.
3. **The swap follows an account or merchant attribute of a specific party** (the mule, the first sender, the victim, the final receiver): `kyc_level`, `bank_code`, `os`, `home_city`, `opened_on`, account-number range or parity. We tested attributes of each row's sender and receiver, not of a role within the event.
4. **The swap follows event size or sequence position.** For example the Nth event of each type, or events above a size. We checked chain length for one pair only.
5. **The swap leaves a trace in noise columns.** `device_age_days`, `sim_age_hours`, `ip_risk_score`, `login_to_txn_sec`, `pin_retries`, `balance_before`, `app` are drawn per row. A swapped event may draw them from its *label's* generator while its structure comes from the *behaviour's* generator. Test: per pair, compare the full distribution of each column (not just the mean) between swapped and normal events, including exact value sets, decimals and ranges.
6. **The amounts are drawn by the label's generator.** For example a mule-labelled chain might use the mule amount distribution for its first hop. Test: first-row amount, last digits and ranges, per pair.
7. **The block phones tell us more than we use.** Within a block, phone numbers follow generation order. Do swapped events sit together in a block (for example at its start or end)? If so, generation order says something about which events are swapped.

A positive result looks like this: a rule or feature that, among rows predicted as type B, separates label A from label B with **precision above 70% on at least 100 training rows per pair**, holding in each of July, August and September separately.

## What you have to work with

`lookalike_rows.csv` (7.3 MB, 29,916 rows): every row our model predicts as fraud plus every row on a fraud-block phone, for training and October.

| Column | Meaning |
|---|---|
| all 21 raw columns | as in `train.csv` |
| `label` | given label (blank for October rows) |
| `split` | `train` or `test` |
| `predicted` | our model's prediction (by behaviour, with block override) |
| `block_class` | the block's type if the row is on a fraud-block phone, else blank |
| `snd_*`, `rcv_*` | account attributes of sender and receiver (home city, bank, KYC, opening date, OS) |
| `mer_*` | merchant category, city, onboarding date |

Swapped rows are those where `split == 'train'`, `block_class` is blank, and `label` and `predicted` are different fraud types. To group rows into events: same receiver within 2 hours for mule and bot bursts; receiver of one row is sender of the next within an hour for chains; same sender within a day for takeover and SIM sessions; same sender-receiver pair for phishing and cashback.

## How to report back

1. Run the check on the file and paste the actual output. Three of your earlier notes contained figures the delivered code did not reproduce, and each one cost a round.
2. For each hypothesis: the table of swap rate by the candidate variable, with row counts.
3. If something separates a pair: the exact rule, its precision and row count per month, and how many October rows it would change.
4. If nothing does after hypotheses 1–7, say so plainly. "Irreducible" is a useful answer.

## Secondary leads (small, only if the above is exhausted)

- Which of bot, layering, QR swap, refund and loan is weakest in October. They sum to about 2.50 F1; we never probed them.
- Merchant collusion: October public F1 was 0.30 on our earlier file. About 77% of collusion-labelled rows are ordinary-looking, so the cap is near 0.39–0.49.
