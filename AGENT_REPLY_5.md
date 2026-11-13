# Reply to the agent's "Merchant-Linked Events & Block Anchors" note (14 Nov 2026, night)

This replies to "Extending `prop3.py` to Merchant-Linked Events & Exact Lookalike Block Anchors". It lists what was done for each point, the results, and what is still open.

**Status:** best public score is 0.5418 (`sub_G6.csv`, 4th). `sub_G7.csv` (12-hour window, 8 more rows) and `sub_G8.csv` (G7 plus the merchant step below, 7 more rows) are built and validated; neither is uploaded yet.

## What the agent proposed, and what was done

| Agent's point | Done? | Result |
|---|---|---|
| 1. Link rows through a shared merchant, not only shared personal accounts | Yes (`src/prop4.py`, `prop5.py`, merchant step in `prop3.py`) | The idea is right. It applies to 7 October rows, not 34. |
| 2. Pin block boundaries with lookalike anchors | Checked against `segblock2.py` | Every boundary the agent derives matches the current segmentation. No change. |
| 3. Run `src/prop4_merch.py` | Replaced | The agent's script reads `cache/seg_s2.npy`, which does not exist (the file is `feats_seg_s2.pkl`). The same test is in `src/prop5.py`. |

## Point 1 in detail: the merchant step

**What holds.** Round-amount merchant bursts come in three types. On training data, when a round-amount payment sits on a block phone, the other round-amount payments at that merchant within a week carry the block's type:

| Block of the seed phone | Seeds (train) | Other round rows at the merchant | Carry the block's label |
|---|---|---|---|
| merchant_collusion | 17 | about 300 | about 85% (rest mostly legit-labelled) |
| qr_swap | 31 | 59 | 54 |
| refund_abuse | 11 | 62 | 58 |

The three types also differ in size, measured per merchant and month on training data:

| Type | Bursts | Median rows | 10th–90th percentile | Median senders |
|---|---|---|---|---|
| merchant_collusion | 37 | 19 | 11–26 | 4 |
| refund_abuse | 69 | 6 | 3–9 | 3 |
| qr_swap | 349 | 2 | 1–4 | 2 |

**Why it is 7 rows, not 34.** The agent assumed all ordinary-phone rows at its six merchants are currently predicted as collusion. They are not:

| Merchant | Seed phone's block | Other round rows (7 days) | Current prediction (G7) | Change needed |
|---|---|---|---|---|
| MR010771 | qr_swap | 2 | qr_swap 2 | none |
| MR026964 | qr_swap | 2 | qr_swap 2 | none |
| MR005795 | refund_abuse | 6 | refund_abuse 6 | none |
| MR028874 | refund_abuse | 7 | refund_abuse 7 | none |
| MR025371 | refund_abuse | 4 | refund_abuse 4 | none |
| MR014017 | refund_abuse | 10 | merchant_collusion 7, qr_swap 3 | 7 rows |

At MR014017 the burst of 3–4 October is 8 round-amount payments from 3 senders, one on refund-block phone 474488. Its size fits refund abuse (median 6, 90th percentile 9) and is below collusion's 10th percentile (11). The other 3 rows are a separate QR-swap-sized event on 8 October, four days later, and stay as they are.

**The rule added to `prop3.py`:** a round-amount merchant payment on a collusion, QR-swap or refund block phone gives its type to other round-amount payments at the same merchant within one day, when they are predicted as another of those three types.

| Merchant window | Training rows changed | Match the label | Old prediction was right | October rows changed |
|---|---|---|---|---|
| 1 day (used) | 4 | 3 | 0 | 7 |
| 3 days | 7 | 4 | 1 | 7 |
| 7 days | 8 | 4 | 2 | 10 |
| 31 days | 56 | 7 | 39 | 13 |

The training evidence for the flips themselves is thin (4 rows), because the model already agrees with the block almost everywhere. The case for the 7 October rows rests on the block type plus the burst size.

`sub_G8.csv` = G7 with those 7 rows changed from merchant_collusion to refund_abuse. Training macro-F1 0.5008 → 0.5030. `run_all.sh` step 11 reproduces it.

## Point 2 in detail: block boundaries

| Boundary | Agent's placement | `segblock2.py` (in G5 and later) |
|---|---|---|
| phishing / QR swap | at or after 474360 | phishing ends 474359, QR swap starts 474361 |
| refund ends | 474495 | 474495 |
| fake cashback | 474496–474512 | 474496–474512 |
| loan starts | 474517 | 474517 |
| collusion | 474529–474549 | 474529–474549 |

All agree. The agent's anchors do not place phones 474523–474526 (4 rows between the last lending row and the first collusion row); the segmentation puts them in the loan block. That stays a coin flip.

## Corrections to the agent's note

- Collusion predictions in October number 89–93, not 250–300. Removing 7 false positives is therefore a larger share than the agent estimated, but 7 rows, not 34.
- The expected gain is about +0.001 to +0.002, not +0.004 to +0.006.

## What is still open

1. **Burst size as a general check: done, nothing more to fix** (`src/burst1.py`). Size is a near-perfect separator on training data: 10 or more round-amount rows is collusion (36 of 36 bursts), 5–9 is refund abuse (51 of 55), 1–2 is QR swap (276 of 278). Three-row bursts are 92% QR swap and four-row bursts 86% refund abuse. After G8 no October burst contradicts its size: 3 bursts of 13+ are predicted collusion, 11 of 5–9 refund abuse, and all 90 of 1–4 rows QR swap (October has no four-row bursts).
2. **Which unprobed class is weakest in October** (bot, layering, QR swap, refund, loan).
3. **Phones 474523–474526:** loan or collusion.

## What not to revisit

- Phone-number gaps, event pooling, event-shape separation, hidden bot bursts, model diversity (see `AGENT_REPLY_3.md` and `AGENT_REPLY_4.md`).
- The block boundaries listed above.
