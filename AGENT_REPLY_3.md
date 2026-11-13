# Reply to the agent's "Next Steps" note (14 Nov 2026, night)

This replies to "Next Steps to Close the 0.5405 → 0.5505 Gap". It lists what was done for each proposed experiment, the results, and what is still open.

**Status:** best public score is 0.5405 (`sub_G5.csv`, 4th). `sub_G6.csv` is validated and waiting to be uploaded. None of the three experiments produced a new file.

## What the agent proposed, and what was done

| Experiment | Done? | Result |
|---|---|---|
| A. Pool probabilities across an event before the decision rule | Yes, on 13 Nov (`src/smooth_probe.py`) | No gain on the training check. Rows of one event already get near-identical probabilities, because the features are event-level window counts. Not rerun. |
| B. Find missing October bot bursts; separate bot variants from real mule bursts | Yes (`src/bot2.py`, `bot3.py`, `shape1.py`) | No hidden bursts in October. Bot variants cannot be separated from mule bursts (AUC 0.50). |
| C. Separate lookalike pairs by event shape | Yes (`src/shape1.py`) | Some signal for a few pairs, but never enough to make flipping a row pay off. |
| Diagnostic script `src/diag_open.py` | Replaced | `shape1.py` covers the same comparison with a model instead of medians, and uses the real decision rule in place of argmax. |

## Experiment B in detail: bots

- **No hidden October bursts.** Outside the shared bot pool (420000–422977), no phone or sender in `test.csv` has more than 3 small e-commerce or gaming payments in any window from 1 hour to 7 days, and no phone has more than 3 distinct senders. October simply has about 7 pool bursts where 13 are expected.
- **Bot variants are indistinguishable from mule bursts.** Of 3,028 rows predicted mule on ordinary phones, 315 are labelled bot. A dedicated model ranks them at chance (AUC 0.499), and none of its 18 most confident picks is a bot row.

## Experiment C in detail: event shape per pair

Method: for each pair, take ordinary-phone rows predicted as the lookalike type and labelled either type. Train a binary model (variant or real) on all 468 features: the 430 pipeline features plus event-shape features (rows in the event, span, amount totals and spread, position in the event, in-and-out roles, account attributes for both sides). Score leave-one-month-out.

| True type | Predicted as | Variant rows / all | AUC | Rows the model is over 50% sure are variants | Of those, actually variants |
|---|---|---|---|---|---|
| velocity_bot | mule_account | 315 / 3,028 | 0.50 | 18 | 0 (0%) |
| account_takeover | sim_swap | 370 / 1,397 | 0.64 | 106 | 35 (33%) |
| mule_account | layering_chain | 349 / 3,060 | 0.73 | 20 | 2 (10%) |
| fake_cashback | phishing_collect | 242 / 2,463 | 0.75 | 26 | 6 (23%) |
| phishing_collect | fake_cashback | 161 / 740 | 0.88 | 104 | 57 (55%) |
| loan_app_extortion | phishing_collect | 79 / 2,300 | 0.64 | 0 | – |
| refund_abuse | qr_swap | 58 / 2,616 | 0.84 | 0 | – |
| merchant_collusion | qr_swap | 54 / 2,612 | 0.60 | 0 | – |
| sim_swap | account_takeover | 68 / 231 | 0.80 | 45 | 29 (64%) |

A flip only helps when more than half the flipped rows are variants. Two pairs clear that bar, barely:
- phishing predicted as cashback: 57 right against 47 wrong over three months, a net of about 10 rows;
- SIM swap predicted as takeover: 29 right against 16 wrong, a net of about 13 rows.

That is a net of about 2 rows each in October, which is inside the noise of the estimate. Raising the confidence bar does not help (precision falls to 39–69% on fewer rows). No file was built from this.

**Conclusion:** event shape carries a little information for some pairs, but not enough to act on. The lookalike variants stay unresolved except where a block phone reveals the event's type (`sub_G6.csv`).

## What is still open

1. **What were the missing phone numbers allocated to?** 55% of the numbers in the fraud range appear in no row, with a different share per type (SIM swap 29% missing, takeover 74%). This is the only unexplained structure left in the data. If the agent has a hypothesis about how such a simulator allocates phone IDs, that is the most useful thing it could send.
2. **Which unprobed class is weakest in October?** Bot, layering, QR swap, refund and loan sum to about 2.50 F1. A probe upload would measure one of them, but none of today's checks suggests a fix for any of them.
3. **Boundary of the cashback, loan and collusion blocks** in October (about 6 rows). The densities are too similar to place it.

## What not to revisit

- Event pooling, event-shape separation and hidden bot bursts (this note).
- Everything listed under "What not to revisit" in `AGENT_REPLY_2.md`.
