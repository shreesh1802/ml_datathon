# Reply to the agent's follow-up diagnostic (14 Nov 2026, night)

This replies to "Follow-Up Findings & Open-Question Diagnostic". It lists what was done for each thing the agent asked, what was found, and what is still open.

**Status:** best public score is 0.5405 (`sub_G5.csv`, 4th place). A new file, `sub_G6.csv`, is built and validated but not yet uploaded. The top three are at 0.5497–0.5505.

## What the agent asked, and what was done

| Agent's request | Done? | Result |
|---|---|---|
| Section 4, part 1: per-class out-of-fold precision, recall and F1 | Yes, already in `OOF_CONFUSION_G5.md` | Uses the actual G5 recipe (E1+M1 blend, F/2 rule, block override). The agent's script used a plain argmax on one run. |
| Section 4, part 2: takeover ↔ SIM swap mix-ups on ordinary phones, by signature | Yes (`src/pairs1.py`) | See question B below. |
| Section 4, part 3: mule ↔ layering mix-ups, fan-in hub or 1-to-1 chain | Partly | Not run as a fan-in count. Answered a different way: a model trained on the given types cannot separate them either (see question B). |
| Section 4, part 4: do missed collusion rows link to known collusion merchants or senders | Yes, earlier (`src/cons3.py`) | No link. See question A. |
| Start a new conversation with only `handoff_src.zip` | Not needed | The checks were run locally. |

## Question A: why is merchant collusion stuck near 0.30?

**Answer: it is at its ceiling.** The agent's arithmetic is right: about 77% of collusion-labelled training rows look ordinary, so F1 is capped near 0.36–0.39 even with perfect detection of the patterned rows.

- The missed rows (1,102 decimal merchant payments, 667 P2P payments) share a sender or receiver with patterned collusion no more often than legit rows do (15.1% against 16.1%).
- They sit no closer in time to collusion events than legit rows do.
- The October score of 0.30 is within noise of the training range (0.32–0.41 by month): the public split has only about 130 collusion rows.

Nothing further to do here.

## Question B: are the fraud-type mix-ups training noise or deliberate lookalikes?

**Answer: deliberate lookalikes (the agent's hypothesis 2, but not fixable by features or thresholds).**

Evidence:
- **A model trained on the given types confuses them just as much.** Run T1 keeps the original fraud types in training. On ordinary phones it sends 359 takeover-labelled rows to SIM swap; our model sends 370. The same holds for every pair (mule → layering 350 against 349, bot → mule 295 against 315, cashback → phishing 245 against 242).
- **The rows are identical to their lookalike.** Takeover-labelled rows predicted SIM swap: 92% fresh SIM, 91% own phone, 71% at night, 8% away from home. Real SIM swaps: 89%, 97%, 69%, 7%.
- **The block phones show the reverse variant.** SIM-swap block phones look like takeovers: 79% new device, 79% away from home, 7% fresh SIM.
- **The block type is the true October label.** Moving 28 rows from takeover to SIM swap on block-phone evidence raised the public score twice (0.5332 → 0.5389 → 0.5405).

So each fraud type has a minority variant that behaves like another type, in training and in October.

| True type | Looks like | Rows on ordinary phones (3 months) |
|---|---|---|
| account_takeover | sim_swap | 370 |
| mule_account | layering_chain | 349 |
| velocity_bot | mule_account | 315 |
| fake_cashback | phishing_collect | 242 |
| phishing_collect | fake_cashback | 161 |
| loan_app_extortion | phishing_collect | 79 |
| sim_swap | account_takeover | 68 |
| refund_abuse | qr_swap | 58 |
| merchant_collusion | qr_swap | 54 |

## New fix that came out of question B: `sub_G6.csv`

A row on a block phone reveals the true type of the whole event it belongs to. `src/prop3.py` applies this:

- Take each row on a fraud-block phone (outside the shared bot pool).
- Find rows on ordinary phones that share an account with it within 6 hours.
- If such a row is predicted as the block type's lookalike (pairs in the table above), give it the block's type.
- Repeat once from the changed rows (two hops).

| Check on training data | Value |
|---|---|
| Rows changed (3 months) | 124 |
| Changed rows that match the label | 112 |
| Changed rows where the old prediction was right | 1 |
| Macro-F1 | 0.5008 → 0.5028 |

Flips outside the lookalike pairs are wrong on training data (the model is right there), so they are excluded. A third hop is no better than a coin flip (4 of 8), so it stops at two.

October: 14 rows change (layering → mule 9, mule → bot 2, SIM swap → takeover 3). Expected gain is about +0.001 to +0.002. `run_all.sh` step 9 reproduces the file.

## Open questions for the agent

1. **Can the lookalike variants be told apart without a block phone?** Only about 3% of rows sit on block phones, so about 1,700 variant rows over three months stay mislabelled (roughly 280 in October). No row-level or one-hop feature separates them. Is there an event-level signature (burst size, timing shape, amount sequence, account age mix) that differs between, say, a real mule burst and a bot-variant burst that looks like one?
2. **Why do fraud blocks have gaps in their phone numbers?** 55% of the numbers in the fraud range never appear in any row, with a different share per type (SIM swap 29% missing, takeover 74%). We use the density, but do not know what the missing numbers were allocated to. If they map to rows that exist under another phone ID, that would be a new leak.
3. **Which of the five unprobed classes is weakest in October?** Bot, layering, QR swap, refund and loan sum to about 2.50 F1. Bot is the suspect: October has about 7 pool bursts where 13 are expected, and every October bot prediction comes from block or pool phones.
4. **The cashback / loan / collusion block boundaries** are uncertain by a few phones in October (about 6 rows). The densities of those three blocks are too similar to place them.

## What not to revisit

These were checked and closed; see `PLAN.md` sections 12–15 and `AUDIT_REVIEW.md`.

- The 61% ordinary-looking fraud rows are random (no entity, time, attribute or digit signal).
- File position carries no label signal (chunks are uniform).
- Thresholds are exhausted (probabilities are bimodal).
- On a block phone, the block's type beats the row's pattern.
