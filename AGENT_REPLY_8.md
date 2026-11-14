# Reply to the agent's "Verified Feature Additions (features3.py)" note (14 Nov 2026, afternoon)

This replies to "Next-Step Strategy: Verified Feature Additions & sub_G8 Roadmap". It lists what was done for each point, the results, and what is still open.

**Status:** the retrain worked. `sub_G9.csv` is built: training macro-F1 0.5057 against 0.5030 for `sub_G8.csv`. Neither is uploaded yet; best public score is still 0.5418 (`sub_G6.csv`).

(Correction to earlier replies: `AGENT_REPLY_6.md` and `AGENT_REPLY_7.md` are dated 15 Nov. They were written on 14 Nov; the deadline is 15 Nov, 10:00 IST.)

## What the agent proposed, and what was done

| Agent's point | Done? | Result |
|---|---|---|
| Add `src/features3.py` (37 features) and retrain | Yes | Real gain: +0.0029 training macro-F1, in every month. |
| Amount features: whole rupees not a multiple of 10, exact phishing amounts | Kept | These carry the gain. |
| Cross-role "fraud ring pool" features | Dropped from the all-months model | The explanation does not hold and the features add nothing there (see below). |
| `under1000` causes phishing false positives on 900–980 collect requests | Checked, wrong | The model already calls all 820 such training rows legit (3 false alarms). |
| Blend E1, M1, E2, M2 | Changed | The old models dilute the gain. The blend is E2n + M2 only. |

## Results on held-out months (leave-one-month-out, same decision rule and block override)

| Model or blend | Macro-F1 | July | August | September |
|---|---|---|---|---|
| E1 (old all-months model) | 0.5005 | 0.5026 | 0.4966 | 0.5000 |
| E2 (all 37 new features) | 0.5035 | 0.5050 | 0.4989 | 0.5043 |
| E2n (without the 6 ring features) | 0.5028 | 0.5048 | 0.4981 | 0.5033 |
| M1 (old monthly model) | 0.4984 | 0.5002 | 0.4952 | 0.4977 |
| M2 (monthly, all 37) | 0.5018 | 0.5054 | 0.4958 | 0.5021 |
| E1 + M1 (the G5 blend) | 0.5008 | 0.5027 | 0.4972 | 0.5001 |
| E2 + M2 | 0.5033 | 0.5060 | 0.4981 | 0.5036 |
| **E2n + M2 (used)** | **0.5037** | 0.5062 | 0.4988 | 0.5037 |

Per-class change, E2n + M2 against E1 + M1: phishing +0.014, account_takeover +0.004, loan +0.007, fake_cashback +0.004, sim_swap +0.003; no class loses more than 0.001.

## Why the ring features were left out of the all-months model

- **The explanation does not check out.** On accounts with ring activity, the old model already catches about 95% of fraud-labelled rows. Among rows it calls legit, ring-linked rows are fraud 3–5% of the time against a 2.3% base (`src/f3chk2.py`).
- **They add nothing in the blend.** E2n + M2 scores 0.5037 against 0.5033 for E2 + M2.
- **They count activity across all four months.** Features that follow accounts across months have misled us before (they validate within July–September and fail in October). The monthly model M2 keeps its ring counts, because there they are computed within one month only.

## Where the amount features help

Checked on the old model's output (`src/f3chk.py`):

| Cell | Training rows | Fraud-labelled | Old model missed |
|---|---|---|---|
| P2P, whole rupees not a multiple of 10, ₹500 or more | 7,501 | 6,094 | 132 |
| P2P collect from a new payee, same amount pattern, no "under" flag | 332 | 282 | 56 |
| P2P, exact phishing amount | 2,642 | 1,868 | 19 |

The old model already caught most of these through other features, so the gain is in the margins: the 56 missed collect rows, and cleaner separation of phishing from legit and from fake cashback.

## The new submission

`sub_G9.csv` = E2n + M2 blend, F/2 rule, block override (`s2`), then the event corrections with G8's settings.

| | G8 | G9 |
|---|---|---|
| Training macro-F1 before event corrections | 0.5008 | 0.5037 |
| Training macro-F1 after | 0.5030 | 0.5057 |
| October rows changed by event corrections | 29 | 30 |

G9 differs from G8 on 58 October rows: 22 phishing → legit, 9 legit → phishing, 8 phishing → cashback, 8 cashback → phishing, 3 QR swap → phishing, and 8 others. `run_all.sh` step 12 reproduces it.

## What is still open

1. **More amount-digit structure.** This worked because fraud amounts are drawn differently from legit ones (99.4% of legit P2P whole-rupee amounts are multiples of 10). The same idea may apply to other digits or fields: last digits of merchant amounts by category, amounts just above round numbers, `login_to_txn_sec` or `pin_retries` value sets by fraud type. A systematic search of value-set differences between patterned fraud and legit rows, per transaction type, has not been done.
2. **Phishing against fake cashback.** They still trade rows (8 each way between G8 and G9). An amount-based rule may separate them.
3. **Which unprobed class is weakest in October.**

## What not to revisit

- The ring-pool explanation, and the `under1000` false-positive claim.
- Everything under "What not to revisit" in `AGENT_REPLY_2.md` to `AGENT_REPLY_7.md`.
