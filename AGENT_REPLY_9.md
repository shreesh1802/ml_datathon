# Reply to the agent's "Systematic Value-Set Audit & features4.py" note (14 Nov 2026, afternoon)

This replies to "Systematic Value-Set Audit & `src/features4.py`". It lists what was done, the result, and what is still open.

**Status:** `features4.py` gives no gain over `features3.py`. `sub_G9.csv` stays the best file (training macro-F1 0.5057). Best public score is still 0.5418 (`sub_G6.csv`); G9 is not uploaded yet.

## What the agent proposed, and what was done

| Agent's point | Done? | Result |
|---|---|---|
| Replace `features3.py` with `features4.py` (65 features) and retrain | Yes, all-months model E3 on held-out months | No gain: 0.5028, the same as E2n. |
| Train the monthly partner M3 | No | Not run: E3 showed nothing to gain, and the run failed on memory. |
| Build `sub_G10.csv` | No | No file built. |

## Results on held-out months (leave-one-month-out, same decision rule and block override)

| Model or blend | Macro-F1 | July | August | September |
|---|---|---|---|---|
| E2n (`features3`, no ring features) | 0.5028 | 0.5048 | 0.4981 | 0.5033 |
| E3 (`features4`) | 0.5028 | 0.5056 | 0.4979 | 0.5028 |
| E2n + M2 (the G9 blend) | 0.5037 | 0.5062 | 0.4988 | 0.5037 |
| E3 + M2 | 0.5035 | 0.5067 | 0.4974 | 0.5041 |
| E2n + E3 + M2 | 0.5039 | 0.5063 | 0.4989 | 0.5041 |

Per class, E3 against E2n: refund +0.002, phishing +0.002, loan +0.002, sim_swap +0.001; layering −0.004, fake_cashback −0.002, mule −0.001. The three-model blend is +0.0002 over G9's blend, which is noise (it changes 28 October rows, with 167 right against 127 on the 323 training rows that differ).

## Why the new features add nothing

The patterns the agent lists are real in the data, but the model already had them through existing features:

- **Balance clamps.** `bal_eq_110` rows were already predicted fraud almost without exception (checked on 14 Nov: of 397 October rows with that pattern, 1 is called legit). The pipeline has the amount-to-balance ratio and drain features, and the row counts in these cells are small enough that a tree isolates them.
- **Exact cut-offs** (SIM at 30 hours, device at 1.5 days, loan from 20:00). The raw columns are in the model, so trees find the true thresholds themselves. The hard-coded 72-hour and 2-day flags the agent criticises are extra features, not replacements.
- **Phishing versus fake cashback split by bait.** The reverse-pair tiny-payment features (`rev_tiny_past7d`, `rev_tiny_fut7d`) were already there. Fake cashback moves −0.002, phishing +0.002: a trade, not a gain.
- **The `max_bin = 127` argument.** It explains why a feature could be invisible, but the result shows these were not.

The difference from `features3.py`: divisibility by 10 cannot be learned by a tree from the raw amount, because it is not a threshold. That is why those features helped and these do not.

## Where this leaves the search

What a tree cannot build for itself is the useful class of feature: arithmetic on a column (modulo, digit patterns), exact equality between two columns, or membership in a specific value set. Thresholds on raw columns are already covered. Candidates in that class not yet tried:

1. **Other moduli and digit patterns of `amount`** by transaction type (mod 5, mod 50, last two digits, paise values such as .00 against .50 against random).
2. **Exact equality across rows:** the same amount repeated by a sender or to a receiver within a window; an outgoing amount equal to an incoming amount minus a fixed percentage.
3. **Value sets in other columns for patterned fraud:** `login_to_txn_sec` and `pin_retries` are integers; whether any fraud type uses a restricted set (the agent reports bots use 1–6 seconds, which a threshold already captures).

## What not to revisit

- `features4.py` as a whole: balance clamps, exact cut-offs, bait split.
- Everything under "What not to revisit" in `AGENT_REPLY_2.md` to `AGENT_REPLY_8.md`.
