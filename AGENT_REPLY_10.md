# Reply to the agent's "Lookalike Fraud Event Separation Report" (14 Nov 2026, night)

This replies to `lookalike_hypotheses_report.md`. It lists what was checked, where we agree, and what remains.

**Status:** best public score is 0.5453 (`sub_G9.csv`). `sub_G12.csv` is built, validated and reproducible; it already contains every row this report proposes to change.

## Assessment of the report

This report was run on the file and its numbers match ours. Its swap counts agree with the brief to within a few rows (356 against 361 for takeover-as-SIM, 346 against 346 for mule-as-layering, 315 against 323 for bot-as-mule), and event purity of 99% confirms the swap is per event.

| Hypothesis | Agent's verdict | Ours |
|---|---|---|
| 1. Swap follows an entity across events | Does not separate | Agreed |
| 2. Swap follows the calendar | Does not separate | Agreed |
| 3. Swap follows a role's account attributes | Does not separate | Agreed |
| 4. Swap follows event size or sequence | Does not separate | Agreed. The size effects it shows (for example 41% for one-row SIM sessions against 12% for four-row) are too weak to act on. |
| 5. Trace in noise columns | Does not separate | Agreed |
| 6. Amounts drawn by the label's generator | Does not separate | Agreed |
| 7. Block phones and generation order | Does not separate at 100 rows | Agreed |

**Conclusion, shared by both sides: the lookalike swap is a per-event coin flip. It cannot be resolved from the data except where a row lands on a fraud-block phone.**

## The two "missed" block links

The report offers two rules it says `src/prop3.py` misses, worth 14 October rows. Checked on the full data (`src/bonus1.py`):

| Rule | Training (any window) | October rows it flags | Of those, still the old type in G12 |
|---|---|---|---|
| Bot-block P2P phone → rows predicted mule sharing the hub account | 78 of 91 match (2 were right before, 10 legit-labelled) | 10 | 0 |
| Takeover-block phone → rows predicted SIM swap by the same sender, within 24 hours | 16 of 20 match (0 were right before, 4 legit-labelled) | 4 | 0 |

All 14 rows are already changed in `sub_G12.csv` (and in G7 onward): the 12-hour window with two hops reaches them. Nothing to add.

Widening the other pairs does not help either: at 7 days the same-account link turns wrong for SIM-block → predicted takeover (2 right, 4 wrong) and phishing-block → predicted cashback (11 right, 3 wrong), and reaches no further October rows.

## The three checks the report says the file cannot answer

| Check | Status |
|---|---|
| 1. Label-ordered blocks of devices or accounts below 420000 | Tested on the full data on 14 Nov: account and device numbers below 420000 carry no label signal (adversarial AUC 0.50–0.52; `src/cons2.py`). |
| 2. Legit history and balance continuity of event participants | Tested: balances are not carried between rows for anyone (`src/cand1.py`). |
| 3. Class-size-aware thresholds between paired classes | Done: `make_sub2.py --sizeaware`, in `sub_G12.csv`. Worth +0.0005 on held-out months, mostly fake cashback. |

## Where this leaves the project

- The lookalike question is closed as irreducible, by two independent analyses.
- The random fraud rows were closed earlier, by about 30 checks.
- Every remaining lever tested in the last day is worth +0.001 or less.
- We have no explanation for scores of 0.60 and above from this data.

No further task for the agent.
