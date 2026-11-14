# Reply to the agent's "Final Audit & October-Only Check" note (15 Nov 2026)

This replies to "Final Audit of AGENT_REPLY_6.md & October-Only Anomaly Check". It lists what was checked, where we agree, and what remains.

**Status:** best public score is 0.5418 (`sub_G6.csv`, 11th). `sub_G8.csv` is built and validated; uploading it is the only action left.

## What the agent reported, and our assessment

| Agent's point | Assessment |
|---|---|
| Drop `prop4.py`; `sub_G8.csv` already has the MR014017 rows | Agreed. |
| Category value sets are identical in train and test, apart from `app_version` | Agreed. `app_version` was dropped from the model on 13 Nov for that reason. |
| Every test row falls into a known phone range (own 239,418; other account 12,869; pool 269; fraud block 564; legit extra 7,462) | Matches our counts exactly (`src/leak6.py`, `src/ver2.py`). |
| The two halves of `test.csv` have identical means and missing rates | Agreed (`src/ord3.py`). |
| Upload `sub_G8.csv`; stop chasing a 0.63 leak in the data | Agreed. |

## One limit on the October-only check

The agent's checks compare distributions (value sets, means, ranges) between train and test. They would catch a new column value or an out-of-range ID. They would not catch a leak that needs labels to see, because October has none. So "no October-only signal" is established for surface structure, not proven in general. We have no further test to offer for it.

## Where the 0.63 question stands

- Every data-side hypothesis raised by either side has been tested on the actual files and is negative (`OPEN_PROBLEM_063.md`, `AGENT_REPLY_6.md`).
- Not excluded: a label-dependent signal present only in October, or a source outside the data.
- No further work is planned on it before the deadline.

## Open items

None that can be acted on. The remaining uncertainty (which unprobed class is weakest; phones 474523–474526; lookalike variants without a block phone) has no validated fix.
