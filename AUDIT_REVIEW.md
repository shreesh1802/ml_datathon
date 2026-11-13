# Review of the external audit (14 Nov 2026)

An outside audit of our entry proposed four fixes and projected a score of 0.551–0.556. This note records what we checked, what we kept and what we rejected.

**Summary:** one observation from the audit held up and everything else was rejected. The audit's script was never used to produce a submission. The two files that raised our score came from our own code, built on that one observation.

| | Public score |
|---|---|
| Before (`sub_G1.csv`) | 0.5332 |
| After the first correction (`sub_G3.csv`) | 0.5389 |
| After building it into the segmentation (`sub_G5.csv`) | 0.5405 |

## What we took from it

| Audit observation | How we checked it | What it became |
|---|---|---|
| Fraud-block phone numbers have gaps; legit ones have none | In the training run, all but 32 of the 45,177 rows after the last missing number are legit | October's fraud range ends at 474551, so 4 rows we had left as legit became collusion |
| (Not in the audit; found while checking the above) each fraud block has its own gap density | SIM swap has 71% of numbers present, takeover 26%; the training labels switch exactly where the density switches (425032 / 425039) | 28 rows moved from takeover to SIM swap |

`src/segblock2.py` adds the gap signal to the block segmentation, so no boundary is typed by hand. `run_all.sh` step 8 reproduces `sub_G5.csv`.

## What we rejected

| Audit claim | Promised gain | Why we rejected it |
|---|---|---|
| Merchant payments of exactly 1000/2000/2500/5000/10000 are 99% collusion | +0.018 | In labelled training data those amounts are QR swap (807 rows), collusion (501), refund abuse (314) and legit (247). Our model already flags all of them as fraud, so the rule changes 0 rows in every training month. |
| `COLLECT_REQUEST` typo in `features2.py` makes four features all zeros | +0.001 to +0.002 (shared with the cleaning fix) | The typo does not exist. Line 70 says `'COLLECT'`, locally and on GitHub. |
| Our segmentation skipped the October bot block and mislabelled 35 phones | part of +0.003 | False. Our segmentation already had a bot block at exactly the range the audit gives (473750–473838). |
| The SIM-swap block is only 9 phones (473899–473921) | part of +0.003 | It contradicts the training run, where the SIM block is the densely numbered one. Following the audit would have moved rows the wrong way; the density evidence moved 28 rows the opposite way, and the score rose. |
| Rows that look like another fraud type should keep their pattern, not their block's class (28 rows) | +0.002 | On training data, when pattern and block disagree, the label sides with the block 404 times of 483 and with the pattern 6 times. The audit's script also crashed in its fake-cashback step when we tried to run this rule on labelled data. |
| Only apply the block override when the model is not confident of another type | not stated | Same evidence as the row above: it would undo correct overrides. |
| Run the first cleaning pass with the extra features (`--extra feats2.pkl`) | +0.001 to +0.002 (shared with the typo fix) | Not tested. It needs a full retrain of about 90 minutes, and its justification rested on the collusion and typo claims, which were both wrong. |
| The file is two separately shuffled halves, so row position leaks information | not stated | The observation is true (training has six such chunks), but every chunk has the same dates, senders and label mix. Position says nothing about a row's label. |

## Why the audit went wrong

It only looked at `test.csv`, which has no labels. Its collusion rule, for example, counted how rare round amounts are among merchant payments and assumed every such row was collusion. The labelled months show three fraud types share those amounts.

Its projected score of 0.551–0.556 was therefore not real. The one true observation in it was worth +0.0073 once it was checked and applied properly.

## Where the checks live

| Script | What it checks |
|---|---|
| `src/ver1.py` | Collusion amount rule on labelled months |
| `src/ver2.py`, `src/ver3.py` | Gap density in both runs; audit's block table against ours |
| `src/ver4.py`, `src/ver5.py` | Density change points for every block boundary |
| `src/xb1.py` | Pattern versus block on confident disagreements |
| `src/ord1.py`–`src/ord3.py` | File chunk order and what defines a chunk |
