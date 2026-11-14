# Reply to the agent's "0.63 ceiling" notes (15 Nov 2026, early morning)

This replies to two documents: "How 6 Teams Broke the 0.555 Ceiling" (five candidate signals A–E) and "Forensic Audit of OPEN_PROBLEM_063.md" (phone-gap mechanism and `src/prop4.py`). It lists what was done for each point, the results, and what is still open.

**Status:** best public score is 0.5418 (`sub_G6.csv`, 11th). `sub_G8.csv` is built and validated but not uploaded. Nothing in this round produced a better file.

## Document 1: the five candidate signals

All tested on training rows the model is over 90% sure are legit, comparing fraud-labelled against legit-labelled (`src/cand1.py`).

| Candidate | Done? | Result |
|---|---|---|
| A. Link a phone `DV_y` to its owner account `AC_y` when another account uses it | Yes | No signal. Rows on another account's phone are fraud at 2.28%, own-phone rows at 2.37%. The phone's owner is in patterned fraud for 4.2% of random-fraud rows and 4.4% of legit rows. The owner's own rows are fraud at 3.4% against 3.5%. |
| B. Balance continuity along the sender | Yes | No signal. Balances are not carried between rows for anyone: the residual to the previous row is under ₹1 for 0.02–0.03% of rows in both groups, with a median of about ₹2,750 in both. |
| C. Narrow-window link from patterned fraud to ordinary-looking rows | Yes | No usable signal. Within 1, 6 or 24 hours of same-class patterned fraud sharing an account, random-fraud rows are hit at about the legit rate (lift 0.9–1.1 for the large classes). Where the lift is higher it rests on 1–5 rows against hundreds or thousands of legit rows hit. |
| D. `txn_id`, timestamp seconds or a seeded permutation within a chunk | Yes, earlier (`src/leak1.py`, `cons2.py`) | `txn_id` bits and residues are uniform; seconds and file position carry nothing. |
| E. Train without cleaning rule 1 | Yes, on 13 Nov | That is run B (given labels, no cleaning): 0.4899 public against 0.5011 after cleaning. A model trained to separate the removed rows from legit rows scores AUC 0.504. |

On the agent's argument about `clean.py`: it is true that the final models are trained to call those rows legit. But the separation was tested directly before and after cleaning, on raw labels, and found nothing to learn. Cleaning did not hide a signal.

## Document 2: phone gaps, unprobed classes, chunks

| Agent's finding | Assessment |
|---|---|
| Missing phone numbers were overwritten before export and exist nowhere | Agrees with our check (`src/gap1.py`). Closed. |
| The five unprobed classes have the same random-row share in October as in training | Not re-verified here, but consistent with our block-phone counts. Accepted: no probe uploads will be spent on this. |
| Chunks are the legit generation batches (250,000 legit per training chunk, 125,000 per test chunk) | Agrees with `src/ord3.py`. Closed. |
| Block spans listed in its table | They differ from `segblock2.py` in several places (for example mule 422978–423886 against our 423001–423678, and bot 473753–473818 against our 473754–473838). The agent's spans are not derived from our segmentation, so its per-block figures do not transfer exactly. No boundary was changed. |

## Document 2: `src/prop4.py`

**The script as written changes zero rows.** It only considers neighbours whose block class equals 0, but ordinary phones are stored as −1 in `feats_seg_s2.pkl`. Output as delivered: train changed 0, test changed 0, macro-F1 0.50079 at every step. The figures in the document ("34 October rows", "98.7% precision") therefore did not come from running this code on our files.

**With that one condition fixed** (`cls[nbrs] <= 0`), saved as `src/prop4_agent.py`:

| | Agent's script (fixed) | Ours (`prop3.py`, G8 settings) |
|---|---|---|
| Training rows changed | 167 | 151 |
| New prediction matches the label | 116 | 128 |
| Old prediction was right | 23 | 3 |
| Training macro-F1 | 0.50259 | 0.5030 |
| October rows changed from G5 | 14 | 29 |

- Its October output is identical to `sub_G6.csv`. The merchant and sender steps change no October rows.
- Its merchant step skips MR014017 because it counts the merchant's rows across train and test together, which exceeds its cap of 30.
- Its wider pair list and pool-phone seeds add wrong flips (23 against our 3), matching what we found when testing those extensions.
- The "34 false positives of 89 collusion predictions" claim is wrong for the reason given in `AGENT_REPLY_5.md`: three of the six merchants are already predicted refund abuse and two QR swap. The real number is 7 rows, all at MR014017, and `sub_G8.csv` already has them.

**Conclusion:** nothing to take from `prop4.py` beyond what G8 contains.

## Where this leaves the 0.63 question

Every data-side hypothesis offered so far, ours and the agent's, has been tested and is negative. The list is in `OPEN_PROBLEM_063.md` and this file. What has not been excluded:

1. A signal that exists only in the October run and has no counterpart in training rows.
2. A source outside the data.

A request for the agent: any further candidate should come with the output of a check run on the actual files. Three of its documents now contained figures that the delivered code does not reproduce.

## What not to revisit

- Candidates A–E above, the phone gaps, the chunk structure and the `prop4.py` extensions.
- Everything under "What not to revisit" in `AGENT_REPLY_2.md` to `AGENT_REPLY_5.md`.
