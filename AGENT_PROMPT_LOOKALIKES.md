I'm attaching two files: `AGENT_TASK_LOOKALIKES.md` (the brief) and `lookalike_rows.csv` (7.3 MB, 29,916 rows). Read the brief first, then work on the CSV directly. The competition closes in a few hours, so I need a fast, honest answer more than a thorough one.

## The question

In this fraud dataset, about one fraud event in five carries the label of a different fraud type from the one its behaviour matches (a layering chain labelled `mule_account`, a fresh-SIM drain labelled `account_takeover`, and so on). Find anything observable that tells a swapped event from a normal one. If nothing does, tell me that.

This is the only thing worth your time. It is worth about +0.05 macro-F1; every other idea we have is worth +0.002 or less.

## How to work

1. **Load the CSV and run code on it.** Every number you report must come from code you actually ran on this file. Do not estimate, recall or extrapolate figures. If the file fails to load, stop and tell me; do not continue from memory.
2. **Build the swap flag first.** Among rows with `split == 'train'` and blank `block_class`, a row is *swapped* when `label` and `predicted` are two different fraud types, and *normal* when they are the same fraud type. Print the count of each per (`label`, `predicted`) pair and check it against the table in the brief (for example about 361 takeover rows predicted as SIM swap). If your counts are far off, stop and tell me.
3. **Group rows into events** using the rules at the end of the brief's data section, and confirm that events are almost always all-swapped or all-normal.
4. **Test the seven hypotheses in the brief in the order given.** For each one, print a table of swap rate by the candidate variable with row counts, for the four largest pairs first: takeover-as-SIM, mule-as-layering, bot-as-mule, cashback-as-phishing.
5. **Stop on a hit.** A hit is a rule that separates swapped from normal with over 70% precision on at least 100 training rows for a pair, and holds in July, August and September separately. If you find one, skip the remaining hypotheses and report it.
6. **Do not repeat anything in the brief's "What does NOT separate them" table.**

## What to send back

Keep it short. For each hypothesis tested, three things:
- the code you ran (one block),
- its printed output, pasted as is,
- one line: "separates" or "does not separate", with the precision and row count if it does.

Then one final section:
- **If something separates a pair:** the exact rule as a pandas expression on the CSV's columns, its precision and row count per month, and how many `split == 'test'` rows it would change.
- **If nothing does after all seven:** say "irreducible on this file" and list the two or three things you would check next that this file cannot answer.

Do not write a strategy document, a projected score or a new pipeline. I will test any rule you find on the full data myself.
