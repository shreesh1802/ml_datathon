# Out-of-fold confusion matrix for the G5 recipe

Recipe: average of runs E1 and M1, F/2 decision rule, block override from the density-aware segmentation (`s2`).
Scored on July-September rows against the given (noisy) training labels. Macro-F1 = 0.5008.
Public score of the same recipe on October: 0.5405.

Note: training labels are noisy (about 12% of fraud is labelled legit; about 61% of fraud-labelled rows look like legit rows and are random). Test labels are clean.

## Per-class scores

| class | true rows (3 months) | predicted rows | precision | recall | F1 | threshold | October predictions (G5) | October F1 from public probes (G1) |
|---|---|---|---|---|---|---|---|---|
| legit | 1506394 | 1538066 | 0.977 | 0.998 | 0.987 | nan | 256500 |  |
| mule_account | 10237 | 3900 | 0.772 | 0.294 | 0.426 | 0.212 | 795 | 0.541 |
| velocity_bot | 8105 | 3509 | 0.847 | 0.367 | 0.512 | 0.256 | 297 |  |
| layering_chain | 6818 | 3774 | 0.767 | 0.425 | 0.547 | 0.273 | 575 |  |
| sim_swap | 4071 | 2198 | 0.684 | 0.369 | 0.48 | 0.235 | 354 | 0.524 |
| account_takeover | 4048 | 1438 | 0.807 | 0.287 | 0.423 | 0.209 | 245 | 0.511 |
| phishing_collect | 6696 | 3714 | 0.652 | 0.362 | 0.465 | 0.232 | 609 | 0.518 |
| qr_swap | 7262 | 3558 | 0.771 | 0.378 | 0.507 | 0.253 | 575 |  |
| refund_abuse | 3231 | 1443 | 0.833 | 0.372 | 0.514 | 0.257 | 270 |  |
| fake_cashback | 2530 | 927 | 0.686 | 0.251 | 0.368 | 0.182 | 173 | 0.517 |
| loan_app_extortion | 1983 | 621 | 0.833 | 0.261 | 0.397 | 0.193 | 100 |  |
| merchant_collusion | 2538 | 765 | 0.83 | 0.25 | 0.384 | 0.193 | 89 | 0.3 |

## Confusion matrix (rows = given label, columns = prediction)

| given \ predicted | legit | mule_acco | velocity_ | layering_ | sim_swap | account_t | phishing_ | qr_swap | refund_ab | fake_cash | loan_app_ | merchant_ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 1502649 | 465 | 447 | 447 | 245 | 178 | 880 | 609 | 185 | 111 | 79 | 99 |
| mule_acco | 6829 | 3009 | 8 | 349 | 10 | 4 | 8 | 7 | 6 | 2 | 2 | 3 |
| velocity_ | 4755 | 323 | 2971 | 4 | 6 | 5 | 14 | 13 | 7 | 3 | 1 | 3 |
| layering_ | 3846 | 27 | 10 | 2895 | 9 | 4 | 9 | 9 | 6 | 1 | 0 | 2 |
| sim_swap | 2442 | 5 | 7 | 17 | 1503 | 70 | 14 | 7 | 3 | 2 | 0 | 1 |
| account_t | 2454 | 10 | 9 | 10 | 375 | 1160 | 7 | 11 | 6 | 1 | 3 | 2 |
| phishing_ | 4026 | 13 | 4 | 13 | 11 | 5 | 2422 | 26 | 6 | 163 | 4 | 3 |
| qr_swap | 4431 | 11 | 12 | 8 | 7 | 6 | 15 | 2742 | 13 | 4 | 2 | 11 |
| refund_ab | 1932 | 7 | 6 | 5 | 2 | 1 | 8 | 63 | 1202 | 2 | 1 | 2 |
| fake_cash | 1597 | 10 | 5 | 8 | 8 | 3 | 242 | 11 | 4 | 636 | 6 | 0 |
| loan_app_ | 1322 | 12 | 15 | 7 | 18 | 1 | 81 | 5 | 1 | 0 | 517 | 4 |
| merchant_ | 1783 | 8 | 15 | 11 | 4 | 1 | 14 | 55 | 4 | 2 | 6 | 635 |

## Where given `merchant_collusion` rows go, by kind of payment

| payment kind | legit | mule_account | velocity_bot | layering_chain | sim_swap | account_takeover | phishing_collect | qr_swap | refund_abuse | fake_cashback | loan_app_extortion | merchant_collusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2P | 667 | 8 | 0 | 11 | 3 | 1 | 14 | 48 | 0 | 2 | 0 | 15 |
| merchant, decimal | 1102 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 4 | 24 |
| merchant, multiple of 500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 1 | 0 | 2 | 521 |
| merchant, other integer | 14 | 0 | 15 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 75 |

## Notes on reading the matrix

- **The `legit` column is mostly the random fraud.** About 61% of fraud-labelled rows are indistinguishable from legit rows, so each class's recall is capped near 0.3–0.4.
- **Off-diagonal fraud cells are directed pairs, not block errors.** Almost all sit on ordinary phones (only 8 of the 323 bot→mule rows and 5 of the 375 takeover→SIM rows are on block phones). The large ones: takeover labelled, predicted SIM swap (375); mule labelled, predicted layering (349); bot labelled, predicted mule (323); cashback labelled, predicted phishing (242); phishing labelled, predicted cashback (163); loan labelled, predicted phishing (81); refund labelled, predicted QR swap (63); collusion labelled, predicted QR swap (55). These fit the statement's "filed under the wrong type" training noise, or deliberate lookalike types. Which of the two is not established.
- **Collusion:** every given-collusion row on a merchant payment that is a multiple of 500 is predicted as fraud (521 of 531 as collusion). The 1,102 decimal merchant rows and 667 P2P rows predicted legit are random rows.
