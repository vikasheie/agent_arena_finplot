# FinPilot self-evaluation results

Generated 2026-10-30T13:30:25 · models `openai/gpt-4.1-mini` / `openai/gpt-4.1` · LLM used: False

## Categorisation accuracy

| Dataset | n | Keyword baseline | LLM agent (Mesh) | LLM macro-F1 |
|---|---|---|---|---|
| Held-out, 74 hand-labelled, unseen merchants | 74 | 37.8% | n/a | n/a |
| Persona: priya | 360 | 97.8% | n/a | n/a |
| Persona: aarav | 317 | 70.7% | n/a | n/a |
| Persona: sharma | 270 | 76.7% | n/a | n/a |

## Anomaly detection (injected anomalies)

| Persona | Injected | Flagged | Precision | Recall |
|---|---|---|---|---|
| priya | 3 | 3 | 100% | 100% |
| aarav | 3 | 3 | 100% | 100% |
| sharma | 3 | 3 | 100% | 100% |

## Recurring-payment detection

| Persona | Known recurring | Recall | Missed |
|---|---|---|---|
| priya | 11 | 100% | - |
| aarav | 6 | 100% | - |
| sharma | 13 | 100% | - |

## Forecast backtest (rolling origin, 1-month-ahead total outflow)

| Persona | Origins | FinPilot MAPE | Naive last-month MAPE | Net-cashflow MAE (INR) |
|---|---|---|---|---|
| priya | 4 | **10.83%** | 9.1% | 11,288 |
| aarav | 4 | **23.2%** | 75.06% | 20,817 |
| sharma | 4 | **16.84%** | 21.5% | 30,142 |

<details><summary>priya: month-by-month</summary>

| Month | Predicted | Actual | Abs % error |
|---|---|---|---|
| 2026-07 | 89,052 | 112,410 | 20.78% |
| 2026-08 | 83,398 | 96,877 | 13.91% |
| 2026-09 | 88,542 | 96,708 | 8.44% |
| 2026-10 | 88,552 | 88,403 | 0.17% |

</details>

<details><summary>aarav: month-by-month</summary>

| Month | Predicted | Actual | Abs % error |
|---|---|---|---|
| 2026-07 | 36,135 | 30,092 | 20.08% |
| 2026-08 | 33,657 | 85,532 | 60.65% |
| 2026-09 | 29,556 | 27,676 | 6.79% |
| 2026-10 | 27,819 | 29,374 | 5.3% |

</details>

<details><summary>sharma: month-by-month</summary>

| Month | Predicted | Actual | Abs % error |
|---|---|---|---|
| 2026-07 | 124,335 | 158,389 | 21.5% |
| 2026-08 | 117,690 | 118,980 | 1.08% |
| 2026-09 | 115,982 | 140,950 | 17.71% |
| 2026-10 | 162,177 | 222,433 | 27.09% |

</details>
