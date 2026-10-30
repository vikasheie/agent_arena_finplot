# FinPilot self-evaluation results

Generated 2026-10-30T10:24:21 · models `openai/gpt-4.1-mini` / `openai/gpt-4.1`

## Held-out categorisation test set (74 hand-labelled, unseen merchants)

| Method | Accuracy | Macro-F1 | n |
|---|---|---|---|
| keyword_baseline | 35.1% | 0.429 | 74 |

## Persona history categorisation (337 synthetic rows)

| Method | Accuracy | Macro-F1 | n |
|---|---|---|---|
| keyword_baseline | 98.2% | 0.992 | 337 |

## Anomaly detection (4 injected anomalies)

Precision **100%**, recall **100%** (4/4 found, 4 flagged).

## Recurring-payment detection

Recall **100%** of 9 known recurring payments; 9 detected in total.

## Forecast backtest (rolling origin, 1-month ahead)

Spend MAPE **11.31%** vs naive last-month baseline 18.74% · net-cashflow MAE INR 15,229

| Month | Predicted spend | Actual spend | Abs % error |
|---|---|---|---|
| 2026-07 | 79,258 | 105,264 | 24.71% |
| 2026-08 | 74,431 | 73,003 | 1.96% |
| 2026-09 | 73,645 | 73,879 | 0.32% |
| 2026-10 | 69,412 | 84,932 | 18.27% |
