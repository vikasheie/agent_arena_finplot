# FinPilot: a proactive multi-agent financial copilot

**Agent Arena 2026, Track 02 (FinPilot).** FinPilot reads a user's bank transactions, categorises them, studies how the user spends, forecasts where their money is heading, simulates what-if decisions against that forecast, and gives specific recommendations. Every recommendation records the evidence that triggered it, and a verifier agent checks each one before the user sees it.

All LLM calls go through **Mesh API** (`https://api.meshapi.ai/v1`, OpenAI-compatible) in [`finpilot/llm.py`](finpilot/llm.py). That file is the only place the LLM client is created.

---

## The demo

Three built-in profiles, each with 7 months of realistic Indian bank data, plus **My data** for your own CSV:

| Profile | Story |
|---|---|
| **Priya**, 28, software engineer | Subscriptions creeping up (ChatGPT added in July), food delivery rising every month, payday impulse shopping, a ₹35k Croma purchase |
| **Aarav**, 23, freelancer | Irregular Upwork/Fiverr income, PG rent, gaming, a ₹59k laptop on a thin buffer |
| **Sharma family** | Two incomes, home and car loan EMIs, quarterly school fees, parents' medicines, electricity climbing from ₹3,100 to ₹5,250 as AC use grows, a hospital bill and Diwali jewellery |

The UI (`/`) has six tabs:
- **Overview**: KPI tiles, a donut of where the money goes, the cash-flow forecast with its 80% band and safety buffer, verified recommendations (each with an on/off switch, and switched-off ones are fed back to the Advisor), Analyzer findings with evidence ids, goals, and fixed bills.
- **Agents & trace**: the LangGraph diagram lights up live while a run is in progress, with a card per agent showing Mesh calls, tokens and output, a live event feed, and trace replay.
- **Ask & what-if**: chat routed by the Coordinator, plus an instant what-if simulator with a baseline vs. scenario chart.
- **Transactions**: drag-and-drop CSV upload, manual entry, and category correction that teaches the merchant memory.
- **Weekly digest**: per-profile on/off switch, the next scheduled time, and delivery history.
- **Audit trail**: every run, and every step with its model, tokens and JSON payload.

## What it does

| Capability | Where | How |
|---|---|---|
| Ingestion | `finpilot/ingest.py` | CSV upload (a signed `amount` column, or separate `debit`/`credit` columns, with several date formats) or manual entry |
| Categorisation | `agents/categorizer.py` | Merchant memory, then LLM batches (`gpt-4.1-mini`), then a **self-check loop**: invalid, low-confidence or sign-contradicting labels are re-asked with `gpt-4.1` plus examples from the user's own history. User corrections teach the merchant memory |
| Behaviour analysis | `agents/analyzer.py` | Monthly totals by category; recurring-payment detection (monthly or quarterly, ±15% amount); robust-z anomaly detection against the merchant's, the category's or the user's own baseline; rising bills; category trends; top discretionary merchants; payday impulse buying. Each finding gets an evidence id (`P1`, `P2`, ...) |
| Forecasting | `agents/forecaster.py` | Income + active recurring payments (quarterly bills placed in the month they fall due) + recency-weighted variable spend (one-off anomalies excluded), over 6 months with an ~80% band. **Backtested** month by month on history it hasn't seen |
| What-if simulation | `forecaster.simulate` | One-off purchase, a new or cancelled recurring cost, or a savings goal, run against the real trajectory. Gives a verdict (`affordable`, `tight`, `not_affordable`), the lowest balance, the shortfall versus a safety buffer (one month of essential spend), and which categories to cut |
| Advice | `agents/advisor.py` | The advisor (`gpt-4.1`) must cite evidence ids and name specific merchants. The **verifier** rejects advice that cites nothing, cites unknown ids, claims more rupees than the evidence shows, doesn't name the merchant, or is generic. Rejected advice goes back to the advisor with the reasons, for up to 3 rounds |
| Goal tracking | `forecaster.goal_progress` | Required saving per month vs. projected surplus, giving `on_track` or `behind` (evidence `G1`, ...), which feeds the advisor |
| Conversational Q&A | `coordinator.ask` | The coordinator classifies a question and sends it to the simulator ("can I afford a ₹40k trip next month?") or the Q&A agent (free-form questions over the user's real data) |
| Proactive digest | `finpilot/scheduler.py` | An **APScheduler cron job** (default Monday 09:00) re-runs the pipeline and delivers one insight and one action through Telegram (if configured) or in the app |
| Decision trail | `decision_trail` table, `/api/trail/{run_id}` | Every agent step and every Mesh call (model, tokens, response preview) is stored, and each recommendation stores its `evidence_ids`, `reason` and `verifier_note` |

## How the agents cooperate

The flow is an explicit **LangGraph** state machine in [`finpilot/coordinator.py`](finpilot/coordinator.py):

```
START → coordinator ──(uncategorised rows?)──► categorizer ─┐
             └──────────────────────────────► analyzer ◄────┘
                                                  │
                                              forecaster ──(no LLM / no forecast)──► finalize → END
                                                  │
                                    ┌────────► advisor
                                    │             │
                     rejected & rounds < 3     verifier
                                    └─────────────┤
                                                  └──(all verified)──► finalize → END
```

| Agent | Job | Own prompt / tools |
|---|---|---|
| Coordinator | Plans the run (skips categorisation if nothing is new), sends questions to the right agent, finalises | Routing prompt; LangGraph conditional edges |
| Categorizer | Labels transactions | Category prompt; merchant memory; recheck with few-shot examples |
| Analyzer | Patterns and anomalies | Statistical detectors; narrative prompt that must cite `[P#]` |
| Forecaster | Trajectory, backtest, what-if, goals | Deterministic cash-flow model; explanation prompt |
| Advisor | Specific recommendations | Advice prompt with a mandatory evidence contract |
| Verifier | Checks every recommendation against the data | Rule-based checker of amounts, evidence and specificity |
| Q&A | Free-form money questions | Q&A prompt over the user's aggregates |

## Design decisions

1. **Numbers come from code; language comes from the LLM.** Forecasts, anomalies and simulations are deterministic, so they can be tested and measured. The LLM never makes up a rupee figure that the verifier can't check.
2. **The verifier is a real loop, not a formality.** It's rule-based on purpose: a second LLM agreeing with the first proves little, but "you claimed ₹5,000/month while P3 shows ₹299" is a fact. `tests/test_agents.py::test_pipeline_advisor_verifier_loop_iterates` proves the graph loops.
3. **Anomalies are judged against the merchant's own history.** A ₹2,000 fuel fill-up isn't unusual for "transport" just because cab rides cost ₹300. This raised anomaly precision on the synthetic data from 18% to 100% (see `eval/`).
4. **One-off anomalies are kept out of the forecast baseline.** Without this, a single ₹35k laptop made the forecast worse than a naive "same as last month" guess.
5. **Failures are visible.** If Mesh fails (outage, no balance), the categorizer falls back to keyword rules, marked `category_source=fallback`, and the error is written to the decision trail. Nothing pretends the LLM ran.

## Self-evaluation

Run `python -m eval.run_eval` for everything, or add `--no-llm` for the deterministic parts and the keyword baseline. Results are written to [`eval/RESULTS.md`](eval/RESULTS.md) and [`eval/results.json`](eval/results.json).

Test data:
- `eval/ground_truth/{priya,aarav,sharma}.csv`: 947 seeded synthetic transactions across three personas (Apr–Oct 2026), with the true category for every row and **9 injected anomalies**. They're generated by `python -m eval.generate_data`; `data/profiles/*.csv` are the same rows without labels, and they're what the app loads.
- `eval/categorization_testset.csv`: **74 hand-labelled held-out transactions** from merchants that never appear in the persona data (Chaayos, Namma Yatri, Star Health, CRED, ...). This is the honest categorisation test.

Current measured numbers (see `eval/RESULTS.md` for the full tables):

| Metric | Priya | Aarav | Sharma |
|---|---|---|---|
| Anomaly detection, precision / recall | 100% / 100% | 100% / 100% | 100% / 100% |
| Recurring-payment recall (11 / 6 / 13 known) | 100% | 100% | 100% |
| Forecast, 1-month outflow MAPE (4 rolling origins) | 10.8% | **23.2%** | **16.8%** |
| Naive "same as last month" MAPE | 9.1% | 75.1% | 21.5% |

| Categorisation | Accuracy |
|---|---|
| Keyword baseline on the held-out set | 33.8% (why an LLM agent is needed) |
| LLM agent on the held-out set | see `eval/RESULTS.md` (needs Mesh credit) |

Honest caveats:
- The data is synthetic and small.
- Priya spends very steadily, so the naive baseline edges out the model for her. The model wins clearly where it matters: irregular income (Aarav) and lumpy quarterly bills (Sharma).
- The keyword baseline does well on the persona data only because the same author wrote both the generator and the keywords. That's why the held-out set exists.

`pytest` runs 8 tests with a fake LLM, so the loops are proven without spending credits.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # put your Mesh key (rsk_...) in MESH_API_KEY
python -m eval.generate_data      # writes the three persona datasets (already committed)
uvicorn finpilot.main:app --reload
# open http://127.0.0.1:8000 → pick Priya / Aarav / Sharma family → "▶ Run agents" → watch "Agents & trace"
# then "Ask & what-if": "Can we afford a ₹1.5 lakh family trip in December?"
```

API (send the `X-Profile` header, one of `priya`, `aarav`, `sharma` or `me`): `POST /api/transactions/upload`, `POST /api/transactions`, `POST /api/run` (starts the run in the background and returns `run_id`), `GET /api/runs/{id}` (live per-agent status and tokens), `GET /api/insights`, `POST /api/recommendations/{id}/toggle`, `POST /api/ask`, `POST /api/simulate`, `POST /api/goals`, `POST /api/digest/toggle`, `GET /api/trail/{run_id}`, `GET /health`. The interactive docs are at `/docs`. Each profile's data lives in `profiles/<id>.db` (gitignored).

Optional settings in `.env`: `MESH_MODEL_FAST` / `MESH_MODEL_SMART` (defaults `openai/gpt-4.1-mini` / `openai/gpt-4.1`), `DIGEST_CRON_DAY_OF_WEEK`, `DIGEST_CRON_HOUR`, and `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` for Telegram delivery.

## Stack

Python 3.10+, FastAPI, LangGraph, the OpenAI SDK pointed at Mesh API, SQLite, APScheduler, and a Jinja2 + vanilla JS interface. Secrets live only in the gitignored `.env`; the CI uses GitHub Actions secrets.
