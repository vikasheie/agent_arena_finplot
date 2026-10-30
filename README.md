# FinPilot: a proactive multi-agent financial copilot

**Agent Arena 2026, Track 02 (FinPilot).** FinPilot reads a user's bank transactions, categorises them, studies how the user spends, forecasts where their money is heading, simulates what-if decisions against that forecast, and gives specific recommendations. Every recommendation records the evidence that triggered it, and a verifier agent checks each one before the user sees it.

All LLM calls go through **Mesh API** (`https://api.meshapi.ai/v1`, OpenAI-compatible) in [`finpilot/llm.py`](finpilot/llm.py). That file is the only place the LLM client is created.

---

## What it does

| Capability | Where | How |
|---|---|---|
| Ingestion | `finpilot/ingest.py` | CSV upload (a signed `amount` column, or separate `debit`/`credit` columns, with several date formats) or manual entry |
| Categorisation | `agents/categorizer.py` | Merchant memory, then LLM batches (`gpt-4.1-mini`), then a **self-check loop**: invalid, low-confidence or sign-contradicting labels are re-asked with `gpt-4.1` plus examples from the user's own history. User corrections teach the merchant memory |
| Behaviour analysis | `agents/analyzer.py` | Monthly totals by category, recurring-payment detection (same merchant, ±15% amount, 20–40 day spacing), robust-z anomaly detection against the merchant's or category's own baseline, category trends, payday impulse buying. Each finding gets an evidence id (`P1`, `P2`, ...) |
| Forecasting | `agents/forecaster.py` | Income + active recurring payments + recency-weighted variable spend (one-off anomalies excluded), over 6 months with an ~80% band. **Backtested** month by month on history it hasn't seen |
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
- `eval/ground_truth.csv`: 337 seeded synthetic transactions for one person (Apr–Oct 2026), with the true category for each row and **4 injected anomalies**. They're generated by `python -m eval.generate_data`; `data/sample_transactions.csv` is the same data without labels.
- `eval/categorization_testset.csv`: **74 hand-labelled held-out transactions** from merchants that never appear in the main data (Chaayos, Namma Yatri, Star Health, CRED, ...). This is the honest categorisation test.

Current measured numbers (see `eval/RESULTS.md` for the full tables):

| Metric | Result |
|---|---|
| Anomaly detection, 4 injected | precision 100%, recall 100% |
| Recurring-payment detection, 9 known | recall 100% |
| Forecast 1-month spend MAPE (rolling backtest, 4 origins) | **11.3%** vs 18.7% for the naive last-month baseline |
| Categorisation, keyword baseline on held-out set | 35.1% accuracy (why an LLM agent is needed) |
| Categorisation, LLM agent on held-out set | see `eval/RESULTS.md` (requires Mesh credit) |

Honest caveats: the data is synthetic and small. The keyword baseline scores 98% on the persona data only because the same author wrote both the generator and the keywords, which is why the held-out set exists. The forecast misses most in July (a one-off ₹35k purchase, which can't be predicted by design) and October (a planned jump in dining spend).

`pytest` runs 8 tests with a fake LLM, so the loops are proven without spending credits.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # put your Mesh key (rsk_...) in MESH_API_KEY
python -m eval.generate_data      # writes the synthetic dataset
uvicorn finpilot.main:app --reload
# open http://127.0.0.1:8000 → "Load demo data" → "Run analysis" → ask "Can I afford a ₹40,000 trip next month?"
```

API: `POST /api/transactions/upload`, `POST /api/transactions`, `POST /api/run`, `GET /api/insights`, `POST /api/ask`, `POST /api/simulate`, `POST /api/goals`, `GET /api/trail/{run_id}`, `GET /api/digests`, `GET /health`. The interactive docs are at `/docs`.

Optional settings in `.env`: `MESH_MODEL_FAST` / `MESH_MODEL_SMART` (defaults `openai/gpt-4.1-mini` / `openai/gpt-4.1`), `DIGEST_CRON_DAY_OF_WEEK`, `DIGEST_CRON_HOUR`, and `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` for Telegram delivery.

## Stack

Python 3.10+, FastAPI, LangGraph, the OpenAI SDK pointed at Mesh API, SQLite, APScheduler, and a Jinja2 + vanilla JS interface. Secrets live only in the gitignored `.env`; the CI uses GitHub Actions secrets.
