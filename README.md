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

The UI (`/`) follows the official FinPilot demo flow, and every screen shows the agents' real output:
- **Live Demo**: persona cards, each with a goal; a goal-progress ring; the CSV loader; a **Run FinPilot** button; playback controls (Pause / Next step / Reset, 1x / 2x, Auto / Step-by-step, keyboard shortcuts); and the stages Import → Categorize → Analyze → Forecast → Advise → Digest.
  - **Categorize** shows every transaction with its category, confidence and **why** (the LLM's reason, a merchant-memory hit, a keyword rule, or "corrected by you"), plus inline correction.
  - **Analyze** shows the spending donut, month-by-month income vs spend, and finding cards with evidence ids.
  - **Forecast** shows the 6-month projection with its 80% band, the goal and the safety buffer, and a one-line summary (numbers from Python, optionally phrased by the Mesh LLM).
  - **Advise** shows verified actions with *saves ₹X/mo*, **Accept**, **Why?** (reason, evidence and the Verifier's note) and **skip** (fed back to the Advisor).
  - **Digest** shows the FinPilot Bot weekly message and the APScheduler schedule.
  - The right-hand **Agent log** streams each step live, marking Mesh calls with model, tokens and latency, and other steps as "(no LLM)".
  - An **Ask a what-if** box routes questions Coordinator → Simulator → Verdict, with a baseline vs decision chart.
- **How it works**: the architecture graph with a trace player (14 steps), the self-evaluation table, and the per-run audit log.

## What it does

| Capability | Where | How |
|---|---|---|
| Ingestion | `finpilot/ingest.py` | CSV upload (a signed `amount` column, or separate `debit`/`credit` / `Withdrawal Amt.`/`Deposit Amt.` columns, several date formats) or manual entry. Re-uploading the same file skips rows already stored, while genuine repeats inside one statement are kept |
| Categorisation | `agents/categorizer.py` | Merchant memory, then LLM batches (`gpt-4.1-mini`), then a **self-check loop**: invalid, low-confidence or sign-contradicting labels are re-asked with `gpt-4.1` plus examples from the user's own history. User corrections teach the merchant memory. Every label stores **why** (the LLM's reason, a memory hit, a keyword rule, or "corrected by you"). Rows labelled by the offline fallback are re-labelled by the LLM on the next run once Mesh is reachable |
| Behaviour analysis | `agents/analyzer.py` | Monthly totals by category; recurring-payment detection (monthly or quarterly, ±15% amount); robust-z anomaly detection against the merchant's, the category's or the user's own baseline; rising bills; category trends; top discretionary merchants; payday impulse buying. Each finding gets an evidence id (`P1`, `P2`, ...) |
| Forecasting | `agents/forecaster.py` | Income + active recurring payments (quarterly bills placed in the month they fall due) + recency-weighted variable spend (one-off anomalies excluded), over 6 months with an ~80% band. **Backtested** month by month on history it hasn't seen |
| What-if simulation | `forecaster.simulate`, `forecaster.earliest_purchase` | One-off purchase, a new or cancelled recurring cost, or a savings goal, run against the real trajectory. Gives a verdict (`affordable`, `tight`, `not_affordable`), the lowest balance, the shortfall versus a safety buffer (one month of essential spend), and which categories to cut. For a purchase that isn't affordable now it answers **"when can I?"**: it tries the purchase in each of the next 24 projected months and reports the earliest comfortable month, the earliest tight month, and the monthly set-aside |
| Advice | `agents/advisor.py` | The advisor (`gpt-4.1`) must cite evidence ids and name specific merchants. The **verifier** rejects advice that cites nothing, cites unknown ids, claims more rupees than the evidence shows, doesn't name the merchant, is generic, claims an implausibly small saving, or asks the user to cut **essentials** (medicines, school fees, rent, EMIs, money sent to family). Rejected advice goes back to the advisor with the reasons, for up to 3 rounds |
| Goal tracking | `forecaster.goal_progress` | Required saving per month vs. projected surplus, giving `on_track` or `behind` (evidence `G1`, ...), which feeds the advisor |
| Conversational Q&A | `coordinator.ask` | The coordinator classifies a question and sends it to the simulator ("can I afford a ₹40k trip next month?", "when can I buy a ₹1 lakh laptop?") or the Q&A agent (free-form questions over the user's real data). Every number in an LLM-written explanation must match a figure the simulator computed; otherwise a template sentence built from the real numbers is shown and the rejection is logged |
| Proactive digest | `finpilot/scheduler.py` | An **APScheduler cron job** (default Monday 09:00, every profile with the digest switched on) re-runs the pipeline and delivers one insight and one action by **Telegram and/or e-mail** when configured, and always in the app. See [Proactive delivery](#proactive-delivery) |
| Decision trail | `decision_trail` table, `/api/trail/{run_id}` | Every agent step and every Mesh call (model, tokens, latency, response preview) is stored, and each recommendation stores its `evidence_ids`, `reason` and `verifier_note` |

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
5. **Never show a number the code didn't compute.** In testing, the model once wrote the ₹ sign as a broken escape (`\u000a` followed by "9"), turning "₹20,000" into "920,000". Prompts now require plain-ASCII "INR" amounts, and a number check compares every figure in an LLM explanation with the simulator's output before the user sees it.
6. **Advice protects essentials.** The Verifier rejects any recommendation that cuts medicines, school fees, rent, EMIs or family support, whatever the prompt said. The digest headlines the most specific non-essential action (e.g. the AC timer for a rising electricity bill).
7. **Failures are visible, and they don't cascade.** A circuit breaker in `finpilot/llm.py` stops calling Mesh for 60 seconds after a billing or auth error (401, 402 or 403). If Mesh fails (outage, no balance), the categorizer falls back to keyword rules, marked `category_source=fallback`, and the error is written to the decision trail. Nothing pretends the LLM ran.

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

`pytest` runs 16 tests with a fake LLM, so the loops, the guards and the delivery channels are proven without spending credits.

**Try your own data:** `data/samples/neha_hdfc_may-oct-2026.csv` is a separate 6-month statement in an HDFC-style export layout (`Date, Narration, Withdrawal Amt., Deposit Amt.`, dates dd/mm/yyyy). Pick **My data** and drop it on the upload card. It contains a ₹42,999 Croma purchase and a ₹4,600 Zomato order as anomalies, plus food delivery that rises every month. `data/samples/kiran_sbi_aug-oct-2026.csv` is a smaller 50-row, 3-month statement in an SBI-style layout (`Txn Date, Description, Debit, Credit`, dates like "01 Aug 2026"), with a ₹54,990 electronics anomaly, a refund, a rising electricity bill and growing food delivery.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # put your Mesh key (rsk_...) in MESH_API_KEY
python -m eval.generate_data      # writes the three persona datasets (already committed)
uvicorn finpilot.main:app --reload
# open http://127.0.0.1:8000 → pick Priya / Aarav / Sharma family → "Run FinPilot" → watch the stages and the Agent log
# then "Ask a what-if": "Can I buy a laptop for 1 lakh? If not now, when?"
```

API (send the `X-Profile` header, one of `priya`, `aarav`, `sharma` or `me`): `POST /api/transactions/upload`, `POST /api/transactions`, `POST /api/run` (starts the run in the background and returns `run_id`), `GET /api/runs/{id}` (live per-agent status and tokens), `GET /api/insights`, `POST /api/recommendations/{id}/state` (`accepted` / `active` / `dismissed`), `POST /api/profile/reset`, `POST /api/ask`, `POST /api/simulate`, `POST /api/goals`, `POST /api/digest/toggle`, `POST /api/digest/run-now`, `GET /api/digests` (includes which channels are configured), `GET /api/trail/{run_id}`, `GET /api/eval`, `GET /health`. The interactive docs are at `/docs`. Each profile's data lives in `profiles/<id>.db` (gitignored).

Optional settings in `.env`: `MESH_MODEL_FAST` / `MESH_MODEL_SMART` (defaults `openai/gpt-4.1-mini` / `openai/gpt-4.1`), `DIGEST_CRON_DAY_OF_WEEK`, `DIGEST_CRON_HOUR`, and the delivery settings below.

## Proactive delivery

The weekly digest is produced by a real background scheduler, not a button: `finpilot/scheduler.py` registers an **APScheduler cron job** (`DIGEST_CRON_DAY_OF_WEEK` / `DIGEST_CRON_HOUR`, default Monday 09:00 server time). It runs for every profile whose digest switch is on, re-runs the whole agent pipeline on the latest transactions, and sends one insight plus the single best verified action:

```
FinPilot weekly digest

Your week: Bescom Electricity rose from ~INR 3,350 to INR 5,250 in 2026-10 (+56.7%)
One action: Set your AC to 26°C night-mode and switch off unused appliances to save INR 1,900/month
```

It is delivered through every configured channel. The in-app copy is always kept (Digest stage → "Delivered", and the audit log), and `delivered_via` records what happened, e.g. `telegram+email` or `in_app`. **Send now** in the Digest stage runs the same job immediately.

**Telegram**
1. In Telegram, message **@BotFather**, send `/newbot`, and copy the bot token.
2. Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`.
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=987654321
   ```

**E-mail (any SMTP server)**

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587                # 587 = STARTTLS, 465 = SSL
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=you@gmail.com      # optional, defaults to SMTP_USER
DIGEST_EMAIL_TO=you@gmail.com
```

For Gmail, create an **App Password** (Google Account → Security → 2-Step Verification → App passwords); your normal password won't work.

Restart the server after editing `.env`. The Digest stage shows which channels are on. A failed send is logged and never blocks the in-app copy.

## Stack

Python 3.10+, FastAPI, LangGraph, the OpenAI SDK pointed at Mesh API, SQLite, APScheduler, and a Jinja2 + vanilla JS interface. Secrets live only in the gitignored `.env`; the CI uses GitHub Actions secrets.
