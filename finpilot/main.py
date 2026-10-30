"""FastAPI app: REST API + a single-page UI. Run with `uvicorn finpilot.main:app --reload`.

Every /api request is scoped to one profile (Priya, Aarav, Sharma family, or the user's own data),
chosen by the `X-Profile` header; each profile has its own SQLite database.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from finpilot import config, coordinator, db, ingest, llm, profiles, scheduler
from finpilot.agents import categorizer, forecaster

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    for pid in profiles.PROFILES:  # create every profile DB and seed the personas on first start
        profiles.activate(pid)
        if pid != "me" and not db.all_transactions():
            try:
                profiles.load_seed(pid)
            except FileNotFoundError:
                pass
        elif profiles.PROFILES[pid].get("goal") and not db.goals():  # profiles created before goals existed
            db.add_goal(**profiles.PROFILES[pid]["goal"])
    db.current_db.set(None)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="FinPilot", description="Proactive multi-agent financial copilot", lifespan=lifespan)


async def use_profile(x_profile: str = Header(profiles.DEFAULT)) -> str:
    # async on purpose: the contextvar set here is copied into the threadpool that runs the sync endpoint
    try:
        profiles.activate(x_profile)
    except KeyError:
        raise HTTPException(404, f"unknown profile '{x_profile}'")
    return x_profile


api = APIRouter(prefix="/api", dependencies=[Depends(use_profile)])


class ManualTx(BaseModel):
    date: str
    description: str = Field(min_length=1)
    amount: float = Field(gt=0)
    kind: str = Field("expense", pattern="^(expense|income)$")


class Correction(BaseModel):
    category: str


class Question(BaseModel):
    question: str = Field(min_length=3)


class Decision(BaseModel):
    type: str = Field(pattern="^(purchase|recurring_change|savings_goal)$")
    amount: float | None = None
    monthly_delta: float | None = None
    month_offset: int = 1
    months: int | None = None


class Goal(BaseModel):
    name: str
    target_amount: float = Field(gt=0)
    target_date: str
    saved_so_far: float = 0


class Balance(BaseModel):
    opening_balance: float


class Toggle(BaseModel):
    enabled: bool


class RecState(BaseModel):
    state: str = Field(pattern="^(active|accepted|dismissed)$")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"categories": config.CATEGORIES})


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": llm.available(), "model_fast": config.MODEL_FAST,
            "model_smart": config.MODEL_SMART, "next_digest": scheduler.next_run()}


@app.get("/api/eval")
def eval_results():
    """Self-evaluation numbers produced by `python -m eval.run_eval` (committed in eval/results.json)."""
    path = ROOT.parent / "eval" / "results.json"
    if not path.exists():
        raise HTTPException(404, "run `python -m eval.run_eval` first")
    return json.loads(path.read_text())


@app.get("/api/profiles")
def list_profiles():
    return [profiles.summary(pid) for pid in profiles.PROFILES]


@api.post("/profile/reset")
def reset_profile(profile: str = Depends(use_profile)):
    if profile == "me":
        db.reset()
        return {"inserted": 0}
    return {"inserted": profiles.load_seed(profile)}


@api.get("/overview")
def overview():
    txs = db.all_transactions()
    latest = db.latest_run("pipeline")
    with db.connect() as conn:
        running = conn.execute("SELECT id FROM runs WHERE kind = 'pipeline' AND status = 'running' "
                               "ORDER BY id DESC LIMIT 1").fetchone()
    return {"transactions": len(txs), "uncategorized": sum(1 for t in txs if not t["category"]),
            "first_date": txs[0]["date"] if txs else None, "last_date": txs[-1]["date"] if txs else None,
            "opening_balance": float(db.get_setting("opening_balance", "0") or 0),
            "digest_enabled": scheduler.digest_enabled(),
            "running_run_id": running["id"] if running else None,
            "latest_run_id": latest["id"] if latest else None}


@api.post("/transactions/upload")
async def upload(file: UploadFile = File(...)):
    try:
        result = ingest.ingest_csv((await file.read()).decode("utf-8", errors="replace"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"inserted": result["inserted"], "skipped_duplicates": result["skipped_duplicates"]}


@api.post("/transactions")
def add_tx(tx: ManualTx):
    try:
        tx_id = ingest.ingest_manual(tx.date, tx.description, tx.amount, tx.kind)
    except ValueError as e:
        raise HTTPException(400, str(e))
    stats = categorizer.categorize_pending()
    return {"id": tx_id, "categorization": stats}


@api.get("/transactions")
def list_tx():
    return db.all_transactions()


@api.post("/transactions/{tx_id}/category")
def correct(tx_id: int, body: Correction):
    try:
        categorizer.correct(tx_id, body.category)
    except (ValueError, StopIteration) as e:
        raise HTTPException(400, str(e) or "transaction not found")
    return {"ok": True}


@api.post("/settings/opening-balance")
def set_balance(body: Balance):
    db.set_setting("opening_balance", str(body.opening_balance))
    return {"ok": True}


@api.post("/run")
def run():
    if not db.all_transactions():
        raise HTTPException(400, "no transactions yet")
    return {"run_id": coordinator.start_pipeline()}


@api.get("/runs/{run_id}")
def run_status(run_id: int):
    status = coordinator.run_status(run_id)
    if not status:
        raise HTTPException(404, "run not found")
    return status


@api.get("/insights")
def insights():
    latest = db.latest_run("pipeline")
    if not latest:
        return {"state": None}
    return {"run_id": latest["id"], "started_at": latest["started_at"], "state": latest["state"],
            "recommendations": db.recommendations(latest["id"])}


@api.post("/recommendations/{rec_id}/state")
def set_rec_state(rec_id: int, body: RecState):
    """accepted = user will do it; dismissed = skip (fed back so the Advisor stops suggesting it)."""
    db.set_recommendation_state(rec_id, body.state)
    return {"ok": True}


@api.post("/ask")
def ask(q: Question):
    try:
        return coordinator.ask(q.question)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, str(e))
    except RuntimeError as e:  # Mesh API returned an error (e.g. 402 insufficient balance)
        raise HTTPException(502, str(e))


@api.post("/simulate")
def simulate(d: Decision):
    fc = forecaster.forecast()
    if "error" in fc:
        raise HTTPException(400, fc["error"])
    sim = forecaster.simulate(fc, d.model_dump(exclude_none=True))
    coordinator.add_earliest(sim)
    return sim


@api.get("/trail/{run_id}")
def trail(run_id: int):
    return db.trail(run_id)


@api.post("/goals")
def add_goal(g: Goal):
    return {"id": db.add_goal(g.name, g.target_amount, ingest.parse_date(g.target_date), g.saved_so_far)}


@api.get("/goals")
def list_goals():
    return db.goals()


@api.post("/digest/toggle")
def toggle_digest(body: Toggle):
    db.set_setting("digest_enabled", "1" if body.enabled else "0")
    return {"enabled": body.enabled}


@api.post("/digest/run-now")
def digest_now():
    try:
        d = scheduler.digest_job()
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    if not d:
        raise HTTPException(400, "digest needs transactions and MESH_API_KEY")
    return d


@api.get("/digests")
def list_digests():
    return {"next_scheduled": scheduler.next_run(), "enabled": scheduler.digest_enabled(), "digests": db.digests()}


@api.get("/runs")
def list_runs():
    with db.connect() as conn:
        rows = conn.execute("SELECT id, kind, status, started_at FROM runs ORDER BY id DESC LIMIT 30").fetchall()
    return [dict(r) for r in rows]


app.include_router(api)
