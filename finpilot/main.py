"""FastAPI app: REST API + a single-page UI. Run with `uvicorn finpilot.main:app --reload`."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from finpilot import config, coordinator, db, ingest, llm, scheduler
from finpilot.agents import categorizer, forecaster

ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT.parent / "data" / "sample_transactions.csv"
templates = Jinja2Templates(directory=str(ROOT / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="FinPilot", description="Proactive multi-agent financial copilot", lifespan=lifespan)


class ManualTx(BaseModel):
    date: str
    description: str
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"categories": config.CATEGORIES})


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": llm.available(), "model_fast": config.MODEL_FAST,
            "model_smart": config.MODEL_SMART, "next_digest": scheduler.next_run()}


@app.post("/api/transactions/upload")
async def upload(file: UploadFile = File(...)):
    try:
        ids = ingest.ingest_csv((await file.read()).decode("utf-8", errors="replace"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"inserted": len(ids)}


@app.post("/api/transactions")
def add_tx(tx: ManualTx):
    try:
        tx_id = ingest.ingest_manual(tx.date, tx.description, tx.amount, tx.kind)
    except ValueError as e:
        raise HTTPException(400, str(e))
    stats = categorizer.categorize_pending()
    return {"id": tx_id, "categorization": stats}


@app.get("/api/transactions")
def list_tx():
    return db.all_transactions()


@app.post("/api/transactions/{tx_id}/category")
def correct(tx_id: int, body: Correction):
    try:
        categorizer.correct(tx_id, body.category)
    except (ValueError, StopIteration) as e:
        raise HTTPException(400, str(e) or "transaction not found")
    return {"ok": True}


@app.post("/api/demo/load")
def load_demo():
    if not SAMPLE_CSV.exists():
        raise HTTPException(404, "run `python -m eval.generate_data` first")
    db.reset()
    db.set_setting("opening_balance", "50000")
    ids = ingest.ingest_csv(SAMPLE_CSV.read_text())
    return {"inserted": len(ids), "opening_balance": 50000}


@app.post("/api/settings/opening-balance")
def set_balance(body: Balance):
    db.set_setting("opening_balance", str(body.opening_balance))
    return {"ok": True}


@app.post("/api/run")
def run():
    if not db.all_transactions():
        raise HTTPException(400, "no transactions yet")
    return coordinator.run_pipeline()


@app.get("/api/insights")
def insights():
    latest = db.latest_run("pipeline")
    if not latest:
        return {"state": None}
    return {"run_id": latest["id"], "started_at": latest["started_at"], "state": latest["state"],
            "recommendations": db.recommendations(latest["id"])}


@app.post("/api/ask")
def ask(q: Question):
    try:
        return coordinator.ask(q.question)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, str(e))
    except RuntimeError as e:  # Mesh API returned an error (e.g. 402 insufficient balance)
        raise HTTPException(502, str(e))


@app.post("/api/simulate")
def simulate(d: Decision):
    fc = forecaster.forecast()
    if "error" in fc:
        raise HTTPException(400, fc["error"])
    return forecaster.simulate(fc, d.model_dump(exclude_none=True))


@app.get("/api/trail/{run_id}")
def trail(run_id: int):
    return db.trail(run_id)


@app.post("/api/goals")
def add_goal(g: Goal):
    return {"id": db.add_goal(g.name, g.target_amount, ingest.parse_date(g.target_date), g.saved_so_far)}


@app.get("/api/goals")
def list_goals():
    return db.goals()


@app.post("/api/digest/run-now")
def digest_now():
    try:
        d = scheduler.digest_job()
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    if not d:
        raise HTTPException(400, "digest needs transactions and MESH_API_KEY")
    return d


@app.get("/api/digests")
def list_digests():
    return {"next_scheduled": scheduler.next_run(), "digests": db.digests()}
