"""SQLite persistence. Amount convention: positive = money in, negative = money out."""
import json
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime

from finpilot import config

# The active profile's database file; set per request / per background job. None -> config.DB_PATH.
current_db: ContextVar[str | None] = ContextVar("current_db", default=None)

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    category TEXT,
    category_source TEXT,          -- memory | llm | llm_recheck | fallback | user
    confidence REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS merchant_memory (
    merchant_key TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    source TEXT NOT NULL           -- llm | user
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,            -- pipeline | question | digest
    status TEXT NOT NULL DEFAULT 'done',  -- running | done | error
    started_at TEXT NOT NULL,
    state_json TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    action TEXT NOT NULL,
    monthly_impact REAL,
    evidence_ids TEXT NOT NULL,    -- JSON list of pattern / forecast ids that triggered it
    reason TEXT NOT NULL,
    verified INTEGER NOT NULL,
    verifier_note TEXT,
    state TEXT NOT NULL DEFAULT 'active',  -- active | dismissed (user toggle; dismissed advice is fed back)
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_trail (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    agent TEXT NOT NULL,
    step TEXT NOT NULL,
    detail TEXT NOT NULL,          -- JSON payload: inputs, outputs, evidence
    model TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    target_amount REAL NOT NULL,
    target_date TEXT NOT NULL,
    saved_so_far REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    insight TEXT NOT NULL,
    action TEXT NOT NULL,
    delivered_via TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(path: str | None = None):
    conn = sqlite3.connect(path or current_db.get() or config.DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN status TEXT NOT NULL DEFAULT 'done'",
    "ALTER TABLE recommendations ADD COLUMN state TEXT NOT NULL DEFAULT 'active'",
]


def init_db(path: str | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        for stmt in MIGRATIONS:  # upgrade databases created by older versions
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass


def insert_transactions(rows: list[dict], path: str | None = None) -> list[int]:
    ids = []
    with connect(path) as conn:
        for r in rows:
            cur = conn.execute(
                "INSERT INTO transactions (date, description, amount, created_at) VALUES (?, ?, ?, ?)",
                (r["date"], r["description"], float(r["amount"]), now()),
            )
            ids.append(cur.lastrowid)
    return ids


def all_transactions(path: str | None = None) -> list[dict]:
    with connect(path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM transactions ORDER BY date, id")]


def uncategorized(path: str | None = None) -> list[dict]:
    with connect(path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM transactions WHERE category IS NULL ORDER BY date, id")]


def set_category(tx_id: int, category: str, source: str, confidence: float, path: str | None = None) -> None:
    with connect(path) as conn:
        conn.execute(
            "UPDATE transactions SET category = ?, category_source = ?, confidence = ? WHERE id = ?",
            (category, source, confidence, tx_id),
        )


def get_memory(path: str | None = None) -> dict[str, str]:
    with connect(path) as conn:
        return {r["merchant_key"]: r["category"] for r in conn.execute("SELECT * FROM merchant_memory")}


def remember_merchant(key: str, category: str, source: str, path: str | None = None) -> None:
    with connect(path) as conn:
        # A user correction always wins over an LLM label.
        existing = conn.execute("SELECT source FROM merchant_memory WHERE merchant_key = ?", (key,)).fetchone()
        if existing and existing["source"] == "user" and source != "user":
            return
        conn.execute(
            "INSERT OR REPLACE INTO merchant_memory (merchant_key, category, source) VALUES (?, ?, ?)",
            (key, category, source),
        )


def start_run(kind: str, path: str | None = None) -> int:
    with connect(path) as conn:
        return conn.execute("INSERT INTO runs (kind, status, started_at) VALUES (?, 'running', ?)",
                            (kind, now())).lastrowid


def finish_run(run_id: int, state: dict, status: str = "done", path: str | None = None) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE runs SET state_json = ?, status = ? WHERE id = ?",
                     (json.dumps(state, default=str), status, run_id))


def get_run(run_id: int, path: str | None = None) -> dict | None:
    with connect(path) as conn:
        r = conn.execute("SELECT id, kind, status, started_at FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(r) if r else None


def latest_run(kind: str = "pipeline", path: str | None = None) -> dict | None:
    with connect(path) as conn:
        r = conn.execute(
            "SELECT * FROM runs WHERE kind = ? AND status = 'done' AND state_json IS NOT NULL ORDER BY id DESC LIMIT 1",
            (kind,)
        ).fetchone()
    if not r:
        return None
    return {"id": r["id"], "started_at": r["started_at"], "state": json.loads(r["state_json"])}


def log_decision(run_id: int | None, agent: str, step: str, detail: dict, model: str | None = None,
                 path: str | None = None) -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO decision_trail (run_id, agent, step, detail, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, agent, step, json.dumps(detail, default=str), model, now()),
        )


def trail(run_id: int, path: str | None = None) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute("SELECT * FROM decision_trail WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]


def save_recommendations(run_id: int, recs: list[dict], path: str | None = None) -> None:
    with connect(path) as conn:
        for r in recs:
            conn.execute(
                """INSERT INTO recommendations (run_id, title, action, monthly_impact, evidence_ids, reason,
                   verified, verifier_note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, r["title"], r["action"], r.get("monthly_impact"), json.dumps(r["evidence_ids"]),
                 r["reason"], int(r.get("verified", False)), r.get("verifier_note", ""), now()),
            )


def recommendations(run_id: int, path: str | None = None) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute("SELECT * FROM recommendations WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return [{**dict(r), "evidence_ids": json.loads(r["evidence_ids"])} for r in rows]


def set_recommendation_state(rec_id: int, state: str, path: str | None = None) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE recommendations SET state = ? WHERE id = ?", (state, rec_id))


def dismissed_recommendations(limit: int = 20, path: str | None = None) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute("SELECT title, action FROM recommendations WHERE state = 'dismissed' ORDER BY id DESC "
                            "LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def token_usage(run_id: int, path: str | None = None) -> dict[str, int]:
    """Total tokens per agent for a run, from the Mesh call log in the decision trail."""
    usage: dict[str, int] = {}
    for e in trail(run_id, path):
        if e["step"] == "llm_call" and e["detail"].get("tokens"):
            usage[e["agent"]] = usage.get(e["agent"], 0) + int(e["detail"]["tokens"])
    return usage


def add_goal(name: str, target_amount: float, target_date: str, saved_so_far: float = 0.0,
             path: str | None = None) -> int:
    with connect(path) as conn:
        return conn.execute(
            "INSERT INTO goals (name, target_amount, target_date, saved_so_far, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, target_amount, target_date, saved_so_far, now()),
        ).lastrowid


def goals(path: str | None = None) -> list[dict]:
    with connect(path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM goals ORDER BY target_date")]


def save_digest(run_id: int, insight: str, action: str, delivered_via: str, path: str | None = None) -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO digests (run_id, insight, action, delivered_via, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, insight, action, delivered_via, now()),
        )


def digests(limit: int = 10, path: str | None = None) -> list[dict]:
    with connect(path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM digests ORDER BY id DESC LIMIT ?", (limit,))]


def get_setting(key: str, default: str = "", path: str | None = None) -> str:
    with connect(path) as conn:
        r = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(key: str, value: str, path: str | None = None) -> None:
    with connect(path) as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))


def reset(path: str | None = None) -> None:
    with connect(path) as conn:
        for t in ("transactions", "runs", "recommendations", "decision_trail", "digests", "merchant_memory", "goals"):
            conn.execute(f"DELETE FROM {t}")
