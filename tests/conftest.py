import csv
from pathlib import Path

import pytest

from finpilot import config, db

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return config.DB_PATH


@pytest.fixture
def persona_rows():
    with open(ROOT / "eval" / "ground_truth" / "priya.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["amount"] = float(r["amount"])
    return rows


@pytest.fixture
def labelled_db(fresh_db, persona_rows):
    """Persona history with ground-truth categories already applied."""
    ids = db.insert_transactions(persona_rows)
    for i, r in zip(ids, persona_rows):
        db.set_category(i, r["category"], "user", 1.0)
    return fresh_db
