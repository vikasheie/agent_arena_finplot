"""User profiles. Each profile has its own SQLite file, so personas never mix data."""
import re
from pathlib import Path

from finpilot import config, db, ingest

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "data" / "profiles"

PROFILES = {
    "priya": {"name": "Priya", "avatar": "PR", "opening_balance": 60000,
              "who": "28 · Software engineer · Bengaluru",
              "story": "Earns well, spends fast. Subscriptions keep creeping up.",
              "goal": {"name": "Emergency fund", "target_amount": 150000, "target_date": "2027-03-31"},
              "blurb": "28 · software engineer, Bengaluru · subscriptions creeping up"},
    "aarav": {"name": "Aarav", "avatar": "AA", "opening_balance": 15000,
              "who": "23 · Freelance designer · Pune",
              "story": "Irregular income. Wants a 6-month safety buffer.",
              "goal": {"name": "6-month safety buffer", "target_amount": 180000, "target_date": "2027-06-30"},
              "blurb": "23 · freelance designer · irregular income, thin buffer"},
    "sharma": {"name": "The Sharma family", "avatar": "RS", "opening_balance": 250000,
               "who": "Rohan & Meera · Bengaluru",
               "story": "Two incomes, two loans. School fees due every quarter, electricity climbing.",
               "goal": {"name": "Family trip to Goa", "target_amount": 150000, "target_date": "2027-04-30"},
               "blurb": "Household of 4 · home-loan EMI, school fees, rising electricity"},
    "me": {"name": "My data", "avatar": "ME", "opening_balance": 0,
           "who": "Your own bank statement",
           "story": "Upload a CSV or log transactions by hand.",
           "goal": None,
           "blurb": "Upload your own bank CSV or add transactions manually"},
}
DEFAULT = "priya"
_VALID = re.compile(r"^[a-z0-9_]{1,32}$")


def path_for(pid: str) -> str:
    if not _VALID.match(pid) or pid not in PROFILES:
        raise KeyError(pid)
    Path(config.PROFILE_DIR).mkdir(parents=True, exist_ok=True)
    return str(Path(config.PROFILE_DIR) / f"{pid}.db")


def activate(pid: str) -> str:
    """Point the db layer at this profile (for the current request / job context)."""
    path = path_for(pid)
    db.current_db.set(path)
    db.init_db()
    return path


def load_seed(pid: str) -> int:
    """Reset the profile and load its seed transactions (personas only)."""
    seed = SEED_DIR / f"{pid}.csv"
    if not seed.exists():
        raise FileNotFoundError(f"no seed data for {pid}; run `python -m eval.generate_data`")
    db.reset()
    meta = PROFILES[pid]
    db.set_setting("opening_balance", str(meta["opening_balance"]))
    if meta.get("goal"):
        db.add_goal(**meta["goal"])
    return ingest.ingest_csv(seed.read_text())["inserted"]


def summary(pid: str) -> dict:
    seed = SEED_DIR / f"{pid}.csv"
    return {"id": pid, **PROFILES[pid], "has_seed": seed.exists(), "seed_file": seed.name if seed.exists() else None}
