"""User profiles. Each profile has its own SQLite file, so personas never mix data."""
import re
from pathlib import Path

from finpilot import config, db, ingest

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "data" / "profiles"

PROFILES = {
    "priya": {"name": "Priya", "avatar": "P", "opening_balance": 60000,
              "blurb": "28 · software engineer, Bengaluru · subscriptions creeping up"},
    "aarav": {"name": "Aarav", "avatar": "A", "opening_balance": 15000,
              "blurb": "23 · freelance designer · irregular income, thin buffer"},
    "sharma": {"name": "Sharma family", "avatar": "S", "opening_balance": 250000,
               "blurb": "Household of 4 · home-loan EMI, school fees, rising electricity"},
    "me": {"name": "My data", "avatar": "M", "opening_balance": 0,
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
    db.set_setting("opening_balance", str(PROFILES[pid]["opening_balance"]))
    return len(ingest.ingest_csv(seed.read_text()))


def summary(pid: str) -> dict:
    return {"id": pid, **PROFILES[pid], "has_seed": (SEED_DIR / f"{pid}.csv").exists()}
