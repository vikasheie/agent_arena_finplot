"""Transaction ingestion: CSV upload (several common bank layouts) or manual entry."""
import csv
import io
from datetime import datetime

from finpilot import db

DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%d-%b-%Y", "%Y/%m/%d")


def parse_date(value: str) -> str:
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date '{value}'")


def _num(value: str | None) -> float:
    if value is None:
        return 0.0
    v = value.replace(",", "").replace("₹", "").replace("INR", "").strip()
    return float(v) if v else 0.0


def parse_csv(text: str) -> list[dict]:
    """Accepts columns date/description/amount (signed), or date/description/debit/credit."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames:
        raise ValueError("empty CSV")
    cols = {c.strip().lower(): c for c in reader.fieldnames}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)  # noqa: E731
    c_date = pick("date", "transaction date", "txn date", "value date")
    c_desc = pick("description", "narration", "details", "particulars", "merchant")
    c_amt = pick("amount", "amount (inr)")
    c_debit = pick("debit", "withdrawal", "withdrawal amt.", "dr")
    c_credit = pick("credit", "deposit", "deposit amt.", "cr")
    if not c_date or not c_desc or not (c_amt or c_debit or c_credit):
        raise ValueError(f"CSV needs date, description and amount (or debit/credit) columns; got {reader.fieldnames}")

    rows = []
    for i, r in enumerate(reader, start=2):
        if not (r.get(c_date) or "").strip():
            continue
        try:
            amount = _num(r[c_amt]) if c_amt else _num(r.get(c_credit)) - _num(r.get(c_debit))
            rows.append({"date": parse_date(r[c_date]), "description": r[c_desc].strip(), "amount": amount})
        except (ValueError, KeyError) as e:
            raise ValueError(f"line {i}: {e}") from e
    return rows


def ingest_csv(text: str) -> list[int]:
    return db.insert_transactions(parse_csv(text))


def ingest_manual(date_str: str, description: str, amount: float, kind: str = "expense") -> int:
    amt = abs(float(amount)) * (1 if kind == "income" else -1)
    return db.insert_transactions([{"date": parse_date(date_str), "description": description, "amount": amt}])[0]
