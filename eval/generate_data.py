"""Generate a seeded, labelled synthetic transaction history for one persona (Apr-Oct 2026).

Writes:
  data/sample_transactions.csv   - what the user uploads (date, description, amount)  -> no labels
  eval/ground_truth.csv          - same rows + true category + injected-anomaly flag   -> for scoring
Bank-style noise (UPI refs, card prefixes, random ids) is added so categorisation is non-trivial.
Run: python -m eval.generate_data
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
START, END = date(2026, 4, 1), date(2026, 10, 28)


def _ref(rng: random.Random) -> str:
    return str(rng.randint(100000, 999999))


def _upi(rng, name: str) -> str:
    return rng.choice([f"UPI/{name}/{_ref(rng)}@ybl", f"UPI-{name}-{_ref(rng)}", f"UPI/P2M/{_ref(rng)}/{name}"])


def _card(rng, name: str) -> str:
    return rng.choice([f"POS {name} {_ref(rng)}", f"DEBIT CARD {name} BLR", f"{name}*{_ref(rng)}"])


def generate(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    def add(d: date, desc: str, amount: float, cat: str, anomaly: bool = False):
        if START <= d <= END:
            rows.append({"date": d.isoformat(), "description": desc, "amount": round(amount, 2),
                         "category": cat, "is_anomaly": int(anomaly)})

    month_starts = [date(2026, m, 1) for m in range(4, 11)]
    for mi, ms in enumerate(month_starts):
        add(ms, "NEFT CR ACME TECHNOLOGIES SALARY", 85000, "income")
        add(ms + timedelta(days=2), "IMPS RENT TO LANDLORD R KUMAR", -22000, "rent")
        add(ms + timedelta(days=4), _card(rng, "NETFLIX.COM"), -649, "subscriptions")
        add(ms + timedelta(days=6), _card(rng, "SPOTIFY INDIA"), -119, "subscriptions")
        add(ms + timedelta(days=9), _upi(rng, "DISNEY HOTSTAR"), -299, "subscriptions")
        add(ms + timedelta(days=11), _upi(rng, "CULTFIT GYM MEMBERSHIP"), -1499, "subscriptions")
        add(ms + timedelta(days=14), _upi(rng, "AIRTEL BROADBAND"), -799, "utilities")
        add(ms + timedelta(days=17), _upi(rng, "BESCOM ELECTRICITY"), -rng.uniform(1200, 2100), "utilities")
        add(ms + timedelta(days=19), _upi(rng, "JIO PREPAID RECHARGE"), -299, "utilities")
        add(ms + timedelta(days=5), "ACH D ZERODHA MUTUAL FUND SIP", -10000, "transfers")
        add(ms + timedelta(days=7), _upi(rng, "TRANSFER TO MOM SAVINGS"), -5000, "transfers")
        if mi % 2 == 1:
            add(ms + timedelta(days=22), _upi(rng, "UPWORK FREELANCE PAYOUT"), rng.uniform(8000, 15000), "income")

        days_in_month = ((ms.replace(day=28) + timedelta(days=4)).replace(day=1) - ms).days
        dining_boost = 1.6 if ms.month == 10 else 1.0  # behaviour change in the latest month -> trend
        for day in range(days_in_month):
            d = ms + timedelta(days=day)
            if d.weekday() == 5:
                add(d, _upi(rng, rng.choice(["BIGBASKET", "BLINKIT", "DMART READY"])), -rng.uniform(1400, 2600), "groceries")
            if rng.random() < 0.18:
                add(d, _upi(rng, rng.choice(["ZEPTO", "KIRANA STORE RAMESH", "MORE RETAIL"])), -rng.uniform(150, 650), "groceries")
            if rng.random() < 0.32 * dining_boost:
                add(d, _upi(rng, rng.choice(["SWIGGY", "ZOMATO", "SWIGGY INSTAMART FOOD"]))
                    if rng.random() < 0.7 else _card(rng, rng.choice(["STARBUCKS", "THIRD WAVE COFFEE", "TRUFFLES CAFE"])),
                    -rng.uniform(220, 900), "dining")
            if rng.random() < 0.35:
                add(d, _upi(rng, rng.choice(["UBER INDIA", "RAPIDO BIKE", "OLA CABS", "NAMMA METRO RECHARGE"])),
                    -rng.uniform(80, 420), "transport")
            if d.day in (8, 23):
                add(d, _card(rng, rng.choice(["HP PETROL PUMP", "INDIAN OIL FUEL"])), -rng.uniform(1500, 2200), "transport")
            if rng.random() < 0.07:
                add(d, _card(rng, rng.choice(["AMAZON IN", "FLIPKART", "MYNTRA"])), -rng.uniform(400, 2500), "shopping")
            if rng.random() < 0.04:
                add(d, _upi(rng, rng.choice(["APOLLO PHARMACY", "MEDPLUS"])), -rng.uniform(150, 900), "health")
            if rng.random() < 0.04:
                add(d, _card(rng, rng.choice(["BOOKMYSHOW", "PVR CINEMAS"])), -rng.uniform(300, 1200), "entertainment")

        # Payday impulse shopping: discretionary, non-recurring, within 3 days of salary.
        for _ in range(rng.randint(1, 2)):
            add(ms + timedelta(days=rng.randint(0, 2)), _card(rng, rng.choice(["MYNTRA", "AMAZON IN", "NYKAA"])),
                -rng.uniform(1800, 4500), "shopping")

    # A trip and a few one-offs
    add(date(2026, 6, 12), _card(rng, "INDIGO AIRLINES"), -9800, "travel")
    add(date(2026, 6, 14), _card(rng, "OYO ROOMS GOA"), -6400, "travel")
    add(date(2026, 8, 20), "IMPS CR AMAZON REFUND", 1299, "shopping")

    # Injected anomalies (ground truth for anomaly-detection scoring)
    add(date(2026, 7, 18), _card(rng, "CROMA ELECTRONICS"), -34999, "shopping", anomaly=True)
    add(date(2026, 9, 6), _upi(rng, "ZOMATO"), -5200, "dining", anomaly=True)
    add(date(2026, 8, 27), _upi(rng, "UBER INDIA"), -3150, "transport", anomaly=True)
    add(date(2026, 10, 15), _upi(rng, "APOLLO PHARMACY"), -6800, "health", anomaly=True)

    rows.sort(key=lambda r: (r["date"], r["description"]))
    return rows


def main() -> None:
    rows = generate()
    (ROOT / "data").mkdir(exist_ok=True)
    with open(ROOT / "data" / "sample_transactions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "description", "amount"])
        w.writeheader()
        w.writerows({k: r[k] for k in ("date", "description", "amount")} for r in rows)
    with open(ROOT / "eval" / "ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "description", "amount", "category", "is_anomaly"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} transactions, {sum(r['is_anomaly'] for r in rows)} injected anomalies")


if __name__ == "__main__":
    main()
