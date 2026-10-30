"""Generate seeded, labelled synthetic transaction histories (Apr-Oct 2026) for three personas.

Writes, per persona id (priya | aarav | sharma):
  data/profiles/<id>.csv          - what the user uploads (date, description, amount) -> no labels
  eval/ground_truth/<id>.csv      - same rows + true category + injected-anomaly flag  -> for scoring
Bank-style noise (UPI refs, card prefixes, random ids) is added so categorisation is non-trivial.
Run: python -m eval.generate_data
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = date(2026, 4, 1), date(2026, 10, 28)
MONTHS = [date(2026, m, 1) for m in range(4, 11)]
PERSONAS = ("priya", "aarav", "sharma")


class Ledger:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.rows: list[dict] = []

    def ref(self) -> str:
        return str(self.rng.randint(100000, 999999))

    def upi(self, name: str) -> str:
        return self.rng.choice([f"UPI/{name}/{self.ref()}@ybl", f"UPI-{name}-{self.ref()}",
                                f"UPI/P2M/{self.ref()}/{name}"])

    def card(self, name: str) -> str:
        return self.rng.choice([f"POS {name} {self.ref()}", f"DEBIT CARD {name} BLR", f"{name}*{self.ref()}"])

    def add(self, d: date, desc: str, amount: float, cat: str, anomaly: bool = False) -> None:
        if START <= d <= END:
            self.rows.append({"date": d.isoformat(), "description": desc, "amount": round(amount, 2),
                              "category": cat, "is_anomaly": int(anomaly)})

    def daily(self, ms: date, fn) -> None:
        days = ((ms.replace(day=28) + timedelta(days=4)).replace(day=1) - ms).days
        for day in range(days):
            fn(ms + timedelta(days=day))

    def done(self) -> list[dict]:
        return sorted(self.rows, key=lambda r: (r["date"], r["description"]))


def priya() -> list[dict]:
    """28, software engineer in Bengaluru. Subscription creep, rising food delivery, payday impulse buys."""
    L = Ledger(42)
    r = L.rng
    for mi, ms in enumerate(MONTHS):
        L.add(ms, "NEFT CR ACME TECHNOLOGIES SALARY", 110000, "income")
        L.add(ms + timedelta(days=2), "IMPS RENT TO LANDLORD R KUMAR", -28000, "rent")
        L.add(ms + timedelta(days=4), L.card("NETFLIX.COM"), -649, "subscriptions")
        L.add(ms + timedelta(days=6), L.card("SPOTIFY INDIA"), -119, "subscriptions")
        L.add(ms + timedelta(days=9), L.upi("DISNEY HOTSTAR"), -299, "subscriptions")
        L.add(ms + timedelta(days=11), L.upi("CULTFIT GYM MEMBERSHIP"), -1499, "subscriptions")
        L.add(ms + timedelta(days=13), L.card("YOUTUBE PREMIUM GOOGLE"), -149, "subscriptions")
        if mi >= 3:  # new subscription picked up in July
            L.add(ms + timedelta(days=16), L.card("OPENAI CHATGPT SUBSCRIPTION"), -1999, "subscriptions")
        L.add(ms + timedelta(days=14), L.upi("AIRTEL BROADBAND"), -799, "utilities")
        L.add(ms + timedelta(days=17), L.upi("BESCOM ELECTRICITY"), -r.uniform(1100, 1800), "utilities")
        L.add(ms + timedelta(days=19), L.upi("JIO PREPAID RECHARGE"), -299, "utilities")
        L.add(ms + timedelta(days=5), "ACH D ZERODHA MUTUAL FUND SIP", -15000, "transfers")
        L.add(ms + timedelta(days=7), L.upi("TRANSFER TO MOM SAVINGS"), -5000, "transfers")
        dining_boost = 1.0 + 0.12 * mi  # food delivery creeping up every month

        def day(d: date) -> None:
            if d.weekday() == 5:
                L.add(d, L.upi(r.choice(["BIGBASKET", "BLINKIT", "DMART READY"])), -r.uniform(1400, 2600), "groceries")
            if r.random() < 0.15:
                L.add(d, L.upi(r.choice(["ZEPTO", "KIRANA STORE RAMESH"])), -r.uniform(150, 650), "groceries")
            if r.random() < 0.30 * dining_boost:
                L.add(d, L.upi(r.choice(["SWIGGY", "ZOMATO"])) if r.random() < 0.75
                      else L.card(r.choice(["STARBUCKS", "THIRD WAVE COFFEE"])), -r.uniform(250, 900), "dining")
            if r.random() < 0.35:
                L.add(d, L.upi(r.choice(["UBER INDIA", "RAPIDO BIKE", "OLA CABS", "NAMMA METRO RECHARGE"])),
                      -r.uniform(80, 420), "transport")
            if r.random() < 0.06:
                L.add(d, L.card(r.choice(["AMAZON IN", "FLIPKART", "MYNTRA"])), -r.uniform(400, 2500), "shopping")
            if r.random() < 0.04:
                L.add(d, L.upi(r.choice(["APOLLO PHARMACY", "MEDPLUS"])), -r.uniform(150, 900), "health")
            if r.random() < 0.05:
                L.add(d, L.card(r.choice(["BOOKMYSHOW", "PVR CINEMAS"])), -r.uniform(300, 1200), "entertainment")
        L.daily(ms, day)
        for _ in range(r.randint(1, 2)):  # payday impulse shopping
            L.add(ms + timedelta(days=r.randint(0, 2)), L.card(r.choice(["MYNTRA", "AMAZON IN", "NYKAA"])),
                  -r.uniform(1800, 4500), "shopping")
    L.add(date(2026, 6, 12), L.card("INDIGO AIRLINES"), -9800, "travel")
    L.add(date(2026, 6, 14), L.card("OYO ROOMS GOA"), -6400, "travel")
    L.add(date(2026, 8, 20), "IMPS CR AMAZON REFUND", 1299, "shopping")
    L.add(date(2026, 7, 18), L.card("CROMA ELECTRONICS"), -34999, "shopping", anomaly=True)
    L.add(date(2026, 9, 6), L.upi("ZOMATO"), -5200, "dining", anomaly=True)
    L.add(date(2026, 8, 27), L.upi("UBER INDIA"), -3150, "transport", anomaly=True)
    return L.done()


def aarav() -> list[dict]:
    """23, freelance designer. Irregular income, PG rent, gaming + food delivery, thin buffer."""
    L = Ledger(7)
    r = L.rng
    payouts = [42000, 28000, 61000, 35000, 26000, 48000, 30000]
    for mi, ms in enumerate(MONTHS):
        L.add(ms + timedelta(days=r.randint(3, 8)), L.upi("UPWORK FREELANCE PAYOUT"), payouts[mi] * 0.7, "income")
        L.add(ms + timedelta(days=r.randint(15, 22)), L.upi("FIVERR INTL PAYOUT"), payouts[mi] * 0.3, "income")
        L.add(ms + timedelta(days=1), L.upi("STANZA LIVING PG RENT"), -12000, "rent")
        L.add(ms + timedelta(days=3), L.card("SPOTIFY INDIA"), -119, "subscriptions")
        L.add(ms + timedelta(days=8), L.card("ADOBE CREATIVE CLOUD"), -1675, "subscriptions")
        L.add(ms + timedelta(days=10), L.card("XBOX GAME PASS"), -549, "subscriptions")
        L.add(ms + timedelta(days=12), L.card("NETFLIX.COM"), -199, "subscriptions")
        L.add(ms + timedelta(days=15), L.upi("JIO POSTPAID BILL"), -399, "utilities")

        def day(d: date) -> None:
            if r.random() < 0.45:
                L.add(d, L.upi(r.choice(["SWIGGY", "ZOMATO", "EATCLUB BOX8"])), -r.uniform(180, 600), "dining")
            if r.random() < 0.12:
                L.add(d, L.upi(r.choice(["BLINKIT", "ZEPTO"])), -r.uniform(200, 900), "groceries")
            if r.random() < 0.40:
                L.add(d, L.upi(r.choice(["RAPIDO BIKE", "NAMMA YATRI", "UBER INDIA"])), -r.uniform(60, 300), "transport")
            if r.random() < 0.06:
                L.add(d, L.card(r.choice(["STEAM GAMES", "PLAYSTATION STORE"])), -r.uniform(500, 2500), "entertainment")
            if r.random() < 0.05:
                L.add(d, L.card(r.choice(["AMAZON IN", "FLIPKART", "AJIO"])), -r.uniform(500, 2500), "shopping")
            if r.random() < 0.03:
                L.add(d, L.upi("MEDPLUS"), -r.uniform(100, 600), "health")
            if r.random() < 0.05:
                L.add(d, L.upi(r.choice(["TRANSFER TO ROHAN SPLIT", "TRANSFER TO KABIR SPLIT"])),
                      -r.uniform(300, 1500), "transfers")
        L.daily(ms, day)
    L.add(date(2026, 5, 24), L.card("DISTRICT EVENTS CONCERT TICKET"), -6500, "entertainment", anomaly=True)
    L.add(date(2026, 8, 9), L.card("RELIANCE DIGITAL ELECTRONICS"), -58990, "shopping", anomaly=True)
    L.add(date(2026, 10, 3), L.upi("SWIGGY"), -3400, "dining", anomaly=True)
    return L.done()


def sharma() -> list[dict]:
    """Household of four. Two incomes, home-loan EMI, school fees, parents' medicines, rising electricity."""
    L = Ledger(2026)
    r = L.rng
    electricity = [3100, 3350, 3600, 3900, 4300, 4750, 5250]  # AC usage climbing every month
    for mi, ms in enumerate(MONTHS):
        L.add(ms, "NEFT CR GLOBEX CORP PAYROLL", 95000, "income")
        L.add(ms + timedelta(days=1), "NEFT CR SUNRISE SCHOOL SALARY", 45000, "income")
        L.add(ms + timedelta(days=5), "ACH D HDFC HOME LOAN EMI", -38500, "emi")
        L.add(ms + timedelta(days=10), "ACH D BAJAJ FINANCE CAR LOAN EMI", -12400, "emi")
        L.add(ms + timedelta(days=18), L.upi("BESCOM ELECTRICITY"), -electricity[mi], "utilities")
        L.add(ms + timedelta(days=12), L.upi("ACT FIBERNET BROADBAND"), -1059, "utilities")
        L.add(ms + timedelta(days=14), L.upi("BWSSB WATER BILL"), -r.uniform(450, 700), "utilities")
        L.add(ms + timedelta(days=16), L.upi("INDANE GAS CYLINDER"), -905, "utilities")
        L.add(ms + timedelta(days=4), L.card("NETFLIX.COM"), -649, "subscriptions")
        L.add(ms + timedelta(days=9), L.card("AMAZON PRIME MEMBERSHIP"), -299, "subscriptions")
        L.add(ms + timedelta(days=11), L.upi("DISNEY HOTSTAR"), -299, "subscriptions")
        L.add(ms + timedelta(days=13), L.card("SONYLIV PREMIUM"), -299, "subscriptions")
        L.add(ms + timedelta(days=2), L.upi("SALARY TO MAID SUNITA"), -6000, "other")
        L.add(ms + timedelta(days=7), L.upi("APOLLO PHARMACY PARENTS MEDICINES"), -r.uniform(2600, 3200), "health")
        L.add(ms + timedelta(days=6), "ACH D LIC PREMIUM", -4200, "health")
        L.add(ms + timedelta(days=20), "ACH D SBI PPF DEPOSIT", -8000, "transfers")
        if ms.month in (4, 7, 10):
            L.add(ms + timedelta(days=8), "NEFT DPS SCHOOL FEES QTR", -42000, "education")
        L.add(ms + timedelta(days=15), L.upi("BYJUS TUITION"), -2500, "education")

        def day(d: date) -> None:
            if d.weekday() == 6:
                L.add(d, L.upi(r.choice(["DMART", "RELIANCE FRESH", "MORE RETAIL"])), -r.uniform(3200, 5200), "groceries")
            if r.random() < 0.3:
                L.add(d, L.upi(r.choice(["COUNTRY DELIGHT MILK", "KIRANA STORE RAMESH", "BLINKIT"])),
                      -r.uniform(150, 900), "groceries")
            if r.random() < 0.15:
                L.add(d, L.upi(r.choice(["SWIGGY", "ZOMATO", "DOMINOS PIZZA"])), -r.uniform(400, 1400), "dining")
            if d.day in (6, 21):
                L.add(d, L.card(r.choice(["BHARAT PETROLEUM", "INDIAN OIL FUEL"])), -r.uniform(2500, 3500), "transport")
            if r.random() < 0.06:
                L.add(d, L.card(r.choice(["AMAZON IN", "FLIPKART", "LIFESTYLE STORES"])), -r.uniform(600, 3500), "shopping")
            if r.random() < 0.03:
                L.add(d, L.card(r.choice(["PVR CINEMAS", "WONDERLA"])), -r.uniform(1200, 3000), "entertainment")
        L.daily(ms, day)
    L.add(date(2026, 5, 20), L.card("IRCTC ETICKET"), -8600, "travel")
    L.add(date(2026, 6, 22), "MANIPAL HOSPITAL IPD BILL", -48000, "health", anomaly=True)
    L.add(date(2026, 9, 14), L.card("MARUTI SERVICE CENTRE REPAIR"), -18500, "transport", anomaly=True)
    L.add(date(2026, 10, 19), L.card("TANISHQ JEWELLERS DIWALI"), -65000, "shopping", anomaly=True)
    return L.done()


GENERATORS = {"priya": priya, "aarav": aarav, "sharma": sharma}


def main() -> None:
    (ROOT / "data" / "profiles").mkdir(parents=True, exist_ok=True)
    (ROOT / "eval" / "ground_truth").mkdir(parents=True, exist_ok=True)
    for pid, gen in GENERATORS.items():
        rows = gen()
        with open(ROOT / "data" / "profiles" / f"{pid}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "description", "amount"])
            w.writeheader()
            w.writerows({k: r[k] for k in ("date", "description", "amount")} for r in rows)
        with open(ROOT / "eval" / "ground_truth" / f"{pid}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "description", "amount", "category", "is_anomaly"])
            w.writeheader()
            w.writerows(rows)
        print(f"{pid}: {len(rows)} transactions, {sum(r['is_anomaly'] for r in rows)} injected anomalies")


if __name__ == "__main__":
    main()
