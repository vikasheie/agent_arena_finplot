"""Analyzer agent: aggregates behaviour over time and detects patterns + anomalies in the user's data.

Detection is deterministic statistics (so it is testable and measured in eval/); the LLM is used
only to turn the detected evidence into a short behavioural narrative. Every finding gets a stable
evidence id (e.g. "P3") that downstream agents must cite.
"""
import json
import statistics
from collections import defaultdict
from datetime import date

from finpilot import config, db, llm
from finpilot.agents.categorizer import merchant_key

AGENT = "analyzer"


def display_name(description: str) -> str:
    """Human-readable merchant name from raw bank text ("UPI/P2M/123/BESCOM ELECTRICITY" -> "Bescom Electricity")."""
    return merchant_key(description).title()


def _month(d: str) -> str:
    return d[:7]


def monthly_by_category(txs: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in txs:
        if t["amount"] < 0:
            out[_month(t["date"])][t["category"] or "other"] += -t["amount"]
    return {m: dict(v) for m, v in sorted(out.items())}


def monthly_totals(txs: list[dict]) -> list[dict]:
    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"income": 0.0, "spend": 0.0})
    for t in txs:
        key = "income" if t["amount"] > 0 and t["category"] == "income" else ("spend" if t["amount"] < 0 else None)
        if key:
            agg[_month(t["date"])][key] += abs(t["amount"])
    return [{"month": m, **{k: round(v, 2) for k, v in vals.items()}, "net": round(vals["income"] - vals["spend"], 2)}
            for m, vals in sorted(agg.items())]


BILL_CATEGORIES = {"utilities", "health", "education"}
PLANNED_LUMPY = {"travel", "education"}  # big but deliberate; never judged against day-to-day spending
CADENCES = {"monthly": (20, 40, 1), "quarterly": (80, 100, 3)}  # gap range in days, months per payment


def detect_recurring(txs: list[dict], min_occurrences: int = 3) -> list[dict]:
    """Same merchant, similar amount (+-15%), regular cadence: monthly (20-40 day gaps) or quarterly."""
    by_merchant: dict[str, list[dict]] = defaultdict(list)
    for t in txs:
        if t["amount"] < 0:
            by_merchant[merchant_key(t["description"])].append(t)
    found = []
    for key, items in by_merchant.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t["date"])
        amounts = [-t["amount"] for t in items]
        med = statistics.median(amounts)
        if med <= 0 or any(abs(a - med) / med > 0.15 for a in amounts[-min_occurrences:]):
            continue
        dates = [date.fromisoformat(t["date"]) for t in items]
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])][-(min_occurrences - 1):]
        cadence = next((name for name, (lo, hi, _) in CADENCES.items() if gaps and all(lo <= g <= hi for g in gaps)),
                       None)
        # Monthly needs min_occurrences payments; a quarterly bill is evident after two (one 80-100 day gap).
        if not cadence or (cadence == "monthly" and len(items) < min_occurrences):
            continue
        per = CADENCES[cadence][2]
        found.append({
            "merchant": key,
            "name": key.title(),
            "description": items[-1]["description"],
            "category": items[-1]["category"],
            "cadence": cadence,
            "amount": round(med, 2),
            "monthly_amount": round(med / per, 2),
            "occurrences": len(items),
            "last_date": items[-1]["date"],
            "tx_ids": [t["id"] for t in items],
        })
    return sorted(found, key=lambda r: -r["monthly_amount"])


def detect_anomalies(txs: list[dict], z_threshold: float = 3.5) -> list[dict]:
    """Outflows far above the user's own norm (robust z-score via median/MAD).

    The norm is the same merchant's history when there are >= 4 payments to it (a fuel fill-up is
    not an anomalous "transport" spend just because cab rides are cheap); otherwise the category.
    """
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_merchant: dict[str, list[float]] = defaultdict(list)
    scheduled = {i for r in detect_recurring(txs) for i in r["tx_ids"]}  # known bills are never anomalies
    outflows = [t for t in txs if t["amount"] < 0 and t["category"] not in {"rent", "emi", "transfers"}
                and t["id"] not in scheduled]
    for t in outflows:
        by_cat[t["category"] or "other"].append(-t["amount"])
        by_merchant[merchant_key(t["description"])].append(-t["amount"])
    all_spend = [-t["amount"] for t in outflows]
    anomalies = []
    for t in outflows:
        a, cat = -t["amount"], t["category"] or "other"
        merchant_hist = by_merchant[merchant_key(t["description"])]
        if len(merchant_hist) >= 4:
            ref, basis = merchant_hist, "merchant"
        elif len(by_cat[cat]) >= 4:
            ref, basis = by_cat[cat], "category"
        elif cat not in PLANNED_LUMPY:  # rare category (e.g. one hospital bill): vs. all day-to-day spending
            ref, basis = all_spend, "all_spending"
        else:
            continue
        med = statistics.median(ref)
        mad = statistics.median([abs(x - med) for x in ref]) or max(1.0, 0.1 * med)
        z = 0.6745 * (a - med) / mad
        if z >= z_threshold and a >= 2.5 * med and a >= 1000:
            anomalies.append({"tx_id": t["id"], "date": t["date"], "description": t["description"],
                              "name": display_name(t["description"]), "category": cat, "amount": round(a, 2), "typical": round(med, 2),
                              "basis": basis, "robust_z": round(z, 1)})
    return sorted(anomalies, key=lambda r: -r["robust_z"])


def category_trends(by_month: dict[str, dict[str, float]]) -> list[dict]:
    """Latest full month vs. average of the previous months, per category."""
    months = sorted(by_month)
    if len(months) < 3:
        return []
    latest, prior = months[-1], months[:-1]
    trends = []
    cats = {c for m in months for c in by_month[m]}
    for c in cats:
        base = statistics.mean(by_month[m].get(c, 0.0) for m in prior)
        cur = by_month[latest].get(c, 0.0)
        if base >= 1000 and cur > 1.25 * base:
            trends.append({"category": c, "month": latest, "amount": round(cur, 2), "baseline": round(base, 2),
                           "change_pct": round(100 * (cur - base) / base, 1)})
    return sorted(trends, key=lambda r: -r["change_pct"])


def merchant_trends(txs: list[dict], min_months: int = 4) -> list[dict]:
    """Bills paid every month whose amount keeps climbing (e.g. electricity as AC use grows)."""
    all_months = {_month(t["date"]) for t in txs}
    by_merchant: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    meta: dict[str, dict] = {}
    for t in txs:
        if t["amount"] < 0 and t["category"] in BILL_CATEGORIES:
            key = merchant_key(t["description"])
            by_merchant[key][_month(t["date"])] += -t["amount"]
            meta[key] = {"description": t["description"], "category": t["category"]}
    out = []
    for key, per_month in by_merchant.items():
        months = sorted(per_month)
        if len(months) < max(min_months, len(all_months) - 1):  # must be a bill paid (nearly) every month
            continue
        early = statistics.mean(per_month[m] for m in months[:3])
        latest = per_month[months[-1]]
        if early > 0 and latest >= 1.3 * early and latest - early >= 500:
            out.append({"merchant": key, **meta[key], "first_months_avg": round(early, 2),
                        "latest": round(latest, 2), "month": months[-1], "increase": round(latest - early, 2),
                        "change_pct": round(100 * (latest - early) / early, 1),
                        "series": {m: round(per_month[m], 2) for m in months}})
    return sorted(out, key=lambda r: -r["increase"])


def detect_impulse(txs: list[dict]) -> dict:
    """Discretionary, non-recurring purchases made within 3 days after a salary credit."""
    salary_days = [date.fromisoformat(t["date"]) for t in txs if t["category"] == "income" and t["amount"] > 0]
    recurring_ids = {i for r in detect_recurring(txs) for i in r["tx_ids"]}
    impulse = []
    for t in txs:
        if t["amount"] >= 0 or t["category"] not in {"shopping", "entertainment", "dining"} or t["id"] in recurring_ids:
            continue
        d = date.fromisoformat(t["date"])
        if any(0 <= (d - s).days <= 3 for s in salary_days) and -t["amount"] >= 1000:
            impulse.append(t)
    total = round(sum(-t["amount"] for t in impulse), 2)
    months = max(1, len({_month(t["date"]) for t in txs}))
    return {"count": len(impulse), "total": total, "monthly_avg": round(total / months, 2),
            "tx_ids": [t["id"] for t in impulse][:50]}


def top_discretionary_merchants(txs: list[dict], limit: int = 3, floor: float = 1500) -> list[dict]:
    """Biggest non-recurring discretionary merchants by average monthly spend."""
    skip = {i for r in detect_recurring(txs) for i in r["tx_ids"]} | {a["tx_id"] for a in detect_anomalies(txs)}
    months = max(1, len({_month(t["date"]) for t in txs}))
    agg: dict[str, dict] = {}
    for t in txs:
        if t["amount"] >= 0 or t["category"] not in config.DISCRETIONARY or t["id"] in skip:
            continue
        m = agg.setdefault(merchant_key(t["description"]), {"total": 0.0, "count": 0, "category": t["category"]})
        m["total"] += -t["amount"]
        m["count"] += 1
    ranked = sorted(agg.items(), key=lambda kv: -kv[1]["total"])
    return [{"merchant": k, "category": v["category"], "count": v["count"],
             "monthly_avg": round(v["total"] / months, 2)} for k, v in ranked[:limit] if v["total"] / months >= floor]


def analyze(run_id: int | None = None) -> dict:
    txs = [t for t in db.all_transactions() if t["category"]]
    if not txs:
        return {"patterns": [], "monthly": [], "by_category": {}, "narrative": "No transactions yet."}

    by_month = monthly_by_category(txs)
    patterns: list[dict] = []

    def add(kind: str, summary: str, data: dict) -> None:
        patterns.append({"id": f"P{len(patterns) + 1}", "kind": kind, "summary": summary, "data": data})

    for r in detect_recurring(txs):
        add("recurring", f"{r['name']} recurs {r['cadence']} at ~INR {r['amount']:,.0f} "
                         f"(INR {r['monthly_amount']:,.0f}/month, {r['occurrences']} times, category {r['category']})", r)
    for a in detect_anomalies(txs)[:8]:
        add("anomaly", f"{a['name']} on {a['date']}: INR {a['amount']:,.0f} vs typical "
                       f"INR {a['typical']:,.0f} for {a['category']}", a)
    for mt in merchant_trends(txs):
        add("bill_trend", f"{mt['merchant'].title()} ({mt['category']}) rose from ~INR {mt['first_months_avg']:,.0f} "
                          f"to INR {mt['latest']:,.0f} in {mt['month']} (+{mt['change_pct']}%)", mt)
    # Trends describe habits: leave out scheduled bills and one-off anomalies (already reported above).
    skip = {i for r in detect_recurring(txs) for i in r["tx_ids"]} | {a["tx_id"] for a in detect_anomalies(txs)}
    for tr in category_trends(monthly_by_category([t for t in txs if t["id"] not in skip])):
        add("trend", f"{tr['category']} spend in {tr['month']} was INR {tr['amount']:,.0f}, "
                     f"{tr['change_pct']}% above the {tr['baseline']:,.0f} baseline", tr)
    imp = detect_impulse(txs)
    if imp["count"]:
        add("impulse", f"{imp['count']} discretionary purchases within 3 days of payday, "
                       f"~INR {imp['monthly_avg']:,.0f}/month", imp)

    for m in top_discretionary_merchants(txs):
        add("top_merchant", f"{m['merchant'].title()} ({m['category']}) takes INR {m['monthly_avg']:,.0f}/month "
                            f"across {m['count']} payments", m)

    months = sorted(by_month)
    avg_by_cat ={c: round(sum(by_month[m].get(c, 0) for m in months) / len(months), 2)
                  for c in {c for m in months for c in by_month[m]}}
    discretionary = round(sum(v for c, v in avg_by_cat.items() if c in config.DISCRETIONARY), 2)
    add("summary", f"Average monthly discretionary spend INR {discretionary:,.0f}",
        {"avg_by_category": avg_by_cat, "discretionary_monthly": discretionary})

    narrative = ""
    if llm.available():
        try:
            out = llm.chat_json(
                "You are the Analyzer agent of FinPilot. Given detected spending patterns (with ids), write a "
                "3-sentence behavioural summary of this user. Cite pattern ids in brackets like [P2]. "
                'Only use facts present in the input. Respond as JSON {"narrative": "..."}',
                json.dumps([{k: p[k] for k in ("id", "kind", "summary")} for p in patterns]),
                agent=AGENT, run_id=run_id,
            )
            narrative = out.get("narrative", "")
        except Exception as e:
            narrative = f"(narrative unavailable: {e})"

    result = {"patterns": patterns, "monthly": monthly_totals(txs), "by_category": by_month, "narrative": narrative}
    db.log_decision(run_id, AGENT, "analyze", {
        "transactions": len(txs), "pattern_ids": [p["id"] for p in patterns],
        "kinds": {k: sum(1 for p in patterns if p["kind"] == k) for k in {p["kind"] for p in patterns}},
    })
    return result
