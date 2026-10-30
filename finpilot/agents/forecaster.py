"""Forecaster agent: projects the user's cash-flow trajectory and simulates what-if decisions.

The projection is a transparent, backtestable model built only from the user's history:
  income        = median monthly income of the last 3 months
  fixed outflow = detected recurring payments (rent, subscriptions, EMIs ...)
  variable      = recency-weighted mean of the remaining spend per category (last 3 months)
  band          = +-1.28 sigma of historical monthly variable spend (~80% interval)
`backtest()` replays the model on rolling origins so its error is measured, not claimed.
"""
import statistics
from collections import defaultdict
from datetime import date

from finpilot import config, db
from finpilot.agents.analyzer import detect_anomalies, detect_recurring

AGENT = "forecaster"
WEIGHTS = [0.5, 0.3, 0.2]  # most recent month first
ESSENTIAL = {"rent", "emi", "groceries", "utilities", "transport", "health", "education"}


def _add_months(ym: str, n: int) -> str:
    y, m = map(int, ym.split("-"))
    m += n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f"{y:04d}-{m:02d}"


def _full_months(txs: list[dict]) -> list[str]:
    months = sorted({t["date"][:7] for t in txs})
    if not months:
        return []
    last_day = max(t["date"] for t in txs)
    # Drop the latest month if it is clearly partial (data ends before the 25th).
    if int(last_day[8:10]) < 25 and len(months) > 1:
        months = months[:-1]
    return months


def project(txs: list[dict], horizon: int = 6, opening_balance: float = 0.0) -> dict:
    txs = [t for t in txs if t.get("category")]
    months = _full_months(txs)
    if len(months) < 2:
        return {"error": "need at least 2 full months of transactions to forecast"}
    recent = months[-3:]
    in_scope = [t for t in txs if t["date"][:7] in months]

    income_by_m = defaultdict(float)
    for t in in_scope:
        if t["amount"] > 0 and t["category"] == "income":
            income_by_m[t["date"][:7]] += t["amount"]
    income = statistics.median([income_by_m.get(m, 0.0) for m in recent])

    recurring = detect_recurring(in_scope)
    recurring_ids = {i for r in recurring for i in r["tx_ids"]}
    # One-off anomalies (a laptop, a hospital bill) are not expected to repeat: keep them out of the baseline.
    one_off_ids = {a["tx_id"] for a in detect_anomalies(in_scope)}
    # Only recurring items still active at the end of the data are projected forward.
    end = date.fromisoformat(max(t["date"] for t in in_scope))
    active = [r for r in recurring
              if (end - date.fromisoformat(r["last_date"])).days <= (45 if r["cadence"] == "monthly" else 100)]
    monthly_items = [r for r in active if r["cadence"] == "monthly"]
    quarterly_items = [r for r in active if r["cadence"] == "quarterly"]
    fixed_monthly = sum(r["monthly_amount"] for r in monthly_items)
    fixed = fixed_monthly + sum(r["monthly_amount"] for r in quarterly_items)  # average monthly equivalent

    def scheduled_for(month: str) -> float:
        """Quarterly bills land in full in the month they fall due, not smeared across months."""
        due = 0.0
        for r in quarterly_items:
            k = 1
            while _add_months(r["last_date"][:7], 3 * k) < month:
                k += 1
            if _add_months(r["last_date"][:7], 3 * k) == month:
                due += r["amount"]
        return due

    var_by_m_cat: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in in_scope:
        if t["amount"] < 0 and t["id"] not in recurring_ids | one_off_ids:
            var_by_m_cat[t["date"][:7]][t["category"]] += -t["amount"]
    cats = {c for m in months for c in var_by_m_cat[m]}
    w = WEIGHTS[:len(recent)]
    variable_by_cat = {
        c: round(sum(wi * var_by_m_cat[m].get(c, 0.0) for wi, m in zip(w, reversed(recent))) / sum(w), 2)
        for c in cats
    }
    variable = sum(variable_by_cat.values())
    var_hist = [sum(var_by_m_cat[m].values()) for m in months]
    sigma = statistics.pstdev(var_hist) if len(var_hist) > 1 else 0.0

    balance = opening_balance + sum(t["amount"] for t in txs)
    # If the data ends part-way through the next month, only the remainder of that month is still ahead.
    last_day = max(t["date"] for t in txs)
    first_fraction = 1.0
    if last_day[:7] == _add_months(months[-1], 1):
        first_fraction = max(0.0, 1 - int(last_day[8:10]) / 30)
    path, bal = [], balance
    for i in range(1, horizon + 1):
        month = _add_months(months[-1], i)
        fixed_m = fixed_monthly + scheduled_for(month)
        net = (income - fixed_m - variable) * (first_fraction if i == 1 else 1.0)
        bal += net
        path.append({"month": month, "income": round(income, 2), "fixed": round(fixed_m, 2),
                     "variable": round(variable, 2), "net": round(net, 2), "balance": round(bal, 2),
                     "low": round(bal - 1.28 * sigma * (i ** 0.5), 2),
                     "high": round(bal + 1.28 * sigma * (i ** 0.5), 2)})

    essential = sum(r["monthly_amount"] for r in active if r["category"] in ESSENTIAL) + \
        sum(v for c, v in variable_by_cat.items() if c in ESSENTIAL)
    return {
        "id": "F1",
        "based_on_months": months,
        "current_balance": round(balance, 2),
        "monthly_income": round(income, 2),
        "monthly_fixed": round(fixed, 2),
        "monthly_variable": round(variable, 2),
        "variable_by_category": variable_by_cat,
        "fixed_items": [{"description": r["description"], "amount": r["amount"], "cadence": r["cadence"],
                         "monthly_amount": r["monthly_amount"], "category": r["category"]} for r in active],
        "monthly_net": round(income - fixed - variable, 2),
        "safety_buffer": round(essential, 2),
        "sigma_variable": round(sigma, 2),
        "path": path,
    }


def backtest(txs: list[dict], min_train_months: int = 3) -> dict:
    """Rolling-origin backtest: train on months[:k], predict month k's total spend and net."""
    txs = [t for t in txs if t.get("category")]
    months = _full_months(txs)
    rows = []
    for k in range(min_train_months, len(months)):
        train = [t for t in txs if t["date"][:7] < months[k]]
        fc = project(train, horizon=1)
        if "error" in fc:
            continue
        # All outflows (incl. SIPs / transfers): that is what moves the balance being forecast.
        actual_spend = sum(-t["amount"] for t in txs if t["date"][:7] == months[k] and t["amount"] < 0)
        actual_income = sum(t["amount"] for t in txs
                            if t["date"][:7] == months[k] and t["amount"] > 0 and t["category"] == "income")
        pred_spend = fc["path"][0]["fixed"] + fc["monthly_variable"]
        rows.append({"month": months[k], "predicted_spend": round(pred_spend, 2), "actual_spend": round(actual_spend, 2),
                     "abs_pct_error": round(100 * abs(pred_spend - actual_spend) / max(actual_spend, 1), 2),
                     "predicted_net": fc["path"][0]["net"], "actual_net": round(actual_income - actual_spend, 2)})
    if not rows:
        return {"error": "not enough history to backtest", "rows": []}
    return {"rows": rows, "mape_spend_pct": round(statistics.mean(r["abs_pct_error"] for r in rows), 2),
            "mae_net": round(statistics.mean(abs(r["predicted_net"] - r["actual_net"]) for r in rows), 2)}


def earliest_purchase(txs: list[dict], amount: float, opening_balance: float = 0.0, horizon: int = 24) -> dict:
    """When can the user make a one-off purchase? Try it in each future month of a long projection.

    affordable = the pessimistic path never drops below the safety buffer; tight = the balance stays >= 0.
    """
    fc = project(txs, horizon=horizon, opening_balance=opening_balance)
    if "error" in fc:
        return {"error": fc["error"]}
    found: dict = {"horizon_months": horizon, "affordable": None, "tight": None}
    for offset in range(1, horizon + 1):
        sim = simulate(fc, {"type": "purchase", "amount": amount, "month_offset": offset})
        hit = {"month": fc["path"][offset - 1]["month"], "month_offset": offset, "min_balance": sim["min_balance"]}
        if found["tight"] is None and sim["verdict"] in ("tight", "affordable"):
            found["tight"] = dict(hit)  # earliest month it fits at all; no savings plan implied
        if sim["verdict"] == "affordable":
            hit["set_aside_per_month"] = round(amount / offset, 2)  # saving towards it until the comfortable month
        if sim["verdict"] == "affordable":
            found["affordable"] = hit
            break
    return found


def simulate(fc: dict, decision: dict) -> dict:
    """Apply a what-if decision to the baseline trajectory.

    decision = {"type": "purchase", "amount": 40000, "month_offset": 1}
             | {"type": "recurring_change", "monthly_delta": -1500, "month_offset": 1}  (negative = cut cost)
             | {"type": "savings_goal", "amount": 100000, "months": 6}
    """
    path = fc["path"]
    buffer = fc["safety_buffer"]
    kind = decision.get("type")
    offset = max(1, int(decision.get("month_offset", 1)))
    scenario, extra = [], 0.0
    for i, p in enumerate(path, start=1):
        if kind == "purchase" and i == offset:
            extra -= float(decision["amount"])
        elif kind == "recurring_change" and i >= offset:
            extra -= float(decision["monthly_delta"])
        elif kind == "savings_goal" and i <= int(decision.get("months", len(path))):
            extra -= float(decision["amount"]) / int(decision.get("months", len(path)))
        scenario.append({"month": p["month"], "baseline": p["balance"], "scenario": round(p["balance"] + extra, 2),
                         "scenario_low": round(p["low"] + extra, 2)})

    min_bal = min(s["scenario"] for s in scenario)
    min_low = min(s["scenario_low"] for s in scenario)
    if min_low >= buffer:
        verdict = "affordable"
    elif min_bal >= 0:
        verdict = "tight"
    else:
        verdict = "not_affordable"
    shortfall = round(max(0.0, buffer - min_bal), 2)

    # Which discretionary categories would cover the shortfall over the horizon?
    cuts, remaining = [], shortfall
    months_avail = max(1, offset if kind == "purchase" else len(path))
    for c, v in sorted(fc["variable_by_category"].items(), key=lambda kv: -kv[1]):
        if remaining <= 0 or c not in config.DISCRETIONARY:
            continue
        cut = min(v * 0.3, remaining / months_avail)
        if cut > 0:
            cuts.append({"category": c, "cut_per_month": round(cut, 2), "current_per_month": v})
            remaining -= cut * months_avail
    return {"decision": decision, "verdict": verdict, "min_balance": round(min_bal, 2),
            "min_balance_pessimistic": round(min_low, 2), "safety_buffer": buffer, "shortfall": shortfall,
            "suggested_cuts": cuts, "path": scenario, "forecast_id": fc.get("id", "F1")}


def forecast(run_id: int | None = None, horizon: int = 6) -> dict:
    opening = float(db.get_setting("opening_balance", "0") or 0)
    fc = project(db.all_transactions(), horizon=horizon, opening_balance=opening)
    db.log_decision(run_id, AGENT, "forecast", {k: fc.get(k) for k in (
        "error", "based_on_months", "current_balance", "monthly_income", "monthly_fixed", "monthly_variable",
        "monthly_net", "safety_buffer")})
    return fc


def summary_facts(fc: dict, goal_status: list[dict]) -> dict:
    """Computed facts behind the forecast sentence (the LLM may only rephrase these)."""
    end = fc["path"][-1]
    facts = {"monthly_net": round(fc["monthly_net"]), "balance_now": round(fc["current_balance"]),
             "end_month": end["month"], "end_balance": round(end["balance"]), "safety_buffer": round(fc["safety_buffer"])}
    sentence = (f"At this pace you add INR {fc['monthly_net']:,.0f} a month and reach INR {end['balance']:,.0f} "
                f"by {end['month']}") if fc["monthly_net"] >= 0 else (
                f"At this pace you lose INR {-fc['monthly_net']:,.0f} a month, falling to INR {end['balance']:,.0f} "
                f"by {end['month']}")
    if goal_status:
        g = goal_status[0]
        facts["goal"] = {k: g[k] for k in ("name", "status", "needed_per_month", "gap_per_month", "months_left")}
        sentence += (f"; “{g['name']}” is on track" if g["status"] == "on_track"
                     else f"; “{g['name']}” needs INR {g['gap_per_month']:,.0f}/month more to land on time")
    facts["sentence"] = sentence + "."
    return facts


def goal_progress(fc: dict, goals: list[dict], today: date | None = None) -> list[dict]:
    """Is each goal on track given the projected monthly surplus?"""
    today = today or date.today()
    out = []
    for g in goals:
        td = date.fromisoformat(g["target_date"])
        months_left = max(1, (td.year - today.year) * 12 + td.month - today.month)
        needed = max(0.0, g["target_amount"] - g["saved_so_far"]) / months_left
        surplus = fc.get("monthly_net", 0.0)
        out.append({"goal_id": g["id"], "name": g["name"], "months_left": months_left,
                    "needed_per_month": round(needed, 2), "projected_surplus": surplus,
                    "status": "on_track" if surplus >= needed else "behind",
                    "gap_per_month": round(max(0.0, needed - surplus), 2)})
    return out
