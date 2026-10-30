"""Advisor agent + Verifier agent.

The advisor reasons over the analysis and forecast and must cite the evidence ids (P*, F1, G*)
that triggered each recommendation. The verifier checks every recommendation against the data:
cited evidence must exist, the claimed rupee impact must be supported by that evidence, the action
must name the specific merchant/category from the evidence, and generic tips are rejected.
Rejected items go back to the advisor with the verifier's notes (the self-correction loop).
"""
import json
import re

from finpilot import config, llm

ADVISOR = "advisor"
VERIFIER = "verifier"

# Spend FinPilot must never advise cutting (medicines, school fees, rent, loan EMIs, money sent to family).
ESSENTIAL_CATEGORIES = {"health", "education", "rent", "emi", "transfers"}

GENERIC = re.compile(r"\b(track your (spending|expenses)|create a budget|spend less|save more|be mindful|"
                     r"cut back on unnecessary|consider budgeting)\b", re.I)

ADVISOR_PROMPT = """You are the Advisor agent of FinPilot (Indian user, INR).
You receive detected spending patterns (ids P1..), a cash-flow forecast (id F1) and goal status (ids G1..).
Produce 3-5 specific, proactive recommendations. Rules:
- Every recommendation MUST cite the evidence ids it is based on in "evidence_ids".
- Name the exact merchant / subscription / category from the evidence (e.g. "Cancel Hotstar (INR 299/month)").
- "monthly_impact" is the INR per month the user gains by following it; it must be supported by the cited
  evidence amounts (never larger than what the evidence shows). Use 0 for one-off or alert-type advice.
- "reason" explains, with numbers, which pattern or forecast triggered it.
- No generic advice such as "create a budget" or "track your spending".
- Never suggest cutting essentials: education/tuition, medicines/health, insurance, rent, loan EMIs, or support
  sent to family. Target discretionary spend (dining, delivery, shopping, subscriptions, entertainment, travel)
  and bills that are rising (electricity, fuel).
- For bills that cannot be cancelled (electricity, fuel, groceries) give ONE concrete behavioural action
  (e.g. "Run the AC on a 26°C night-mode timer") with a saving no larger than the rise shown in the trend.
- Prioritise items that fix a negative forecast, a goal that is behind, or a sharp anomaly.
Respond as JSON: {"recommendations": [{"title": str, "action": str, "monthly_impact": number,
"evidence_ids": [str], "reason": str}]}"""


def _evidence_index(analysis: dict, fc: dict, goal_status: list[dict]) -> dict[str, dict]:
    idx = {p["id"]: p for p in analysis.get("patterns", [])}
    if "error" not in fc:
        idx["F1"] = {"id": "F1", "kind": "forecast", "data": fc}
    for i, g in enumerate(goal_status, start=1):
        idx[f"G{i}"] = {"id": f"G{i}", "kind": "goal", "data": g}
    return idx


def _max_supported_impact(ev: dict) -> float:
    d = ev["data"]
    return {
        "recurring": lambda: d["monthly_amount"],
        "top_merchant": lambda: d["monthly_avg"],
        "impulse": lambda: d["monthly_avg"],
        "trend": lambda: d["amount"] - d["baseline"],
        "bill_trend": lambda: d["increase"],
        "anomaly": lambda: d["amount"],
        "summary": lambda: d["discretionary_monthly"] * 0.5,
        "forecast": lambda: max(d.get("monthly_variable", 0) * 0.3, 0),
        "goal": lambda: d.get("gap_per_month", 0),
    }.get(ev["kind"], lambda: 0)()


def _names_in(ev: dict) -> set[str]:
    d = ev["data"]
    names = {str(d.get(k, "")).lower() for k in ("merchant", "category", "description", "name") if d.get(k)}
    if ev["kind"] == "summary":
        names |= set(d.get("avg_by_category", {}))
    if ev["kind"] == "forecast":
        names |= set(d.get("variable_by_category", {})) | {f["description"].lower() for f in d.get("fixed_items", [])}
    names |= {w for n in list(names) for w in n.split() if len(w) > 3}
    return {n for n in names if n}


def verify(recs: list[dict], evidence: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic checks of each recommendation against the evidence it cites."""
    ok, rejected = [], []
    for r in recs:
        problems = []
        ids = r.get("evidence_ids") or []
        missing = [i for i in ids if i not in evidence]
        if not ids:
            problems.append("cites no evidence")
        if missing:
            problems.append(f"cites unknown evidence {missing}")
        cited = [evidence[i] for i in ids if i in evidence]
        try:
            impact = float(r.get("monthly_impact") or 0)
        except (TypeError, ValueError):
            impact = -1
            problems.append("monthly_impact is not a number")
        cap = sum(_max_supported_impact(e) for e in cited)
        if 0 < impact < 100 and cap >= 1000:  # e.g. "+INR 3/month" on a ₹6,500/month habit: a slip, not advice
            problems.append(f"claims only INR {impact:,.0f}/month although the evidence involves up to "
                            f"INR {cap:,.0f}/month; state the realistic monthly saving")
        if cited and impact > cap * 1.05 + 1:
            problems.append(f"claims INR {impact:,.0f}/month but cited evidence supports at most INR {cap:,.0f}")
        text = f"{r.get('title', '')} {r.get('action', '')}".lower()
        names = set().union(*[_names_in(e) for e in cited]) if cited else set()
        if cited and not any(n in text for n in names):
            problems.append("action does not name the specific merchant/category from its evidence")
        if GENERIC.search(text) and not names:
            problems.append("generic advice")
        cut_essential = [e["id"] for e in cited if e["kind"] in {"recurring", "top_merchant", "anomaly"}
                         and e["data"].get("category") in ESSENTIAL_CATEGORIES]
        if cut_essential and impact > 0:
            problems.append(f"asks the user to cut essential spend ({', '.join(cut_essential)}); target discretionary "
                            "spend or rising bills instead")
        if problems:
            rejected.append({**r, "verified": False, "verifier_note": "; ".join(problems)})
        else:
            ok.append({**r, "verified": True,
                       "verifier_note": f"evidence {ids} supports up to INR {cap:,.0f}/month"})
    return ok, rejected


def advise(analysis: dict, fc: dict, goal_status: list[dict], feedback: list[dict] | None = None,
           accepted: list[dict] | None = None, dismissed: list[dict] | None = None,
           run_id: int | None = None) -> list[dict]:
    evidence = _evidence_index(analysis, fc, goal_status)
    compact = {
        "patterns": [{k: p[k] for k in ("id", "kind", "summary")} for p in analysis.get("patterns", [])],
        "forecast": None if "error" in fc else {
            "id": "F1", "current_balance": fc["current_balance"], "monthly_income": fc["monthly_income"],
            "monthly_fixed": fc["monthly_fixed"], "monthly_variable": fc["monthly_variable"],
            "monthly_net": fc["monthly_net"], "safety_buffer": fc["safety_buffer"],
            "balance_in_6_months": fc["path"][-1]["balance"], "variable_by_category": fc["variable_by_category"]},
        "goals": [{"id": k, **v["data"]} for k, v in evidence.items() if k.startswith("G")],
    }
    user = json.dumps(compact)
    if feedback:
        user += ("\n\nThe Verifier rejected these recommendations; fix or replace them, keeping amounts within "
                 "what the evidence supports:\n" + json.dumps(
                     [{k: r.get(k) for k in ("title", "monthly_impact", "evidence_ids", "verifier_note")}
                      for r in feedback]))
    if accepted:
        user += "\n\nAlready accepted (do not repeat): " + json.dumps([r["title"] for r in accepted])
    if dismissed:
        user += ("\n\nThe user switched these off earlier; do not suggest them again: "
                 + json.dumps([r["title"] for r in dismissed]))
    out = llm.chat_json(ADVISOR_PROMPT, user, agent=ADVISOR, run_id=run_id, model=config.MODEL_SMART,
                        temperature=0.2)
    recs = out.get("recommendations", [])
    return [r for r in recs if isinstance(r, dict) and r.get("title")]
