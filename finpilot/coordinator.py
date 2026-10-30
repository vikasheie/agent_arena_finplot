"""Coordinator: orchestrates the agents with an explicit LangGraph state machine.

Pipeline graph:
    coordinator -> categorize -> analyze -> forecast -> advise -> verify --(rejections & budget left)--> advise
                                                                         +--> finalize
Question flow: the coordinator classifies the user's question and routes it to the forecaster's
what-if simulator (affordability / decisions) or the Q&A agent (free-form questions over real data).
"""
import json
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from finpilot import config, db, llm
from finpilot.agents import advisor, analyzer, categorizer, forecaster

MAX_ADVICE_ROUNDS = 3


class PipelineState(TypedDict, total=False):
    run_id: int
    plan: list[str]
    categorization: dict
    analysis: dict
    forecast: dict
    goal_status: list[dict]
    drafts: list[dict]
    accepted: list[dict]
    rejected: list[dict]
    rejected_history: list[dict]
    advice_round: int
    recommendations: list[dict]


def _coordinator(state: PipelineState) -> PipelineState:
    pending = len(db.uncategorized())
    plan = (["categorize"] if pending else []) + ["analyze", "forecast"] + (["advise"] if llm.available() else [])
    db.log_decision(state["run_id"], "coordinator", "plan", {
        "uncategorized": pending, "plan": plan, "llm_available": llm.available()})
    return {"plan": plan, "advice_round": 0, "accepted": [], "rejected_history": []}


def _categorize(state: PipelineState) -> PipelineState:
    return {"categorization": categorizer.categorize_pending(state["run_id"])}


def _analyze(state: PipelineState) -> PipelineState:
    return {"analysis": analyzer.analyze(state["run_id"])}


def _forecast(state: PipelineState) -> PipelineState:
    fc = forecaster.forecast(state["run_id"])
    gs = forecaster.goal_progress(fc, db.goals()) if "error" not in fc else []
    if gs:
        db.log_decision(state["run_id"], "forecaster", "goal_progress", {"goals": gs})
    return {"forecast": fc, "goal_status": gs}


def _advise(state: PipelineState) -> PipelineState:
    rnd = state.get("advice_round", 0) + 1
    try:
        drafts = advisor.advise(state["analysis"], state["forecast"], state.get("goal_status", []),
                                feedback=state.get("rejected"), accepted=state.get("accepted"),
                                run_id=state["run_id"])
    except Exception as e:  # Mesh outage: stop advising, keep what is already verified
        db.log_decision(state["run_id"], "advisor", "error", {"round": rnd, "error": str(e)})
        return {"drafts": [], "advice_round": MAX_ADVICE_ROUNDS}
    db.log_decision(state["run_id"], "advisor", "draft", {"round": rnd, "titles": [d["title"] for d in drafts]},
                    model=config.MODEL_SMART)
    return {"drafts": drafts, "advice_round": rnd}


def _verify(state: PipelineState) -> PipelineState:
    evidence = advisor._evidence_index(state["analysis"], state["forecast"], state.get("goal_status", []))
    ok, bad = advisor.verify(state.get("drafts", []), evidence)
    accepted = state.get("accepted", []) + ok
    db.log_decision(state["run_id"], "verifier", "verify", {
        "round": state["advice_round"],
        "accepted": [r["title"] for r in ok],
        "rejected": [{"title": r["title"], "why": r["verifier_note"]} for r in bad]})
    return {"accepted": accepted, "rejected": bad, "rejected_history": state.get("rejected_history", []) + bad}


def _finalize(state: PipelineState) -> PipelineState:
    seen, recs = set(), []
    for r in state.get("accepted", []):
        if r["title"].lower() not in seen:
            seen.add(r["title"].lower())
            recs.append(r)
    db.save_recommendations(state["run_id"], recs)
    db.log_decision(state["run_id"], "coordinator", "finalize", {
        "recommendations": len(recs), "advice_rounds": state.get("advice_round", 0),
        "rejected_total": len(state.get("rejected_history", []))})
    return {"recommendations": recs}


def _after_forecast(state: PipelineState) -> str:
    return "advise" if "advise" in state["plan"] and "error" not in state["forecast"] else "finalize"


def _after_verify(state: PipelineState) -> str:
    if state.get("rejected") and state["advice_round"] < MAX_ADVICE_ROUNDS:
        return "advise"
    return "finalize"


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("coordinator", _coordinator)
    g.add_node("categorize", _categorize)
    g.add_node("analyze", _analyze)
    g.add_node("forecast", _forecast)
    g.add_node("advise", _advise)
    g.add_node("verify", _verify)
    g.add_node("finalize", _finalize)
    g.add_edge(START, "coordinator")
    g.add_conditional_edges("coordinator", lambda s: s["plan"][0], {"categorize": "categorize", "analyze": "analyze"})
    g.add_edge("categorize", "analyze")
    g.add_edge("analyze", "forecast")
    g.add_conditional_edges("forecast", _after_forecast, {"advise": "advise", "finalize": "finalize"})
    g.add_edge("advise", "verify")
    g.add_conditional_edges("verify", _after_verify, {"advise": "advise", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def run_pipeline() -> dict:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    run_id = db.start_run("pipeline")
    state = _GRAPH.invoke({"run_id": run_id})
    keep = {k: state.get(k) for k in ("run_id", "categorization", "analysis", "forecast", "goal_status",
                                      "recommendations", "rejected_history", "advice_round")}
    db.finish_run(run_id, keep)
    return keep


def _latest_state() -> dict:
    latest = db.latest_run("pipeline")
    if latest and not db.uncategorized():
        return latest["state"]
    return run_pipeline()


ROUTER_PROMPT = """You are the Coordinator of FinPilot. Classify the user's message and extract parameters.
Today is {today}. Forecast months available: {months}.
Return JSON: {{"intent": "what_if" | "question",
  "decision": null | {{"type": "purchase", "amount": number, "month_offset": int}}
                   | {{"type": "recurring_change", "monthly_delta": number, "month_offset": int}}
                   | {{"type": "savings_goal", "amount": number, "months": int}}}}
Use "what_if" for affordability or planned decisions ("can I afford", "what if I buy", "if I cancel X",
"save 1 lakh by"). month_offset 1 = next month. monthly_delta is positive for a new cost, negative for a cut.
Amounts in INR (1 lakh = 100000, 1k = 1000)."""

EXPLAIN_PROMPT = """You are FinPilot answering the user's money question using ONLY the provided numbers.
Be direct (start with Yes / No / Tight), quote rupee figures, mention the lowest projected balance and the
safety buffer, and if relevant name the specific categories to cut from "suggested_cuts".
Respond as JSON {"answer": "..."} (max 120 words)."""

QA_PROMPT = """You are FinPilot's Q&A agent. Answer the user's question using ONLY the data provided
(their real categorised spending, recurring payments, patterns and forecast). Quote figures in INR.
If the data cannot answer it, say so. Respond as JSON {"answer": "...", "evidence_ids": [..]} (max 150 words)."""


def ask(question: str) -> dict:
    if not llm.available():
        raise llm.LLMUnavailable("Questions need MESH_API_KEY")
    state = _latest_state()
    fc = state["forecast"]
    run_id = db.start_run("question")
    route = llm.chat_json(ROUTER_PROMPT.format(today=date.today().isoformat(),
                                               months=[p["month"] for p in fc.get("path", [])]),
                          question, agent="coordinator", run_id=run_id)
    db.log_decision(run_id, "coordinator", "route", {"question": question, "route": route})

    if route.get("intent") == "what_if" and route.get("decision") and "error" not in fc:
        sim = forecaster.simulate(fc, route["decision"])
        db.log_decision(run_id, "forecaster", "simulate", {k: sim[k] for k in (
            "decision", "verdict", "min_balance", "min_balance_pessimistic", "safety_buffer", "shortfall",
            "suggested_cuts")})
        expl = llm.chat_json(EXPLAIN_PROMPT, json.dumps({"question": question, "simulation": {
            k: v for k, v in sim.items() if k != "path"}, "path": sim["path"]}), agent="forecaster", run_id=run_id)
        result = {"run_id": run_id, "intent": "what_if", "answer": expl.get("answer", ""), "simulation": sim,
                  "evidence_ids": ["F1"]}
    else:
        analysis = state["analysis"]
        ctx = {"patterns": [{k: p[k] for k in ("id", "kind", "summary")} for p in analysis["patterns"]],
               "monthly_totals": analysis["monthly"], "by_category_by_month": analysis["by_category"],
               "forecast": {k: fc.get(k) for k in ("monthly_income", "monthly_fixed", "monthly_variable",
                                                  "monthly_net", "current_balance", "variable_by_category")}}
        out = llm.chat_json(QA_PROMPT, json.dumps({"question": question, "data": ctx}), agent="qa", run_id=run_id)
        result = {"run_id": run_id, "intent": "question", "answer": out.get("answer", ""),
                  "evidence_ids": out.get("evidence_ids", [])}
    db.finish_run(run_id, result)
    return result


def build_digest() -> dict:
    """One key insight + one action, generated from the latest real analysis (used by the scheduler)."""
    state = run_pipeline()
    recs = sorted(state.get("recommendations") or [], key=lambda r: -(r.get("monthly_impact") or 0))
    fc = state["forecast"]
    if "error" in fc:
        insight = "Not enough history yet to forecast; keep logging transactions."
        action = "Upload at least two full months of transactions."
    else:
        end = fc["path"][-1]
        insight = (f"At your current pace you net INR {fc['monthly_net']:,.0f}/month; projected balance in "
                   f"{end['month']} is INR {end['balance']:,.0f}.")
        action = f"{recs[0]['action']} ({recs[0]['reason']})" if recs else "No new action this week."
    db.log_decision(state["run_id"], "coordinator", "digest", {
        "insight": insight, "action": action, "based_on": recs[0]["evidence_ids"] if recs else ["F1"]})
    return {"run_id": state["run_id"], "insight": insight, "action": action}
