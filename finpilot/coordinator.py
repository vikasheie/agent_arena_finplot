"""Coordinator: orchestrates the agents with an explicit LangGraph state machine.

Pipeline graph:
    coordinator -> categorize -> analyze -> forecast -> advise -> verify --(rejections & budget left)--> advise
                                                                         +--> finalize
Question flow: the coordinator classifies the user's question and routes it to the forecaster's
what-if simulator (affordability / decisions) or the Q&A agent (free-form questions over real data).
"""
import contextvars
import json
import threading
from datetime import date, datetime
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
    digest_preview: dict


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
    if "error" not in fc:
        fc["summary"], fc["summary_source"] = _forecast_summary(fc, gs, state["run_id"])
    return {"forecast": fc, "goal_status": gs}


def _forecast_summary(fc: dict, gs: list[dict], run_id: int) -> tuple[str, str]:
    """One plain-language sentence about the trajectory: computed facts, optionally phrased by the LLM."""
    facts = forecaster.summary_facts(fc, gs)
    if llm.available():
        try:
            out = llm.chat_json(
                "You are the Forecaster agent of FinPilot. Rewrite these computed facts as ONE friendly sentence "
                "(max 30 words) for the user, keeping every number exactly. Respond as JSON {\"summary\": \"...\"}",
                json.dumps(facts), agent="forecaster", run_id=run_id)
            if out.get("summary"):
                return out["summary"], "llm"
        except Exception as e:
            db.log_decision(run_id, "forecaster", "error", {"error": str(e)})
    return facts["sentence"], "template"


def _advise(state: PipelineState) -> PipelineState:
    rnd = state.get("advice_round", 0) + 1
    try:
        drafts = advisor.advise(state["analysis"], state["forecast"], state.get("goal_status", []),
                                feedback=state.get("rejected"), accepted=state.get("accepted"),
                                dismissed=db.dismissed_recommendations(), run_id=state["run_id"])
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
    digest = compose_digest(state.get("analysis") or {}, state.get("forecast") or {}, recs)
    db.log_decision(state["run_id"], "digest", "compose", digest)
    return {"recommendations": recs, "digest_preview": digest}


def compose_digest(analysis: dict, fc: dict, recs: list[dict]) -> dict:
    """The weekly message: one insight from the user's own data + the single highest-impact verified action."""
    patterns = {p["id"]: p for p in analysis.get("patterns", [])}
    top = max(recs, key=lambda r: r.get("monthly_impact") or 0) if recs else None
    if top:
        ev = next((patterns[i] for i in top["evidence_ids"] if i in patterns), None)
        insight = ev["summary"] if ev else top["reason"]
        saving = f" to save INR {top['monthly_impact']:,.0f}/month" if top.get("monthly_impact") else ""
        return {"insight": insight, "action": f"{top['action']}{saving}", "based_on": top["evidence_ids"]}
    if not fc or "error" in fc:
        return {"insight": "Not enough history yet to forecast.", "action": "Upload at least two full months.",
                "based_on": []}
    end = fc["path"][-1]
    return {"insight": f"You net INR {fc['monthly_net']:,.0f}/month; projected balance in {end['month']} is "
                       f"INR {end['balance']:,.0f}.", "action": "No new action this week.", "based_on": ["F1"]}


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


def run_pipeline(run_id: int | None = None) -> dict:
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    run_id = run_id or db.start_run("pipeline")
    try:
        state = _GRAPH.invoke({"run_id": run_id})
    except Exception as e:
        db.log_decision(run_id, "coordinator", "error", {"error": str(e)})
        db.finish_run(run_id, {"error": str(e)}, status="error")
        raise
    keep = {k: state.get(k) for k in ("run_id", "categorization", "analysis", "forecast", "goal_status",
                                      "recommendations", "rejected_history", "advice_round", "digest_preview")}
    keep["tokens"] = db.token_usage(run_id)
    db.finish_run(run_id, keep)
    return keep


def start_pipeline() -> int:
    """Run the graph in a background thread so the UI can watch the trail fill in live."""
    run_id = db.start_run("pipeline")
    ctx = contextvars.copy_context()  # carries the active profile's database

    def work():
        try:
            run_pipeline(run_id)
        except Exception:
            pass  # already recorded in the run + decision trail

    threading.Thread(target=lambda: ctx.run(work), daemon=True, name=f"pipeline-{run_id}").start()
    return run_id


AGENT_ORDER = ["coordinator", "categorizer", "analyzer", "forecaster", "advisor", "verifier", "digest"]


def run_status(run_id: int) -> dict:
    run = db.get_run(run_id)
    if not run:
        return {}
    events = db.trail(run_id)
    agents = {a: {"status": "pending", "events": 0, "tokens": 0, "llm_calls": 0, "errors": 0} for a in AGENT_ORDER}
    for e in events:
        a = agents.setdefault(e["agent"], {"status": "pending", "events": 0, "tokens": 0, "llm_calls": 0, "errors": 0})
        a["events"] += 1
        a["status"] = "done"
        if e["step"] == "llm_call":
            a["llm_calls"] += 1
            a["tokens"] += int(e["detail"].get("tokens") or 0)
        if e["step"] in ("llm_error", "error"):
            a["errors"] += 1
    # Wall-clock time each agent spent: from the previous agent's last event to this agent's last event.
    prev_t = datetime.fromisoformat(run["started_at"])
    for e in events:
        t = datetime.fromisoformat(e["created_at"])
        agents[e["agent"]]["duration_ms"] = agents[e["agent"]].get("duration_ms", 0) + \
            max(0, round((t - prev_t).total_seconds() * 1000))
        prev_t = t
    latencies = [e["detail"]["latency_ms"] for e in events if e["step"] == "llm_call" and e["detail"].get("latency_ms")]
    active = events[-1]["agent"] if events and run["status"] == "running" else None
    if active:
        agents[active]["status"] = "running"
    verify = [e["detail"] for e in events if e["step"] == "verify"]
    drafted = sum(len(v["accepted"]) + len(v["rejected"]) for v in verify)
    return {**run, "active_agent": active, "agents": agents, "events": events,
            "tokens_total": sum(a["tokens"] for a in agents.values()),
            "llm_calls": len(latencies), "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            "verified_drafts": sum(len(v["accepted"]) for v in verify), "total_drafts": drafted}


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
    d = state.get("digest_preview") or compose_digest(state.get("analysis") or {}, state.get("forecast") or {}, [])
    return {"run_id": state["run_id"], "insight": d["insight"], "action": d["action"]}
