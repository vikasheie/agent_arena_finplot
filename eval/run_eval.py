"""Self-evaluation: measures categorisation, anomaly detection, recurring detection, forecast accuracy
and the advisor/verifier loop on labelled synthetic data. Writes eval/results.json and eval/RESULTS.md.

Run:  python -m eval.run_eval            (full, needs MESH_API_KEY)
      python -m eval.run_eval --no-llm   (deterministic parts + keyword baseline only)
"""
import argparse
import csv
import json
import statistics
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from finpilot import config, db, llm
from finpilot.agents import analyzer, categorizer, forecaster

HERE = Path(__file__).resolve().parent
PERSONAS = ("priya", "aarav", "sharma")
# Known scheduled payments per persona (ground truth for recurring detection).
EXPECTED_RECURRING = {
    "priya": {"netflix", "spotify", "hotstar", "cultfit", "youtube", "chatgpt", "airtel", "jio", "zerodha", "mom",
              "rent"},
    "aarav": {"stanza", "spotify", "adobe", "xbox", "netflix", "jio"},
    "sharma": {"home loan", "car loan", "act fibernet", "indane", "netflix", "prime", "hotstar", "sonyliv", "maid",
               "lic", "ppf", "school fees", "byjus"},
}


def _read(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["amount"] = float(r["amount"])
    return rows


def _fresh_db() -> str:
    path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    config.DB_PATH = path
    db.init_db()
    return path


def _scores(truth: list[str], pred: list[str]) -> dict:
    acc = sum(t == p for t, p in zip(truth, pred)) / len(truth)
    per = {}
    for c in sorted(set(truth)):
        tp = sum(t == c and p == c for t, p in zip(truth, pred))
        fp = sum(t != c and p == c for t, p in zip(truth, pred))
        fn = sum(t == c and p != c for t, p in zip(truth, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0, "support": tp + fn}
    errors = Counter(f"{t} -> {p}" for t, p in zip(truth, pred) if t != p)
    return {"accuracy": round(acc, 4), "macro_f1": round(statistics.mean(v["f1"] for v in per.values()), 4),
            "n": len(truth), "per_category": per, "top_confusions": errors.most_common(8)}


def eval_categorization(rows: list[dict], use_llm: bool) -> dict:
    _fresh_db()
    db.insert_transactions(rows)
    saved_key = config.MESH_API_KEY
    if not use_llm:
        config.MESH_API_KEY = ""  # forces the keyword fallback path
    try:
        stats = categorizer.categorize_pending(run_id=None)
    finally:
        config.MESH_API_KEY = saved_key
    # Transactions come back ordered by (date, id); id follows insertion (= row) order.
    pred = [t["category"] for t in db.all_transactions()]
    truth = [r["category"] for _, r in sorted(enumerate(rows), key=lambda x: (x[1]["date"], x[0]))]
    return {**_scores(truth, pred), "sources": {k: v for k, v in stats.items() if k != "recheck_reasons"},
            "recheck_rounds": stats.get("recheck_reasons", [])}


def _load_labelled(rows: list[dict]) -> list[dict]:
    """Ground-truth categories loaded directly, to evaluate downstream agents in isolation."""
    _fresh_db()
    ids = db.insert_transactions(rows)
    for i, r in zip(ids, rows):
        db.set_category(i, r["category"], "user", 1.0)
    return db.all_transactions()


def eval_anomalies(rows: list[dict]) -> dict:
    txs = _load_labelled(rows)
    truth = {(r["date"], r["description"]) for r in rows if r["is_anomaly"] == "1"}
    found = {(a["date"], a["description"]) for a in analyzer.detect_anomalies(txs)}
    tp = len(truth & found)
    return {"injected": len(truth), "flagged": len(found), "true_positives": tp,
            "precision": round(tp / len(found), 3) if found else 0.0,
            "recall": round(tp / len(truth), 3) if truth else 0.0,
            "false_positives": sorted(found - truth)[:10]}


def eval_recurring(rows: list[dict], expected: set[str]) -> dict:
    txs = _load_labelled(rows)
    found = analyzer.detect_recurring(txs)
    keys = " | ".join(r["merchant"] + " " + r["description"].lower() for r in found)
    hit = {e for e in expected if e in keys}
    return {"expected": sorted(expected), "detected_expected": sorted(hit), "missed": sorted(expected - hit),
            "recall": round(len(hit) / len(expected), 3), "total_detected": len(found),
            "detected": [f"{r['description']} ~{r['monthly_amount']}" for r in found]}


def eval_forecast(rows: list[dict]) -> dict:
    txs = _load_labelled(rows)
    bt = forecaster.backtest(txs)
    # Naive baseline: next month's spend = last month's spend.
    months = forecaster._full_months(txs)
    spend = {m: sum(-t["amount"] for t in txs if t["date"][:7] == m and t["amount"] < 0) for m in months}
    naive = [100 * abs(spend[months[k - 1]] - spend[months[k]]) / spend[months[k]] for k in range(3, len(months))]
    return {**bt, "naive_last_month_mape_pct": round(statistics.mean(naive), 2) if naive else None}


def eval_advisor(rows: list[dict], opening_balance: float) -> dict:
    from finpilot import coordinator
    _fresh_db()
    db.set_setting("opening_balance", str(opening_balance))
    db.insert_transactions([{k: r[k] for k in ("date", "description", "amount")} for r in rows])
    db.add_goal("Emergency fund top-up", 60000, "2027-03-31")
    state = coordinator.run_pipeline()
    recs = state.get("recommendations") or []
    rejected = state.get("rejected_history") or []
    trail = db.trail(state["run_id"])
    return {
        "recommendations": len(recs),
        "advice_rounds": state.get("advice_round"),
        "rejected_by_verifier_total": len(rejected),
        "rejection_reasons": [r["verifier_note"] for r in rejected][:8],
        "all_cite_evidence": all(r["evidence_ids"] for r in recs),
        "llm_calls": sum(1 for e in trail if e["step"] == "llm_call"),
        "tokens": sum(state.get("tokens", {}).values()),
        "sample": [{k: r[k] for k in ("title", "monthly_impact", "evidence_ids", "reason")} for r in recs[:5]],
    }


def _pct(x: float) -> str:
    return f"{x:.0%}"


def to_markdown(res: dict) -> str:
    out = [f"# FinPilot self-evaluation results\n\nGenerated {res['generated_at']} · models "
           f"`{res['models']['fast']}` / `{res['models']['smart']}` · LLM used: {res['llm_used']}\n"]

    out.append("## Categorisation accuracy\n\n| Dataset | n | Keyword baseline | LLM agent (Mesh) | LLM macro-F1 |"
               "\n|---|---|---|---|---|")
    for key, label in [("heldout", "Held-out, 74 hand-labelled, unseen merchants")] + \
            [(p, f"Persona: {p}") for p in PERSONAS]:
        c = res["categorization"][key]
        kb, la = c["keyword_baseline"], c.get("llm_agent")
        out.append(f"| {label} | {kb['n']} | {kb['accuracy']:.1%} | "
                   f"{format(la['accuracy'], '.1%') if la else 'n/a'} | "
                   f"{la['macro_f1'] if la else 'n/a'} |")
    la = res["categorization"]["heldout"].get("llm_agent")
    if la:
        out.append(f"\nHeld-out label sources: `{la['sources']}` · top confusions: {la['top_confusions'][:5]}")

    out.append("\n## Anomaly detection (injected anomalies)\n\n| Persona | Injected | Flagged | Precision | Recall |"
               "\n|---|---|---|---|---|")
    for p in PERSONAS:
        a = res["anomalies"][p]
        out.append(f"| {p} | {a['injected']} | {a['flagged']} | {_pct(a['precision'])} | {_pct(a['recall'])} |")

    out.append("\n## Recurring-payment detection\n\n| Persona | Known recurring | Recall | Missed |\n|---|---|---|---|")
    for p in PERSONAS:
        r = res["recurring"][p]
        out.append(f"| {p} | {len(r['expected'])} | {_pct(r['recall'])} | {', '.join(r['missed']) or '-'} |")

    out.append("\n## Forecast backtest (rolling origin, 1-month-ahead total outflow)\n\n"
               "| Persona | Origins | FinPilot MAPE | Naive last-month MAPE | Net-cashflow MAE (INR) |\n|---|---|---|---|---|")
    for p in PERSONAS:
        f = res["forecast"][p]
        if f.get("rows"):
            out.append(f"| {p} | {len(f['rows'])} | **{f['mape_spend_pct']}%** | {f['naive_last_month_mape_pct']}% | "
                       f"{f['mae_net']:,.0f} |")
    for p in PERSONAS:
        f = res["forecast"][p]
        if f.get("rows"):
            out.append(f"\n<details><summary>{p}: month-by-month</summary>\n\n| Month | Predicted | Actual | Abs % error |"
                       "\n|---|---|---|---|")
            out += [f"| {x['month']} | {x['predicted_spend']:,.0f} | {x['actual_spend']:,.0f} | {x['abs_pct_error']}% |"
                    for x in f["rows"]]
            out.append("\n</details>")

    if res.get("advisor"):
        out.append("\n## Advisor ⇄ Verifier loop (end-to-end, real Mesh calls)\n")
        for p, v in res["advisor"].items():
            out.append(f"\n**{p}**: {v['recommendations']} verified recommendations after {v['advice_rounds']} round(s); "
                       f"verifier rejected {v['rejected_by_verifier_total']} drafts; all cite evidence: "
                       f"{v['all_cite_evidence']}; {v['llm_calls']} Mesh calls, {v['tokens']:,} tokens.\n")
            out += [f"- **{s['title']}** (+INR {s['monthly_impact']}/mo, evidence {s['evidence_ids']}) — {s['reason']}"
                    for s in v["sample"]]
            if v["rejection_reasons"]:
                out.append("\n  Verifier rejections: " + " · ".join(v["rejection_reasons"][:3]))
    return "\n".join(out) + "\n"


def main() -> None:
    from finpilot.profiles import PROFILES
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    use_llm = llm.available() and not args.no_llm

    datasets = {"heldout": _read(HERE / "categorization_testset.csv")}
    personas = {p: _read(HERE / "ground_truth" / f"{p}.csv") for p in PERSONAS}
    for p, rows in personas.items():
        datasets[p] = [{k: r[k] for k in ("date", "description", "amount", "category")} for r in rows]

    res = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "models": {"fast": config.MODEL_FAST, "smart": config.MODEL_SMART}, "llm_used": use_llm,
           "categorization": {}, "anomalies": {}, "recurring": {}, "forecast": {}}
    for key, rows in datasets.items():
        res["categorization"][key] = {"keyword_baseline": eval_categorization(rows, use_llm=False)}
        if use_llm:
            res["categorization"][key]["llm_agent"] = eval_categorization(rows, use_llm=True)
    for p, rows in personas.items():
        res["anomalies"][p] = eval_anomalies(rows)
        res["recurring"][p] = eval_recurring(rows, EXPECTED_RECURRING[p])
        res["forecast"][p] = eval_forecast(rows)
    if use_llm:
        res["advisor"] = {p: eval_advisor(rows, PROFILES[p]["opening_balance"]) for p, rows in personas.items()}

    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=str))
    (HERE / "RESULTS.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
