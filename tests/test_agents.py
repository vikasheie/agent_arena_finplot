import json

import pytest

from finpilot import coordinator, db, ingest, llm
from finpilot.agents import advisor, analyzer, categorizer, forecaster


def test_merchant_key_normalises_bank_noise():
    assert categorizer.merchant_key("UPI/SWIGGY/123456@ybl") == "swiggy"
    assert categorizer.merchant_key("UPI/P2M/998877/SWIGGY") == "swiggy"
    assert categorizer.merchant_key("DEBIT CARD SPOTIFY INDIA BLR") == categorizer.merchant_key("SPOTIFY INDIA*4411")


def test_csv_ingest_signed_and_debit_credit():
    signed = ingest.parse_csv("date,description,amount\n2026-01-05,Swiggy,-450\n05/01/2026,Salary,\"85,000\"\n")
    assert [r["amount"] for r in signed] == [-450.0, 85000.0]
    assert signed[1]["date"] == "2026-01-05"
    dc = ingest.parse_csv("Txn Date,Narration,Debit,Credit\n2026-01-05,ZOMATO,300,\n2026-01-06,NEFT SAL,,5000\n")
    assert [r["amount"] for r in dc] == [-300.0, 5000.0]
    with pytest.raises(ValueError):
        ingest.parse_csv("foo,bar\n1,2\n")


def test_categorizer_self_check_loop_rechecks_invalid_labels(fresh_db, monkeypatch):
    db.insert_transactions([
        {"date": "2026-01-01", "description": "NEFT CR ACME SALARY", "amount": 85000},
        {"date": "2026-01-02", "description": "UPI/ZOMATO/1", "amount": -400},
    ])
    calls = []

    def fake(system, user, *, agent, run_id=None, model=None, **kw):
        calls.append(model)
        ids = [int(x) for x in __import__("re").findall(r'"id": (\d+)', user)]
        if len(calls) == 1:  # first pass: one contradiction (money in labelled dining) + one valid
            return {"labels": [{"id": ids[0], "category": "dining", "confidence": 0.9},
                               {"id": ids[1], "category": "dining", "confidence": 0.95}]}
        assert "rejected" in user  # the recheck carries the verifier's reasons
        return {"labels": [{"id": i, "category": "income", "confidence": 0.9} for i in ids]}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", fake)
    stats = categorizer.categorize_pending()
    txs = db.all_transactions()
    assert [t["category"] for t in txs] == ["income", "dining"]
    assert txs[0]["category_source"] == "llm_recheck"
    assert len(calls) == 2 and stats["llm_recheck"] == 1


def test_user_correction_teaches_memory(fresh_db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    db.insert_transactions([{"date": "2026-01-01", "description": "UPI/RAMESH TIFFIN/11", "amount": -120}])
    categorizer.categorize_pending()
    categorizer.correct(db.all_transactions()[0]["id"], "dining")
    db.insert_transactions([{"date": "2026-01-08", "description": "UPI/RAMESH TIFFIN/99", "amount": -150}])
    categorizer.categorize_pending()
    assert db.all_transactions()[1]["category_source"] == "memory"
    assert db.all_transactions()[1]["category"] == "dining"


def test_analyzer_finds_injected_anomalies_and_subscriptions(labelled_db, persona_rows):
    txs = db.all_transactions()
    flagged = {(a["date"], a["description"]) for a in analyzer.detect_anomalies(txs)}
    injected = {(r["date"], r["description"]) for r in persona_rows if r["is_anomaly"] == "1"}
    assert injected <= flagged
    recurring = " ".join(r["description"].lower() for r in analyzer.detect_recurring(txs))
    for name in ("netflix", "spotify", "hotstar", "cultfit"):
        assert name in recurring


def test_forecast_beats_naive_baseline_and_simulation_verdicts(labelled_db):
    txs = db.all_transactions()
    bt = forecaster.backtest(txs)
    assert bt["rows"] and bt["mape_spend_pct"] < 20
    fc = forecaster.project(txs, opening_balance=50000)
    small = forecaster.simulate(fc, {"type": "purchase", "amount": 1000, "month_offset": 1})
    huge = forecaster.simulate(fc, {"type": "purchase", "amount": 10_000_000, "month_offset": 1})
    assert small["verdict"] == "affordable"
    assert huge["verdict"] == "not_affordable" and huge["shortfall"] > 0
    cut = forecaster.simulate(fc, {"type": "recurring_change", "monthly_delta": -2000, "month_offset": 1})
    assert cut["path"][-1]["scenario"] == pytest.approx(fc["path"][-1]["balance"] + 2000 * len(fc["path"]))


def _evidence(labelled_db):
    analysis = analyzer.analyze()
    fc = forecaster.project(db.all_transactions(), opening_balance=50000)
    return analysis, fc, advisor._evidence_index(analysis, fc, [])


def test_verifier_rejects_ungrounded_and_accepts_grounded(labelled_db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    analysis, fc, ev = _evidence(labelled_db)
    hotstar = next(p for p in analysis["patterns"] if p["kind"] == "recurring" and "hotstar" in p["summary"].lower())
    good = {"title": "Cancel Disney Hotstar", "action": "Cancel Disney Hotstar", "monthly_impact": 299,
            "evidence_ids": [hotstar["id"]], "reason": "recurring"}
    inflated = {**good, "monthly_impact": 5000}
    no_evidence = {**good, "evidence_ids": []}
    fake_id = {**good, "evidence_ids": ["P999"]}
    generic = {"title": "Create a budget", "action": "Create a budget and spend less", "monthly_impact": 0,
               "evidence_ids": [hotstar["id"]], "reason": "x"}
    ok, bad = advisor.verify([good, inflated, no_evidence, fake_id, generic], ev)
    assert [r["title"] for r in ok] == ["Cancel Disney Hotstar"]
    notes = " | ".join(r["verifier_note"] for r in bad)
    assert "supports at most" in notes and "cites no evidence" in notes and "unknown evidence" in notes
    assert "does not name" in notes


def test_pipeline_advisor_verifier_loop_iterates(labelled_db, monkeypatch):
    """The LangGraph loop must route rejected advice back to the advisor and record why."""
    rounds = []

    def fake(system, user, *, agent, run_id=None, model=None, **kw):
        if agent in ("analyzer", "forecaster"):
            return {"narrative": "test", "summary": "test"}
        assert agent == "advisor"
        rounds.append(user)
        data = json.loads(user.split("\n\n")[0])
        sub = next(p for p in data["patterns"] if p["kind"] == "recurring" and "netflix" in p["summary"].lower())
        if len(rounds) == 1:  # over-claims savings -> verifier must bounce it
            return {"recommendations": [{"title": "Cancel Netflix", "action": "Cancel Netflix",
                                         "monthly_impact": 99999, "evidence_ids": [sub["id"]], "reason": "r"}]}
        assert "Verifier rejected" in user
        return {"recommendations": [{"title": "Cancel Netflix", "action": "Cancel Netflix",
                                     "monthly_impact": 649, "evidence_ids": [sub["id"]], "reason": "r"}]}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", fake)
    state = coordinator.run_pipeline()
    assert state["advice_round"] == 2 and len(rounds) == 2
    recs = db.recommendations(state["run_id"])
    assert len(recs) == 1 and recs[0]["verified"] == 1 and recs[0]["evidence_ids"]
    steps = [(e["agent"], e["step"]) for e in db.trail(state["run_id"])]
    for expected in [("coordinator", "plan"), ("analyzer", "analyze"), ("forecaster", "forecast"),
                     ("verifier", "verify"), ("coordinator", "finalize")]:
        assert expected in steps
    assert steps.count(("verifier", "verify")) == 2


def test_circuit_breaker_stops_calls_after_billing_error(fresh_db, monkeypatch):
    import httpx
    from openai import APIStatusError

    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw["model"])
            req = httpx.Request("POST", "https://api.meshapi.ai/v1/chat/completions")
            raise APIStatusError("Insufficient balance", response=httpx.Response(402, request=req), body=None)

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm, "client", lambda: FakeClient())
    monkeypatch.setattr(llm, "_breaker", {"open_until": 0.0, "reason": ""})
    for _ in range(3):
        with pytest.raises(RuntimeError):
            llm.chat_json("s", "u", agent="categorizer")
    assert len(calls) == 1  # first call trips the breaker; the next two never reach Mesh


def test_reupload_skips_duplicates_but_keeps_genuine_repeats(fresh_db):
    csv_text = "date,description,amount\n2026-01-05,METRO,-40\n2026-01-05,METRO,-40\n2026-01-06,SWIGGY,-300\n"
    first = ingest.ingest_csv(csv_text)
    assert first["inserted"] == 3  # two identical metro rides on one day are both real
    again = ingest.ingest_csv(csv_text)
    assert again == {"inserted": 0, "skipped_duplicates": 3, "ids": []}
    assert len(db.all_transactions()) == 3


def test_fallback_labels_are_relabelled_once_llm_is_back(fresh_db, monkeypatch):
    db.insert_transactions([{"date": "2026-01-01", "description": "UPI/CHAAYOS/1", "amount": -250}])
    monkeypatch.setattr(llm, "available", lambda: False)
    categorizer.categorize_pending()
    assert db.all_transactions()[0]["category_source"] == "fallback"

    def fake(system, user, *, agent, run_id=None, model=None, **kw):
        ids = [int(x) for x in __import__("re").findall(r'"id": (\d+)', user)]
        return {"labels": [{"id": i, "category": "dining", "confidence": 0.95, "why": "Chaayos is a tea cafe"}
                           for i in ids]}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "chat_json", fake)
    stats = categorizer.categorize_pending()
    tx = db.all_transactions()[0]
    assert stats["llm"] == 1 and tx["category"] == "dining" and tx["category_source"] == "llm"
    assert tx["category_reason"] == "Chaayos is a tea cafe"


def test_verifier_rejects_cutting_essentials(labelled_db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    analysis, fc, ev = _evidence(labelled_db)
    rent = next(p for p in analysis["patterns"] if p["kind"] == "recurring" and p["data"]["category"] == "rent")
    rec = {"title": "Move to a cheaper flat", "action": f"Negotiate rent with {rent['data']['name']}",
           "monthly_impact": 2000, "evidence_ids": [rent["id"]], "reason": "rent is high"}
    ok, bad = advisor.verify([rec], ev)
    assert not ok and "essential spend" in bad[0]["verifier_note"]


def test_verifier_rejects_implausibly_small_saving(labelled_db, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    analysis, fc, ev = _evidence(labelled_db)
    top = next(p for p in analysis["patterns"] if p["kind"] == "top_merchant")
    rec = {"title": f"Cut {top['data']['merchant']}", "action": f"Order less from {top['data']['merchant']}",
           "monthly_impact": 3, "evidence_ids": [top["id"]], "reason": "delivery adds up"}
    ok, bad = advisor.verify([rec], ev)
    assert not ok and "realistic monthly saving" in bad[0]["verifier_note"]


def test_garbled_llm_explanation_is_replaced_by_computed_numbers(labelled_db, monkeypatch):
    """The model once wrote '\\u000a920,000' for '₹20,000'. Numbers the simulator didn't compute must never reach the user."""
    fc = forecaster.project(db.all_transactions(), opening_balance=50000)
    sim = forecaster.simulate(fc, {"type": "purchase", "amount": 20000, "month_offset": 1})
    garbled = f"Yes, you can afford the 920,000 purchase. Lowest balance {sim['min_balance_pessimistic']:.2f}."
    assert coordinator.unsupported_numbers(garbled, sim) == ["920,000"]
    good = f"Yes, the INR 20,000 purchase is fine; lowest balance INR {sim['min_balance']:,.0f}."
    assert coordinator.unsupported_numbers(good, sim) == []
    assert "INR 20,000" in coordinator.simulation_sentence(sim)


def test_earliest_purchase_finds_future_month(labelled_db):
    txs = db.all_transactions()
    now = forecaster.simulate(forecaster.project(txs, opening_balance=0), {"type": "purchase", "amount": 150000,
                                                                          "month_offset": 1})
    assert now["verdict"] != "affordable"
    e = forecaster.earliest_purchase(txs, 150000, opening_balance=0)
    assert e["affordable"] and e["affordable"]["month_offset"] > 1
    later = forecaster.simulate(forecaster.project(txs, horizon=24, opening_balance=0),
                                {"type": "purchase", "amount": 150000, "month_offset": e["affordable"]["month_offset"]})
    assert later["verdict"] == "affordable"
    sim = {**now, "earliest": e}
    text = coordinator.simulation_sentence(sim)
    assert "comfortably affordable in" in text and coordinator.unsupported_numbers(text, sim) == []
