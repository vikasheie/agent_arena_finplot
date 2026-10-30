"""Categorizer agent: labels each transaction from its own description, amount and direction.

Pipeline per transaction:
  1. merchant memory (labels the user corrected or the agent learned with high confidence)
  2. LLM batch classification via Mesh
  3. self-check loop: invalid / low-confidence / sign-inconsistent labels are re-asked with the
     stronger model plus few-shot examples taken from the user's own already-labelled history.
A keyword fallback exists only for running without a Mesh key; it is tagged `fallback` and
evaluated separately in eval/.
"""
import json
import re

from finpilot import config, db, llm

AGENT = "categorizer"
BATCH = 40
CONFIDENCE_FLOOR = 0.7
MAX_RECHECK_ROUNDS = 2

CATEGORY_GUIDE = {
    "income": "salary, freelance payments, interest, refunds of income type (money in)",
    "rent": "house rent, PG, maintenance paid to landlord/society",
    "emi": "loan EMIs: home loan, car loan, personal loan, consumer-durable EMI",
    "groceries": "supermarkets, kirana, vegetables, grocery delivery (BigBasket, Blinkit, Zepto)",
    "dining": "restaurants, cafes, food delivery (Swiggy, Zomato)",
    "transport": "fuel, cabs (Uber, Ola, Rapido), metro, parking, tolls",
    "utilities": "electricity, water, gas, mobile recharge, broadband",
    "subscriptions": "recurring digital services (Netflix, Spotify, Prime, Hotstar, gym membership, SaaS)",
    "shopping": "e-commerce and retail (Amazon, Flipkart, Myntra), electronics, clothes",
    "entertainment": "movies, events, gaming, BookMyShow",
    "health": "pharmacy, doctor, hospital, insurance premium",
    "education": "school/college fees, tuition, courses, books for study",
    "travel": "flights, trains, hotels, holiday bookings",
    "transfers": "transfers to self/friends/family, credit card bill payment, investments, ATM cash",
    "other": "anything that clearly fits none of the above",
}

_PREFIX = re.compile(r"^(upi|pos|neft|imps|ach|nach|debit card|card)[\s/:-]+", re.I)
_NOISE = re.compile(r"[^a-z ]+")
_STOP = {"pvt", "ltd", "india", "payment", "ref", "txn", "upi", "pos", "pm", "ybl", "com", "in",
         "blr", "bangalore", "bengaluru", "mumbai", "delhi", "to", "by", "from", "the", "cr", "dr"}


def merchant_key(description: str) -> str:
    """Normalise a raw bank description to a stable merchant key ("UPI/SWIGGY*1234" -> "swiggy")."""
    d = _PREFIX.sub("", description.strip().lower())
    d = _NOISE.sub(" ", d)
    tokens = [t for t in d.split() if len(t) > 1 and t not in _STOP]
    return " ".join(tokens[:2]) or description.strip().lower()


def _fallback_category(tx: dict) -> tuple[str, float, str]:
    """Offline keyword heuristic over the transaction text. Used only when Mesh is unavailable."""
    text = tx["description"].lower()
    if tx["amount"] > 0:
        hit = next((k for k in ("salary", "payroll", "freelance", "interest", "payout") if k in text), None)
        return ("income", 0.6, f"keyword rule: '{hit}'") if hit else ("transfers", 0.4, "money in, no income keyword")
    for cat, keys in {
        "emi": ("emi", "loan"),
        "rent": ("rent", "landlord", "society", " pg"),
        "education": ("school", "tuition", "fees", "course", "byjus"),
        "groceries": ("bigbasket", "blinkit", "zepto", "dmart", "grocery", "kirana", "more retail"),
        "dining": ("swiggy", "zomato", "cafe", "restaurant", "starbucks", "dominos", "kfc"),
        "transport": ("uber", "ola", "rapido", "fuel", "petrol", "metro", "fastag"),
        "utilities": ("electricity", "bescom", "airtel", "jio", "broadband", "water", "gas"),
        "subscriptions": ("netflix", "spotify", "prime", "hotstar", "youtube", "gym", "cult"),
        "shopping": ("amazon", "flipkart", "myntra", "ajio", "croma", "nykaa"),
        "entertainment": ("bookmyshow", "pvr", "inox", "steam"),
        "health": ("pharmacy", "apollo", "medplus", "hospital", "clinic", "insurance"),
        "travel": ("indigo", "irctc", "makemytrip", "airbnb", "oyo", "goibibo"),
        "transfers": ("transfer", "atm", "cred", "zerodha", "groww", "sip"),
    }.items():
        hit = next((k.strip() for k in keys if re.search(rf"\b{re.escape(k.strip())}\b", text)), None)
        if hit:
            return cat, 0.5, f"keyword rule: '{hit}'"
    return "other", 0.2, "no keyword matched"


def _invalid_reason(tx: dict, label: dict | None) -> str | None:
    if not label:
        return "missing label"
    cat, conf = label.get("category"), label.get("confidence", 0)
    if cat not in config.CATEGORIES:
        return f"unknown category '{cat}'"
    if cat == "income" and tx["amount"] < 0:
        return "labelled income but money went out"
    if tx["amount"] > 0 and cat not in {"income", "transfers", "shopping", "travel", "other"}:
        return f"money came in but labelled '{cat}'"
    try:
        if float(conf) < CONFIDENCE_FLOOR:
            return f"low confidence {conf}"
    except (TypeError, ValueError):
        return "confidence not numeric"
    return None


def _system_prompt() -> str:
    guide = "\n".join(f"- {k}: {v}" for k, v in CATEGORY_GUIDE.items())
    return (
        "You are the Categorizer agent of FinPilot, a personal finance system for an Indian user (INR).\n"
        "Classify each bank transaction into exactly one category using its description, amount and "
        "direction (amount > 0 is money in, < 0 is money out).\n"
        f"Categories:\n{guide}\n"
        'Respond with JSON: {"labels": [{"id": <int>, "category": <category>, "confidence": <0..1>, '
        '"why": <max 8 words, e.g. "Swiggy is food delivery">}]}. '
        "Give a label for every id. Use lower confidence when the description is ambiguous."
    )


def _ask(batch: list[dict], run_id: int | None, model: str, extra: str = "") -> dict[int, dict]:
    payload = [{"id": t["id"], "date": t["date"], "description": t["description"], "amount": t["amount"]}
               for t in batch]
    data = llm.chat_json(_system_prompt(), extra + json.dumps(payload), agent=AGENT, run_id=run_id, model=model)
    out = {}
    for item in data.get("labels", []):
        try:
            out[int(item["id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _few_shot(labelled: list[dict], k: int = 25) -> str:
    seen, shots = set(), []
    for t in labelled:
        key = merchant_key(t["description"])
        if key in seen or t.get("category_source") not in {"user", "memory", "llm"}:
            continue
        seen.add(key)
        shots.append(f'"{t["description"]}" ({t["amount"]:+.0f}) -> {t["category"]}')
        if len(shots) >= k:
            break
    return "Examples from this user's own history:\n" + "\n".join(shots) + "\n\n" if shots else ""


def categorize_pending(run_id: int | None = None) -> dict:
    pending = db.uncategorized()
    stats = {"total": len(pending), "memory": 0, "llm": 0, "llm_recheck": 0, "fallback": 0, "recheck_reasons": []}
    if not pending:
        return stats

    memory = db.get_memory()
    to_llm = []
    for tx in pending:
        key = merchant_key(tx["description"])
        if key in memory:
            db.set_category(tx["id"], memory[key], "memory", 0.99, f"merchant memory: '{key}' seen before")
            stats["memory"] += 1
        else:
            to_llm.append(tx)

    if not llm.available():
        for tx in to_llm:
            cat, conf, why = _fallback_category(tx)
            db.set_category(tx["id"], cat, "fallback", conf, why)
            stats["fallback"] += 1
        db.log_decision(run_id, AGENT, "categorize", stats)
        return stats

    labels: dict[int, dict] = {}
    for i in range(0, len(to_llm), BATCH):
        try:
            labels.update(_ask(to_llm[i:i + BATCH], run_id, config.MODEL_FAST))
        except Exception as e:  # Mesh outage: these rows fall through to the flagged fallback below
            db.log_decision(run_id, AGENT, "error", {"batch_start": i, "error": str(e)})

    # Self-check loop: re-ask anything invalid with the smart model and the user's own examples.
    recheck = [t for t in to_llm if _invalid_reason(t, labels.get(t["id"]))]
    for round_no in range(MAX_RECHECK_ROUNDS):
        if not recheck:
            break
        reasons = {t["id"]: _invalid_reason(t, labels.get(t["id"])) for t in recheck}
        stats["recheck_reasons"].append({"round": round_no + 1, "items": len(recheck),
                                         "sample": list(reasons.items())[:5]})
        labelled_history = [t for t in db.all_transactions() if t["category"]]
        hint = _few_shot(labelled_history) + "Previous attempt was rejected: " + json.dumps(reasons) + "\n\n"
        try:
            new = _ask(recheck, run_id, config.MODEL_SMART, extra=hint)
        except Exception as e:
            db.log_decision(run_id, AGENT, "error", {"recheck_round": round_no + 1, "error": str(e)})
            break
        for tid, lab in new.items():
            lab["_rechecked"] = True
            labels[tid] = lab
        recheck = [t for t in recheck if _invalid_reason(t, labels.get(t["id"]))]

    for tx in to_llm:
        lab = labels.get(tx["id"])
        reason = _invalid_reason(tx, lab)
        # Accept a valid label even if confidence stayed low; reject structural contradictions.
        if lab and (reason is None or reason.startswith("low confidence")):
            cat, conf = lab["category"], float(lab.get("confidence", 0.5))
            source = "llm_recheck" if lab.get("_rechecked") else "llm"
            why = str(lab.get("why") or "")[:120] + (" (re-checked)" if source == "llm_recheck" else "")
        else:  # still invalid after the loop: deterministic fallback, flagged
            cat, conf, why = _fallback_category(tx)
            source = "fallback"
        db.set_category(tx["id"], cat, source, conf, why)
        stats[source] += 1
        if source != "fallback" and conf >= 0.85:
            db.remember_merchant(merchant_key(tx["description"]), cat, "llm")

    db.log_decision(run_id, AGENT, "categorize", stats, model=config.MODEL_FAST)
    return stats


def correct(tx_id: int, category: str) -> None:
    """User correction: fixes this transaction and teaches the merchant memory."""
    if category not in config.CATEGORIES:
        raise ValueError(f"unknown category {category}")
    tx = next(t for t in db.all_transactions() if t["id"] == tx_id)
    db.set_category(tx_id, category, "user", 1.0, "corrected by you")
    db.remember_merchant(merchant_key(tx["description"]), category, "user")
