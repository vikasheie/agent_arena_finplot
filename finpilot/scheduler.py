"""Proactive delivery: a real APScheduler cron job builds and sends the digest (no manual button needed)."""
import logging

import httpx
from apscheduler.schedulers.background import BackgroundScheduler

from finpilot import config, db, llm
from finpilot.coordinator import build_digest

log = logging.getLogger("finpilot.scheduler")
_scheduler: BackgroundScheduler | None = None


def _send_telegram(text: str) -> bool:
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return False
    r = httpx.post(f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                   json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=15)
    return r.status_code == 200


def digest_job() -> dict | None:
    if not db.all_transactions():
        log.info("digest skipped: no transactions")
        return None
    if not llm.available():
        log.info("digest skipped: MESH_API_KEY missing")
        return None
    d = build_digest()
    text = f"FinPilot weekly digest\nInsight: {d['insight']}\nAction: {d['action']}"
    via = "telegram" if _send_telegram(text) else "in_app"
    db.save_digest(d["run_id"], d["insight"], d["action"], via)
    log.info("digest delivered via %s", via)
    return {**d, "delivered_via": via}


def start() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
        _scheduler.add_job(digest_job, "cron", day_of_week=config.DIGEST_DAY_OF_WEEK, hour=config.DIGEST_HOUR,
                           id="weekly_digest", replace_existing=True)
        _scheduler.start()
    return _scheduler


def shutdown() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def next_run() -> str | None:
    if _scheduler:
        job = _scheduler.get_job("weekly_digest")
        return job.next_run_time.isoformat() if job and job.next_run_time else None
    return None
