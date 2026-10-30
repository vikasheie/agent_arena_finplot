"""Proactive delivery: a real APScheduler cron job builds and sends the digest (no manual button needed)."""
import contextvars
import logging
import smtplib
from email.message import EmailMessage

import httpx
from apscheduler.schedulers.background import BackgroundScheduler

from finpilot import config, db, llm, profiles
from finpilot.coordinator import build_digest

log = logging.getLogger("finpilot.scheduler")
_scheduler: BackgroundScheduler | None = None


def _send_telegram(text: str) -> bool:
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return False
    try:
        r = httpx.post(f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                       json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=15)
    except httpx.HTTPError as e:
        log.warning("telegram delivery failed: %s", e)
        return False
    if r.status_code != 200:
        log.warning("telegram delivery failed: HTTP %s %s", r.status_code, r.text[:200])
    return r.status_code == 200


def _send_email(subject: str, text: str) -> bool:
    if not (config.SMTP_HOST and config.SMTP_FROM and config.DIGEST_EMAIL_TO):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, config.SMTP_FROM, config.DIGEST_EMAIL_TO
    msg.set_content(text)
    try:
        if config.SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=20)
        else:
            server = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20)
            server.starttls()
        with server:
            if config.SMTP_USER:
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except (smtplib.SMTPException, OSError) as e:
        log.warning("email delivery failed: %s", e)
        return False


def channels() -> dict:
    """Which delivery channels are configured (for the UI / health check)."""
    return {"telegram": bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID),
            "email": bool(config.SMTP_HOST and config.SMTP_FROM and config.DIGEST_EMAIL_TO), "in_app": True}


def deliver(insight: str, action: str) -> str:
    """Send through every configured channel; the in-app copy is always kept. Returns e.g. 'telegram+email'."""
    text = f"FinPilot weekly digest\n\nYour week: {insight}\nOne action: {action}"
    sent = [name for name, ok in (("telegram", _send_telegram(text)),
                                  ("email", _send_email("FinPilot: your weekly money digest", text))) if ok]
    return "+".join(sent) or "in_app"


def digest_enabled() -> bool:
    return db.get_setting("digest_enabled", "1") == "1"


def digest_job() -> dict | None:
    """Build + deliver the digest for the active profile."""
    if not db.all_transactions():
        log.info("digest skipped: no transactions")
        return None
    if not llm.available():
        log.info("digest skipped: MESH_API_KEY missing")
        return None
    d = build_digest()
    via = deliver(d["insight"], d["action"])
    db.save_digest(d["run_id"], d["insight"], d["action"], via)
    log.info("digest delivered via %s", via)
    return {**d, "delivered_via": via}


def all_profiles_job() -> None:
    """The scheduled entry point: every profile that has digests switched on."""
    for pid in profiles.PROFILES:
        def one(pid=pid):
            profiles.activate(pid)
            if digest_enabled():
                try:
                    digest_job()
                except Exception as e:  # one profile failing must not stop the others
                    log.warning("digest failed for %s: %s", pid, e)
        contextvars.copy_context().run(one)


def start() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
        _scheduler.add_job(all_profiles_job, "cron", day_of_week=config.DIGEST_DAY_OF_WEEK, hour=config.DIGEST_HOUR,
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
