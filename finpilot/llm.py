"""Single gateway for every LLM call. All traffic goes through Mesh API (OpenAI-compatible)."""
import json
import re
import time

from openai import APIStatusError, OpenAI

from finpilot import config, db

_client: OpenAI | None = None
# Circuit breaker: after a billing/auth failure (401/402/403) stop calling Mesh for a while instead of
# hammering it with requests that cannot succeed; callers fall back exactly as they do for any failure.
BREAKER_SECONDS = 60
_breaker: dict = {"open_until": 0.0, "reason": ""}


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    return bool(config.MESH_API_KEY)


def client() -> OpenAI:
    global _client
    if not available():
        raise LLMUnavailable("MESH_API_KEY is not set; add it to .env")
    if _client is None:
        _client = OpenAI(api_key=config.MESH_API_KEY, base_url=config.MESH_BASE_URL)
    return _client


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_json(system: str, user: str, *, agent: str, run_id: int | None = None, model: str | None = None,
              temperature: float = 0.0, retries: int = 2) -> dict:
    """Call the model, require a JSON object back, and record the call in the decision trail."""
    model = model or config.MODEL_FAST
    if time.monotonic() < _breaker["open_until"]:
        db.log_decision(run_id, agent, "llm_error", {"error": f"circuit open: {_breaker['reason']}"}, model=model)
        raise RuntimeError(f"Mesh API circuit open for {agent}: {_breaker['reason']}")
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            started = time.perf_counter()
            resp = client().chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            content = resp.choices[0].message.content or "{}"
            data = _parse_json(content)
            usage = getattr(resp, "usage", None)
            db.log_decision(run_id, agent, "llm_call", {
                "prompt_chars": len(system) + len(user),
                "response_preview": content[:600],
                "tokens": getattr(usage, "total_tokens", None),
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "attempt": attempt,
            }, model=model)
            return data
        except LLMUnavailable:
            raise
        except APIStatusError as e:
            last_err = e
            if e.status_code in (401, 402, 403):  # auth / billing: trip the breaker
                _breaker.update(open_until=time.monotonic() + BREAKER_SECONDS, reason=f"HTTP {e.status_code}")
                break
            if 400 <= e.status_code < 500 and e.status_code != 429:  # bad request: retrying won't help
                break
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # network / parse errors: retry with backoff
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    db.log_decision(run_id, agent, "llm_error", {"error": str(last_err)}, model=model)
    raise RuntimeError(f"Mesh API call failed for {agent}: {last_err}")
