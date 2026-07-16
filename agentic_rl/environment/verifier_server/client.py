"""Worker-side HTTP client for the verifier (judge) server.

Both helpers run in the TRUSTED rollout-worker process, never inside the agent
sandbox — this is the "server-side scores" half of the reward pipeline:

  ``fetch_agent_submissions``  read back every submission an episode's AGENT_ID
                               made mid-episode, with the judge's own scores
                               (replaces trusting the sandbox submissions.jsonl).
  ``grade_solution``           grade the final solution.cpp by submitting it from
                               the worker (replaces trusting the sandbox-written
                               reward.json of the in-sandbox evaluate.py).

stdlib urllib only (like autostart.py) so the worker needs no extra deps.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_POLL_INTERVAL_SEC = 2.0


class JudgeClientError(RuntimeError):
    """The judge server could not be reached or answered unusably."""


def _get_json(url: str, *, timeout: float = 15.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
        raise JudgeClientError(f"GET {url}: {e}") from e


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 30.0) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
        raise JudgeClientError(f"POST {url}: {e}") from e


def fetch_agent_submissions(base_url: str, agent_id: str, *, timeout: float = 15.0, attempts: int = 3) -> list[dict[str, Any]]:
    """All submissions the judge attributes to ``agent_id``, as recorded
    server-side: ``[{sid, pid, ts, status, score, scoreUnbounded, passed}]``.

    Raises :class:`JudgeClientError` when the endpoint is unreachable (or the
    judge predates it — 404) so the caller can fall back explicitly.
    """
    url = f"{base_url.rstrip('/')}/agent/{urllib.parse.quote(agent_id, safe='')}/submissions"
    last: Exception | None = None
    for i in range(attempts):
        try:
            data = _get_json(url, timeout=timeout)
            subs = (data or {}).get("submissions")
            if not isinstance(subs, list):
                raise JudgeClientError(f"malformed response from {url}: {str(data)[:200]}")
            return subs
        except JudgeClientError as e:
            last = e
            if i < attempts - 1:
                time.sleep(2.0)
    raise JudgeClientError(f"fetch_agent_submissions: {last}")


def grade_solution(base_url: str, pid: str, code: str, *, agent_id: str = "", lang: str = "cpp", poll_timeout_sec: float = 600.0) -> dict[str, Any]:
    """Submit ``code`` from the worker and poll until the judge finishes.

    Returns the judge's terminal result dict (``status`` done/error, plus
    score/passed/cases on done). Raises :class:`JudgeClientError` on submit
    failure or poll timeout; the caller decides the fallback.
    """
    base = base_url.rstrip("/")
    payload: dict[str, Any] = {"pid": pid, "lang": lang, "code": code}
    if agent_id:
        payload["agent_id"] = agent_id
    # Retry the submit through a brief judge restart (the sandbox entrypoint
    # restarts node in ~1-2s) — the in-sandbox evaluate.py had a readiness loop
    # for the same reason. A resubmit after a lost response only orphans one
    # already-queued sid (extra judge work, no effect on the grade we read).
    submitted: Any = None
    last: Exception | None = None
    for i in range(3):
        try:
            submitted = _post_json(f"{base}/submit", payload)
            break
        except JudgeClientError as e:
            last = e
            if i < 2:
                time.sleep(5.0)
    if submitted is None:
        raise JudgeClientError(f"submit: {last}")
    sid = (submitted or {}).get("sid")
    if sid is None:
        raise JudgeClientError(f"submit returned no sid: {str(submitted)[:200]}")

    deadline = time.monotonic() + poll_timeout_sec
    while time.monotonic() < deadline:
        try:
            result = _get_json(f"{base}/result/{sid}", timeout=15.0)
        except JudgeClientError:
            result = None  # transient poll failure; keep trying until deadline
        if isinstance(result, dict) and result.get("status") in ("done", "error"):
            return result
        time.sleep(_POLL_INTERVAL_SEC)
    raise JudgeClientError(f"result for sid={sid} not final within {poll_timeout_sec:.0f}s")
