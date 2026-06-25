"""Parse the Frontier-CS iterative-submission log (``/logs/agent/submissions.jsonl``).

The dataset's ``submit.py`` appends one ``started`` record and one terminal
(``done``/``error``) record per agent submission, keyed by ``submission_uuid``.
The judge grades only the FINAL ``solution.cpp`` for reward, so these are the
agent's *intermediate* attempts — written into the sandbox on every Frontier-CS
rollout and otherwise discarded at teardown.

``parse_submissions_log`` collapses the raw JSONL into one ordered entry per
attempt plus a summary, so an episode can persist a score-vs-attempt trace into
``sample.metadata`` (analysis now). Each entry keeps its ``ordinal``, which aligns
1:1 with the agent's submit tool-calls in the recorded trajectory — the join the
dataset's ``submit.py`` docstring describes — so per-turn / per-weight-version
reward shaping can be wired on top later.

Pure stdlib (no modal/slime imports) so it stays unit-testable on its own.
"""

from __future__ import annotations

import json
from typing import Any

_SOLVED_EPS = 1e-6
_TERMINAL = ("done", "error")
_SCALAR_KEYS = ("score", "score_raw", "code_chars", "sid")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_submissions_log(text: str, *, max_records: int = 50, max_detail: int = 300) -> dict[str, Any]:
    """Collapse ``submissions.jsonl`` text into ``{"submissions": [...], "summary": {...}}``.

    One entry per ``submission_uuid`` in first-seen order, each carrying the
    terminal status/score (``done``/``error`` wins over ``started``). The summary
    spans ALL attempts (best/final/solved) and is computed before the list is
    truncated to the most recent ``max_records``.
    """
    by_uuid: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(rec, dict):
            continue

        uid = str(rec.get("submission_uuid") or rec.get("sid") or f"_line{len(order)}")
        entry = by_uuid.get(uid)
        if entry is None:
            entry = {
                "uuid": uid,
                "status": None,
                "score": None,
                "score_raw": None,
                "code_chars": None,
                "sid": None,
                "detail": "",
                "ts_started": None,
                "ts_done": None,
            }
            by_uuid[uid] = entry
            order.append(uid)

        status = rec.get("status")
        if status == "started":
            entry["ts_started"] = rec.get("ts") or entry["ts_started"]
        elif rec.get("ts"):
            entry["ts_done"] = rec["ts"]
        if status and (entry["status"] is None or status in _TERMINAL):
            entry["status"] = status
        for key in _SCALAR_KEYS:
            if rec.get(key) is not None:
                entry[key] = rec[key]
        detail = rec.get("detail") or rec.get("error")
        if detail:
            entry["detail"] = str(detail)[:max_detail]

    submissions: list[dict[str, Any]] = []
    for i, uid in enumerate(order, start=1):
        entry = by_uuid[uid]
        raw = _as_float(entry["score_raw"])
        entry["ordinal"] = i
        entry["is_solved"] = bool(raw is not None and raw >= 100.0 - _SOLVED_EPS)
        submissions.append(entry)

    scored = [s for s in submissions if _as_float(s["score"]) is not None]
    summary: dict[str, Any] = {
        "n": len(submissions),
        "n_scored": len(scored),
        "best_score": max((_as_float(s["score"]) for s in scored), default=None),
        "final_score": _as_float(scored[-1]["score"]) if scored else None,
        "solved": any(s["is_solved"] for s in submissions),
        "first_solved_ordinal": next((s["ordinal"] for s in submissions if s["is_solved"]), None),
    }

    if len(submissions) > max_records:
        submissions = submissions[-max_records:]
        summary["truncated"] = True
    return {"submissions": submissions, "summary": summary}
