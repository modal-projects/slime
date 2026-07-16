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


def merge_server_submissions(
    server_rows: list[dict[str, Any]],
    log_parsed: dict[str, Any] | None = None,
    *,
    max_records: int = 50,
) -> dict[str, Any]:
    """Build the episode's submission artifacts from the JUDGE SERVER's records
    (``GET /agent/:id/submissions``), so scores come from the server and never
    from the sandbox-writable ``submissions.jsonl``.

    ``server_rows``: ``[{sid, pid, ts, status, score, scoreUnbounded, passed}]``
    with ``score`` in the judge's native 0..100 units. Each row becomes one
    entry with ``score`` normalized exactly like the verifier's final grade
    (``clamp01(score/100)``) so "best" and "final" share units by construction.

    ``log_parsed`` (``parse_submissions_log`` output) is used only to ENRICH
    matching entries (join by sid: uuid/code_chars) and to feed the forgery
    tripwire counters in the summary:

      ``n_log_only``       sandbox-log records carrying a sid the server never
                           saw for this agent (forged, or submitted without
                           AGENT_ID — either way they contribute no score)
      ``n_server_only``    server sids missing from the sandbox log (log
                           tampering/truncation, or the agent leg died mid-poll)
      ``n_score_mismatch`` joined records whose logged ``score_raw`` disagrees
                           with the server's score (log edited after the fact)

    Output schema matches ``parse_submissions_log`` (entries with ordinal/
    status/score/score_raw/is_solved; summary with n/best_score/solved/...)
    plus ``summary["source"] = "server"``, so the downstream outcome shaping
    (``rewards.episode_outcome_from_artifacts``) consumes it unchanged.
    """
    log_by_sid: dict[int, dict[str, Any]] = {}
    log_sids: set[int] = set()
    for e in (log_parsed or {}).get("submissions") or []:
        sid = _as_int(e.get("sid"))
        if sid is not None:
            log_sids.add(sid)
            log_by_sid.setdefault(sid, e)

    rows = sorted((r for r in server_rows or [] if _as_int(r.get("sid")) is not None), key=lambda r: _as_int(r["sid"]))
    submissions: list[dict[str, Any]] = []
    n_score_mismatch = 0
    for i, row in enumerate(rows, start=1):
        sid = _as_int(row["sid"])
        raw = _as_float(row.get("score"))
        done = row.get("status") == "done"
        entry: dict[str, Any] = {
            "ordinal": i,
            "sid": sid,
            "status": row.get("status") or "unknown",
            "score": max(0.0, min(1.0, raw / 100.0)) if (done and raw is not None) else None,
            "score_raw": raw if done else None,
            "is_solved": bool(done and (row.get("passed") or (raw is not None and raw >= 100.0 - _SOLVED_EPS))),
            "ts": row.get("ts"),
        }
        logged = log_by_sid.get(sid)
        if logged is not None:
            entry["uuid"] = logged.get("uuid")
            if logged.get("code_chars") is not None:
                entry["code_chars"] = logged.get("code_chars")
            logged_raw = _as_float(logged.get("score_raw"))
            if done and raw is not None and logged_raw is not None and abs(logged_raw - raw) > 1e-4 * max(1.0, abs(raw)):
                n_score_mismatch += 1
        submissions.append(entry)

    server_sids = {s["sid"] for s in submissions}
    scored = [s for s in submissions if s["score"] is not None]
    summary: dict[str, Any] = {
        "n": len(submissions),
        "n_scored": len(scored),
        "best_score": max((s["score"] for s in scored), default=None),
        "final_score": scored[-1]["score"] if scored else None,
        "solved": any(s["is_solved"] for s in submissions),
        "first_solved_ordinal": next((s["ordinal"] for s in submissions if s["is_solved"]), None),
        "source": "server",
        "n_log_only": len(log_sids - server_sids),
        "n_server_only": len(server_sids - log_sids),
        "n_score_mismatch": n_score_mismatch,
    }
    if len(submissions) > max_records:
        submissions = submissions[-max_records:]
        summary["truncated"] = True
    return {"submissions": submissions, "summary": summary}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
