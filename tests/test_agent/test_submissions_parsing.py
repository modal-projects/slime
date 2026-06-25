"""Unit tests for the Frontier-CS iterative-submission log parser.

Pure stdlib target (``agentic_rl.environment.submissions``) — no modal/slime
imports — so this runs without a GPU/sandbox.
"""

import json

from agentic_rl.environment.submissions import parse_submissions_log


def _log(*records: dict) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def test_pairs_started_and_done_by_uuid():
    text = _log(
        {"submission_uuid": "a", "ts": "t1", "status": "started", "problem_id": "p", "code_chars": 0},
        {
            "submission_uuid": "a",
            "ts": "t2",
            "status": "done",
            "sid": 1,
            "score": 0.5,
            "score_raw": 50.0,
            "code_chars": 1200,
        },
    )
    out = parse_submissions_log(text)
    assert len(out["submissions"]) == 1
    s = out["submissions"][0]
    assert s["ordinal"] == 1 and s["status"] == "done" and s["sid"] == 1
    assert s["score"] == 0.5 and s["score_raw"] == 50.0 and s["code_chars"] == 1200
    assert s["is_solved"] is False
    assert s["ts_started"] == "t1" and s["ts_done"] == "t2"
    assert out["summary"]["best_score"] == 0.5 and out["summary"]["final_score"] == 0.5
    assert out["summary"]["solved"] is False


def test_summary_best_final_and_first_solved_ordinal():
    text = _log(
        {"submission_uuid": "a", "status": "done", "score": 0.4, "score_raw": 40.0},
        {"submission_uuid": "b", "status": "done", "score": 1.0, "score_raw": 100.0},
        {"submission_uuid": "c", "status": "done", "score": 0.7, "score_raw": 70.0},
    )
    out = parse_submissions_log(text)
    assert [s["ordinal"] for s in out["submissions"]] == [1, 2, 3]
    assert out["submissions"][1]["is_solved"] is True
    assert out["summary"]["best_score"] == 1.0
    assert out["summary"]["final_score"] == 0.7  # final = last *scored* attempt, not best
    assert out["summary"]["solved"] is True
    assert out["summary"]["first_solved_ordinal"] == 2


def test_error_record_kept_without_score():
    text = _log(
        {"submission_uuid": "a", "status": "started"},
        {"submission_uuid": "a", "status": "error", "error": "compile failed: xyz"},
    )
    out = parse_submissions_log(text)
    s = out["submissions"][0]
    assert s["status"] == "error" and s["score"] is None and s["is_solved"] is False
    assert "compile failed" in s["detail"]
    assert out["summary"]["n"] == 1 and out["summary"]["n_scored"] == 0
    assert out["summary"]["best_score"] is None and out["summary"]["solved"] is False


def test_zero_score_counts_as_scored():
    out = parse_submissions_log(_log({"submission_uuid": "a", "status": "done", "score": 0.0, "score_raw": 0.0}))
    assert out["summary"]["n_scored"] == 1
    assert out["summary"]["best_score"] == 0.0 and out["summary"]["final_score"] == 0.0


def test_ignores_blank_and_malformed_lines():
    text = "\n".join(
        [
            "",
            "not json",
            "{bad}",
            json.dumps({"submission_uuid": "a", "status": "done", "score": 0.9, "score_raw": 90.0}),
            "  ",
        ]
    )
    out = parse_submissions_log(text)
    assert len(out["submissions"]) == 1 and out["summary"]["best_score"] == 0.9


def test_truncates_to_most_recent_but_summary_spans_all():
    records = [
        {"submission_uuid": f"u{i}", "status": "done", "score": i / 100.0, "score_raw": float(i)} for i in range(60)
    ]
    out = parse_submissions_log(_log(*records), max_records=50)
    assert len(out["submissions"]) == 50
    assert out["submissions"][0]["ordinal"] == 11  # kept the most recent 50 -> ordinals 11..60
    assert out["submissions"][-1]["ordinal"] == 60
    assert out["summary"]["n"] == 60 and out["summary"].get("truncated") is True
    assert out["summary"]["best_score"] == 0.59 and out["summary"]["final_score"] == 0.59


def test_detail_capped():
    long_detail = "x" * 1000
    out = parse_submissions_log(
        _log({"submission_uuid": "a", "status": "error", "error": long_detail}), max_detail=300
    )
    assert len(out["submissions"][0]["detail"]) == 300
