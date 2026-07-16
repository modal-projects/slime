"""Server-side reward scores for Frontier-CS (the anti-reward-hacking path).

Locks the two trust boundaries introduced for the O1/O2 runs:

  * ``merge_server_submissions`` — the episode's submission scores come from the
    judge server's own records; the sandbox's agent-writable ``submissions.jsonl``
    can enrich entries but can never contribute a score (forged/edited records
    only trip the mismatch counters);
  * ``FrontierCsEnv._verify`` — the final solution.cpp is graded by the WORKER
    submitting to the judge (normalization identical to tests/evaluate.py), so
    the grade never transits the sandbox.

The merge tests are pure stdlib; the env tests import ``frontiercs`` with the
missing-heavy-deps stubbing used across this suite.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from agentic_rl.environment import rewards as rw
from agentic_rl.environment.submissions import merge_server_submissions, parse_submissions_log


def _import_with_stubs(modname: str):
    """Import ``modname``, permissively stubbing each *missing* dependency and
    retrying (same pattern as test_exec_timeout.py) so modal/minisweagent-free
    CPU envs still run these tests."""
    for _ in range(60):
        try:
            return importlib.import_module(modname)
        except ImportError as exc:
            missing = exc.name
            if not missing or missing in sys.modules:
                raise
            stub = types.ModuleType(missing)
            stub.__path__ = []
            stub.__getattr__ = lambda _name: MagicMock()  # noqa: B023
            sys.modules[missing] = stub
    return importlib.import_module(modname)


def _row(sid, score=None, status="done", passed=False, pid="p1", ts=1000):
    return {"sid": sid, "pid": pid, "ts": ts, "status": status, "score": score, "scoreUnbounded": score, "passed": passed}


def _log(*records: dict) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


# ---------------------------------------------------------------------------
# merge_server_submissions: server rows are the only score source
# ---------------------------------------------------------------------------


def test_merge_normalizes_like_the_verifier():
    # judge-native 0..100 -> clamp01(score/100), the exact final-grade formula.
    out = merge_server_submissions([_row(1, score=50.0), _row(2, score=93.905)])
    subs = out["submissions"]
    assert [s["score"] for s in subs] == [0.5, 0.93905]
    assert [s["score_raw"] for s in subs] == [50.0, 93.905]
    assert out["summary"]["best_score"] == 0.93905
    assert out["summary"]["source"] == "server"
    assert not out["summary"]["solved"]


def test_merge_clamps_performance_scored_units():
    # The rollout-13 incident: performance judges report score in unbounded
    # units (835.76 -> raw 83576 would be /100=835.76 in the log). Server-side
    # the same clamp that bounds the final grade bounds best_submitted.
    out = merge_server_submissions([_row(1, score=9200.2), _row(2, score=0.018)])
    assert out["submissions"][0]["score"] == 1.0
    assert out["submissions"][1]["score"] == pytest.approx(0.00018)
    assert out["summary"]["best_score"] == 1.0


def test_merge_solved_from_passed_or_full_score():
    out = merge_server_submissions([_row(1, score=99.0), _row(2, score=100.0), _row(3, score=40.0, passed=True)])
    assert [s["is_solved"] for s in out["submissions"]] == [False, True, True]
    assert out["summary"]["solved"] and out["summary"]["first_solved_ordinal"] == 2


def test_merge_non_done_rows_carry_no_score():
    out = merge_server_submissions([
        _row(1, score=80.0),
        _row(2, score=None, status="queued"),
        _row(3, score=None, status="error"),
        _row(4, score=55.0, status="unknown"),  # evicted result: status lost -> untrusted
    ])
    assert [s["score"] for s in out["submissions"]] == [0.8, None, None, None]
    assert out["summary"]["n"] == 4 and out["summary"]["n_scored"] == 1


def test_merge_orders_by_sid_and_enriches_from_log():
    log = parse_submissions_log(_log(
        {"submission_uuid": "u7", "status": "done", "sid": 7, "score": 0.9, "score_raw": 90.0, "code_chars": 1234},
        {"submission_uuid": "u3", "status": "done", "sid": 3, "score": 0.5, "score_raw": 50.0, "code_chars": 999},
    ))
    out = merge_server_submissions([_row(7, score=90.0), _row(3, score=50.0)], log)
    subs = out["submissions"]
    assert [s["sid"] for s in subs] == [3, 7]  # sid order == submission order
    assert subs[0]["uuid"] == "u3" and subs[0]["code_chars"] == 999
    assert subs[1]["uuid"] == "u7" and subs[1]["code_chars"] == 1234
    s = out["summary"]
    assert (s["n_log_only"], s["n_server_only"], s["n_score_mismatch"]) == (0, 0, 0)


def test_merge_forged_log_record_cannot_contribute_score():
    # The agent appends a fake perfect-score record with a sid the server never
    # saw: it must not move best_score, only trip n_log_only.
    log = parse_submissions_log(_log(
        {"submission_uuid": "real", "status": "done", "sid": 1, "score": 0.3, "score_raw": 30.0},
        {"submission_uuid": "forged", "status": "done", "sid": 999, "score": 1.0, "score_raw": 100.0},
    ))
    out = merge_server_submissions([_row(1, score=30.0)], log)
    assert out["summary"]["best_score"] == 0.3
    assert not out["summary"]["solved"]
    assert out["summary"]["n_log_only"] == 1


def test_merge_edited_log_score_trips_mismatch_but_server_wins():
    log = parse_submissions_log(_log(
        {"submission_uuid": "a", "status": "done", "sid": 5, "score": 1.0, "score_raw": 100.0},  # edited up
    ))
    out = merge_server_submissions([_row(5, score=20.0)], log)
    assert out["summary"]["best_score"] == 0.2
    assert out["summary"]["n_score_mismatch"] == 1


def test_merge_server_only_counted():
    # Agent leg died mid-poll (or truncated its log): the server still saw the sid.
    out = merge_server_submissions([_row(1, score=60.0)], parse_submissions_log(""))
    assert out["summary"]["n_server_only"] == 1
    assert out["summary"]["best_score"] == 0.6


def test_merge_feeds_outcome_shaping_unchanged():
    out = merge_server_submissions([_row(1, score=40.0), _row(2, score=93.905)])
    o = rw.episode_outcome_from_artifacts(0.2, False, {"submissions": out["submissions"], "submission_summary": out["summary"]})
    assert o.best_submitted == 0.93905
    assert o.n_invalid_scores == 0  # pre-normalized server scores pass the [0,1] filter
    reward, _ = rw.shape_outcome(o, "best")
    assert reward == 0.93905


def test_merge_truncates_but_summary_spans_all():
    rows = [_row(i, score=float(i)) for i in range(1, 60)]
    out = merge_server_submissions(rows, max_records=50)
    assert len(out["submissions"]) == 50
    assert out["summary"]["n"] == 59 and out["summary"]["truncated"]
    assert out["summary"]["best_score"] == 0.59


# ---------------------------------------------------------------------------
# FrontierCsEnv: worker-side final grading + server-side collection
# ---------------------------------------------------------------------------

try:
    fcs_mod = _import_with_stubs("agentic_rl.environment.frontiercs")
except Exception as exc:  # pragma: no cover - unsatisfiable import env
    pytest.skip(f"frontiercs unimportable: {exc}", allow_module_level=True)


class _FakeSb:
    def __init__(self, files: dict[str, str]):
        self.files = files

    def read_file(self, path: str) -> str:
        return self.files.get(path, "")


def _verify_kwargs(**over):
    kw = {
        "tests_dir": None,
        "workdir": "/app",
        "verifier": {"env": {"PROBLEM_ID": "p1", "AGENT_ID": "ep1"}},
        "eval_timeout_sec": 60,
        "instance_id": "t-1",
    }
    kw.update(over)
    return kw


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("FRONTIER_CS_JUDGE_URL", "http://judge.test")
    monkeypatch.delenv(fcs_mod.SERVER_VERIFY_ENV, raising=False)
    return fcs_mod.FrontierCsEnv()


def test_verify_grades_from_worker(env, monkeypatch):
    seen = {}

    def fake_grade(url, pid, code, *, agent_id="", lang="cpp", poll_timeout_sec=600.0):
        seen.update(url=url, pid=pid, code=code, agent_id=agent_id, timeout=poll_timeout_sec)
        return {"status": "done", "passed": False, "score": 93.905,
                "cases": [{"scoreRatio": 1.0}, {"scoreRatio": 0.5}]}

    monkeypatch.setattr(fcs_mod.judge_client, "grade_solution", fake_grade)
    sb = _FakeSb({"/app/solution.cpp": "int main(){}"})
    rewards = env._verify(sb, **_verify_kwargs())
    assert rewards == {"reward": 0.93905, "score_raw": 93.905, "cases_passed": 1,
                       "cases_total": 2, "is_solved": False, "graded_from": "server"}
    assert seen["agent_id"] == "ep1:final" and seen["pid"] == "p1" and seen["timeout"] == 60
    # the downstream signal parse must agree
    sig = rw.signal_from_reward_dict(rewards)
    assert sig.fraction == pytest.approx(0.93905) and not sig.is_solved


def test_verify_solved_on_full_score(env, monkeypatch):
    monkeypatch.setattr(fcs_mod.judge_client, "grade_solution",
                        lambda *a, **k: {"status": "done", "passed": True, "score": 100.0, "cases": []})
    rewards = env._verify(_FakeSb({"/app/solution.cpp": "x"}), **_verify_kwargs())
    assert rewards["reward"] == 1.0 and rewards["is_solved"]


def test_verify_missing_solution_scores_zero(env, monkeypatch):
    monkeypatch.setattr(fcs_mod.judge_client, "grade_solution",
                        lambda *a, **k: pytest.fail("must not submit an empty solution"))
    rewards = env._verify(_FakeSb({}), **_verify_kwargs())
    assert rewards["reward"] == 0.0 and rewards["detail"] == "no solution.cpp"


def test_verify_judge_error_scores_zero_without_sandbox_fallback(env, monkeypatch):
    def boom(*a, **k):
        raise fcs_mod.judge_client.JudgeClientError("judge down")

    monkeypatch.setattr(fcs_mod.judge_client, "grade_solution", boom)
    monkeypatch.setattr(fcs_mod.HarborEnv, "_verify",
                        lambda *a, **k: pytest.fail("must not fall back into the sandbox on judge failure"))
    rewards = env._verify(_FakeSb({"/app/solution.cpp": "x"}), **_verify_kwargs())
    assert rewards["reward"] == 0.0 and "judge down" in rewards["detail"]


def test_verify_switch_off_uses_legacy_path(env, monkeypatch):
    monkeypatch.setenv(fcs_mod.SERVER_VERIFY_ENV, "0")
    sentinel = {"reward": 0.42}
    monkeypatch.setattr(fcs_mod.HarborEnv, "_verify", lambda *a, **k: sentinel)
    assert env._verify(_FakeSb({"/app/solution.cpp": "x"}), **_verify_kwargs()) is sentinel


def test_collect_artifacts_prefers_server(env, monkeypatch):
    log_text = _log({"submission_uuid": "u", "status": "done", "sid": 1, "score": 1.0, "score_raw": 100.0})
    monkeypatch.setattr(fcs_mod.judge_client, "fetch_agent_submissions",
                        lambda url, aid, **k: [_row(1, score=30.0)])
    md = {"judge_url": "http://judge.test", "judge_agent_id": "ep1", "instance_id": "t-1"}
    art = env._collect_artifacts(_FakeSb({fcs_mod.SUBMISSIONS_LOG: log_text}), "/app", md)
    assert art["submission_summary"]["source"] == "server"
    assert art["submission_summary"]["best_score"] == 0.3  # server wins over the edited log
    assert art["submission_summary"]["n_score_mismatch"] == 1


def test_collect_artifacts_falls_back_to_log_when_server_unreachable(env, monkeypatch):
    def boom(url, aid, **k):
        raise fcs_mod.judge_client.JudgeClientError("404")

    monkeypatch.setattr(fcs_mod.judge_client, "fetch_agent_submissions", boom)
    log_text = _log({"submission_uuid": "u", "status": "done", "sid": 1, "score": 0.5, "score_raw": 50.0})
    md = {"judge_url": "http://judge.test", "judge_agent_id": "ep1", "instance_id": "t-1"}
    art = env._collect_artifacts(_FakeSb({fcs_mod.SUBMISSIONS_LOG: log_text}), "/app", md)
    assert art["submission_summary"]["source"] == "agent_log"
    assert art["submission_summary"]["best_score"] == 0.5


def test_collect_artifacts_empty_everywhere_returns_nothing(env, monkeypatch):
    monkeypatch.setattr(fcs_mod.judge_client, "fetch_agent_submissions", lambda url, aid, **k: [])
    md = {"judge_url": "http://judge.test", "judge_agent_id": "ep1", "instance_id": "t-1"}
    assert env._collect_artifacts(_FakeSb({}), "/app", md) == {}
