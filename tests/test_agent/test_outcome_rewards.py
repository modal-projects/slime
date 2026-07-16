"""Episode-level outcome-reward interface (``agentic_rl.environment.rewards``).

Pure stdlib target — no modal/slime/torch imports — so this runs without a
GPU/sandbox. Locks the contracts the reward-shaping ablations depend on:

  * defaults (strategy "final", bonus 0) are bit-identical to the old reward;
  * "best" = max(final, best submitted), telescoping target of per-turn deltas;
  * "disc_sum" = Kevin-style discounted sum over the submission sequence;
  * no submissions / no artifacts / SWE-style episodes degrade to final;
  * env switches (ASYNC_RL_OUTCOME_REWARD / ASYNC_RL_SOLVED_BONUS /
    ASYNC_RL_OUTCOME_GAMMA) and metadata.outcome_reward per-row override.
"""

import pytest

from agentic_rl.environment import rewards as rw


def _outcome(final=0.5, is_solved=False, scores=(), summary_best=None, submitted_solved=False):
    return rw.EpisodeOutcome(
        final=final,
        is_solved=is_solved,
        submission_scores=tuple(scores),
        summary_best=summary_best,
        submitted_solved=submitted_solved,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (rw.OUTCOME_ENV, rw.SOLVED_BONUS_ENV, rw.OUTCOME_GAMMA_ENV):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# default path == current behavior
# ---------------------------------------------------------------------------


def test_default_is_identity_on_final():
    for final in (0.0, 0.33, 1.0):
        reward, info = rw.shape_outcome(_outcome(final=final, scores=(0.9, 0.95)))
        assert reward == final
        assert info["strategy"] == "final" and info["bonus"] == 0.0


def test_resolve_outcome_defaults_and_fallback(monkeypatch):
    assert rw.resolve_outcome({}) == "final"
    monkeypatch.setenv(rw.OUTCOME_ENV, "best")
    assert rw.resolve_outcome({}) == "best"
    # per-row metadata beats the env
    assert rw.resolve_outcome({"outcome_reward": "disc_sum"}) == "disc_sum"
    monkeypatch.setenv(rw.OUTCOME_ENV, "not_a_strategy")
    assert rw.resolve_outcome({}) == "final"


# ---------------------------------------------------------------------------
# strategy: best (plan A / outcome-collapsed plan B)
# ---------------------------------------------------------------------------


def test_best_takes_max_of_final_and_submissions():
    reward, info = rw.shape_outcome(_outcome(final=0.2, scores=(0.1, 0.9, 0.4)), "best")
    assert reward == 0.9
    assert info["best_submitted"] == 0.9


def test_best_keeps_final_when_final_wins():
    reward, _ = rw.shape_outcome(_outcome(final=0.95, scores=(0.1, 0.9)), "best")
    assert reward == 0.95


def test_best_without_submissions_degrades_to_final():
    reward, info = rw.shape_outcome(_outcome(final=0.42), "best")
    assert reward == 0.42
    assert info["best_submitted"] is None


def test_best_prefers_summary_best_over_truncated_list():
    # the submissions list truncates to the most recent 50; summary_best spans all
    reward, _ = rw.shape_outcome(_outcome(final=0.1, scores=(0.2, 0.3), summary_best=0.7), "best")
    assert reward == 0.7


# ---------------------------------------------------------------------------
# strategy: disc_sum (outcome-collapsed plan C, Kevin sum w/ gamma=0.4)
# ---------------------------------------------------------------------------


def test_disc_sum_kevin_default_gamma():
    scores = (0.5, 1.0, 0.25)
    reward, _ = rw.shape_outcome(_outcome(final=0.9, scores=scores), "disc_sum")
    assert reward == pytest.approx(0.5 + 0.4 * 1.0 + 0.16 * 0.25)


def test_disc_sum_gamma_env_override(monkeypatch):
    monkeypatch.setenv(rw.OUTCOME_GAMMA_ENV, "0.95")
    reward, _ = rw.shape_outcome(_outcome(final=0.0, scores=(0.5, 1.0)), "disc_sum")
    assert reward == pytest.approx(0.5 + 0.95)


def test_disc_sum_without_submissions_degrades_to_final():
    reward, _ = rw.shape_outcome(_outcome(final=0.42), "disc_sum")
    assert reward == 0.42


# ---------------------------------------------------------------------------
# solved bonus (Kevin's 0.3 * 1{correct})
# ---------------------------------------------------------------------------


def test_bonus_applies_on_solved_final(monkeypatch):
    monkeypatch.setenv(rw.SOLVED_BONUS_ENV, "0.3")
    reward, info = rw.shape_outcome(_outcome(final=1.0, is_solved=True))
    assert reward == pytest.approx(1.3)
    assert info["bonus"] == pytest.approx(0.3)


def test_bonus_off_by_default_even_when_solved():
    reward, info = rw.shape_outcome(_outcome(final=1.0, is_solved=True))
    assert reward == 1.0 and info["bonus"] == 0.0


def test_bonus_not_applied_when_unsolved(monkeypatch):
    monkeypatch.setenv(rw.SOLVED_BONUS_ENV, "0.3")
    reward, info = rw.shape_outcome(_outcome(final=0.9, is_solved=False))
    assert reward == pytest.approx(0.9) and info["bonus"] == 0.0


def test_bonus_solved_semantics_per_strategy(monkeypatch):
    # A solved mid-episode submission counts under "best" (which trains on it)
    # but not under "final" (verifier-only semantics).
    monkeypatch.setenv(rw.SOLVED_BONUS_ENV, "0.3")
    o = _outcome(final=0.5, is_solved=False, scores=(1.0,), submitted_solved=True)
    reward_final, _ = rw.shape_outcome(o, "final")
    reward_best, _ = rw.shape_outcome(o, "best")
    assert reward_final == pytest.approx(0.5)
    assert reward_best == pytest.approx(1.0 + 0.3)


# ---------------------------------------------------------------------------
# artifacts -> EpisodeOutcome (the harbor.py join)
# ---------------------------------------------------------------------------


def test_from_artifacts_full():
    artifacts = {
        "submissions": [
            {"ordinal": 1, "score": 0.5, "status": "done"},
            {"ordinal": 2, "score": None, "status": "error"},  # unscored attempt skipped
            {"ordinal": 3, "score": 0.93905, "status": "done"},
        ],
        "submission_summary": {"n": 3, "best_score": 0.93905, "final_score": 0.93905, "solved": False},
    }
    o = rw.episode_outcome_from_artifacts(0.2, False, artifacts)
    assert o.submission_scores == (0.5, 0.93905)
    assert o.best_submitted == 0.93905
    assert not o.submitted_solved
    reward, _ = rw.shape_outcome(o, "best")
    assert reward == 0.93905


def test_from_artifacts_empty_and_none():
    for artifacts in ({}, None, {"submissions": [], "submission_summary": {}}):
        o = rw.episode_outcome_from_artifacts(0.7, True, artifacts)
        assert o.submission_scores == ()
        assert o.best_submitted is None
        for strategy in rw.OUTCOMES:
            reward, _ = rw.shape_outcome(o, strategy)
            assert reward == 0.7, f"{strategy} must degrade to final without submissions"


def test_from_artifacts_solved_submission():
    artifacts = {
        "submissions": [{"ordinal": 1, "score": 1.0, "status": "done"}],
        "submission_summary": {"n": 1, "best_score": 1.0, "solved": True},
    }
    o = rw.episode_outcome_from_artifacts(0.0, False, artifacts)
    assert o.submitted_solved
    assert o.n_invalid_scores == 0
    reward, _ = rw.shape_outcome(o, "best")
    assert reward == 1.0


# ---------------------------------------------------------------------------
# out-of-range submission scores (performance-scored problems)
# ---------------------------------------------------------------------------


def test_from_artifacts_discards_out_of_range_scores():
    # Observed in the o1-best run (rollout 13): performance-scored problems
    # report `score` in unbounded judge units (92.0, 835.76) while the
    # verifier's clamped final is ~0 -- "best" must not train on them.
    artifacts = {
        "submissions": [
            {"ordinal": 1, "score": 0.4, "status": "done"},
            {"ordinal": 2, "score": 92.00208333333333, "status": "done"},
            {"ordinal": 3, "score": 835.76, "status": "done"},
            {"ordinal": 4, "score": -0.5, "status": "done"},
        ],
        "submission_summary": {"n": 4, "best_score": 835.76, "solved": True},
    }
    o = rw.episode_outcome_from_artifacts(0.2, False, artifacts)
    assert o.submission_scores == (0.4,)
    assert o.summary_best is None
    assert o.n_invalid_scores == 4  # 3 entries + the out-of-range summary best
    # summary "solved" derives from score_raw >= 100, meaningless in these
    # units -> recomputed from the surviving valid scores (none are 1.0).
    assert not o.submitted_solved
    reward, info = rw.shape_outcome(o, "best")
    assert reward == 0.4
    assert info["invalid_scores"] == 4


def test_from_artifacts_all_scores_invalid_degrades_to_final():
    artifacts = {
        "submissions": [{"ordinal": 1, "score": 835.76, "status": "done"}],
        "submission_summary": {"n": 1, "best_score": 835.76, "solved": True},
    }
    o = rw.episode_outcome_from_artifacts(0.3, False, artifacts)
    assert o.best_submitted is None
    for strategy in rw.OUTCOMES:
        reward, _ = rw.shape_outcome(o, strategy)
        assert reward == 0.3


def test_from_artifacts_clamps_float_overshoot():
    artifacts = {
        "submissions": [{"ordinal": 1, "score": 1.0000000001, "status": "done"}],
        "submission_summary": {"n": 1, "best_score": 1.0000000001, "solved": True},
    }
    o = rw.episode_outcome_from_artifacts(0.0, False, artifacts)
    assert o.submission_scores == (1.0,)
    assert o.summary_best == 1.0
    assert o.n_invalid_scores == 0
    assert o.submitted_solved
