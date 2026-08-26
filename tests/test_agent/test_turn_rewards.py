"""Turn-level reward shaping (O3): the ``next_sub`` rule, the rollout-side
reward post-process hook, the train-side advantage painting, and the turn-span
recording in RecordingModel.

CPU-only. The rollout hook is exercised with real ``slime.utils.types.Sample``
objects and a SimpleNamespace args (same knobs the stock ``_post_process_rewards``
reads); the train hook with hand-built rollout_data (kl/rewards/metadata), so
the painted advantages can be checked token-by-token against the builtin GRPO
broadcast.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from agentic_rl import turn_advantage as ta
from agentic_rl import turn_reward as tr
from slime.utils.types import Sample

# ---------------------------------------------------------------------------
# next_sub rule
# ---------------------------------------------------------------------------

T0 = 1_700_000_000.0  # epoch base so ms-vs-sec coercion is exercised for real


def _sub(ts_sec: float | None, score, **extra) -> dict:
    e = {"score": score, **extra}
    if ts_sec is not None:
        e["ts"] = int(ts_sec * 1000)  # judge rows carry Date.now() ms
    return e


def test_next_sub_basic_alignment():
    # turns end at +10, +100, +300; submissions at +50 (0.3) and +200 (0.6)
    turn_ts = [T0 + 10, T0 + 100, T0 + 300]
    subs = [_sub(T0 + 50, 0.3), _sub(T0 + 200, 0.6)]
    assert tr.turn_rewards_next_sub(turn_ts, subs) == [0.3, 0.6, 0.6]


def test_next_sub_own_turn_submission_counts():
    # A submission fired BY turn t (ts just after its generation end) credits turn t.
    assert tr.turn_rewards_next_sub([T0 + 10], [_sub(T0 + 11, 0.8)]) == [0.8]


def test_next_sub_no_scored_submissions_is_none():
    assert tr.turn_rewards_next_sub([T0 + 10], []) is None
    assert tr.turn_rewards_next_sub([T0 + 10], [{"score": None, "ts": int((T0 + 50) * 1000)}]) is None


def test_next_sub_discards_out_of_range_scores():
    # Judge-unit leakage (92, 835.76) must not become a turn reward.
    subs = [_sub(T0 + 50, 92.0), _sub(T0 + 200, 0.4)]
    assert tr.turn_rewards_next_sub([T0 + 10, T0 + 300], subs) == [0.4, 0.4]


def test_next_sub_iso_timestamp_fallback():
    # Sandbox-log entries carry ISO-8601 ts_started instead of ms ts.
    from datetime import datetime, timezone

    iso = datetime.fromtimestamp(T0 + 50, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    subs = [{"score": 0.7, "ts_started": iso}]
    assert tr.turn_rewards_next_sub([T0 + 10, T0 + 100], subs) == [0.7, 0.7]


def test_next_sub_untimed_submission_only_feeds_last_fallback():
    # No usable ts -> can't align as "next", but still serves as the episode's
    # last submission for turns past every timed one.
    subs = [_sub(T0 + 50, 0.2), {"score": 0.9}]
    assert tr.turn_rewards_next_sub([T0 + 10, T0 + 100], subs) == [0.2, 0.9]


def test_compute_turn_rewards_requires_spans_and_ts():
    agentic = {"turn_ts": [T0], "submissions": [_sub(T0 + 1, 0.5)]}
    assert tr.compute_turn_rewards(agentic, "next_sub") is None  # no spans
    agentic = {"turn_spans": [[0, 4], [8, 12]], "turn_ts": [T0], "submissions": [_sub(T0 + 1, 0.5)]}
    assert tr.compute_turn_rewards(agentic, "next_sub") is None  # length mismatch
    agentic = {"turn_spans": [[0, 4]], "turn_ts": [T0], "submissions": [_sub(T0 + 1, 0.5)]}
    assert tr.compute_turn_rewards(agentic, "next_sub") == ([[0, 4]], [0.5])
    assert tr.compute_turn_rewards(agentic, "bogus") is None


# ---------------------------------------------------------------------------
# rollout-side hook (post_process_rewards)
# ---------------------------------------------------------------------------


def _args(**over):
    base = dict(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _sample(reward, group_index, agentic=None):
    md = {"agentic": agentic} if agentic is not None else {}
    return Sample(reward=reward, group_index=group_index, metadata=md)


def _agentic(turn_ts, subs, spans=None):
    spans = spans or [[i * 10, i * 10 + 5] for i in range(len(turn_ts))]
    return {"turn_spans": spans, "turn_ts": turn_ts, "submissions": subs}


def test_hook_off_matches_stock_normalization(monkeypatch):
    monkeypatch.delenv(tr.TURN_REWARD_ENV, raising=False)
    samples = [_sample(r, g) for g, rs in enumerate([(0.0, 1.0), (0.5, 0.5)]) for r in rs]
    raw, rewards = tr.post_process_rewards(_args(), samples)
    assert raw == [0.0, 1.0, 0.5, 0.5]
    # group 1: centered +- 0.5 then / (std + 1e-6); group 2: zero-std -> 0
    std = torch.tensor([-0.5, 0.5]).std().item()
    assert rewards[0] == pytest.approx(-0.5 / (std + 1e-6))
    assert rewards[1] == pytest.approx(0.5 / (std + 1e-6))
    assert rewards[2] == rewards[3] == pytest.approx(0.0)
    assert all(s.train_metadata is None for s in samples)  # untouched when off


def test_hook_ships_normalized_turn_metadata(monkeypatch):
    monkeypatch.setenv(tr.TURN_REWARD_ENV, "next_sub")
    # group 0: one submitting sample (2 turns), one silent; group 1: two submitting.
    s0 = _sample(0.4, 0, _agentic([T0 + 10, T0 + 100], [_sub(T0 + 50, 0.2), _sub(T0 + 150, 0.8)]))
    s1 = _sample(0.0, 0, _agentic([T0 + 10], []))  # no scored subs -> no turn component
    s2 = _sample(0.6, 1, _agentic([T0 + 10], [_sub(T0 + 50, 1.0)]))
    s3 = _sample(0.2, 1, _agentic([T0 + 10], [_sub(T0 + 50, 0.5)]))
    samples = [s0, s1, s2, s3]

    raw, rewards = tr.post_process_rewards(_args(), samples)
    assert raw == [0.4, 0.0, 0.6, 0.2]  # outcome scalars untouched by turn shaping

    # every sample carries train_metadata (the emit gate reads samples[0])
    assert all(s.train_metadata is not None for s in samples)
    assert "turn_spans" not in s1.train_metadata  # silent sample ships no turn keys

    # group 0 pool = s0's raw turn rewards [0.2, 0.8]: centered +-0.3, std-scaled
    md0 = s0.train_metadata
    assert md0["turn_rewards"] == [0.2, 0.8]
    std0 = torch.tensor([0.2, 0.8]).std().item()
    assert md0["turn_adv"][0] == pytest.approx(-0.3 / (std0 + 1e-6))
    assert md0["turn_adv"][1] == pytest.approx(0.3 / (std0 + 1e-6))
    assert md0["turn_spans"] == [[0, 5], [10, 15]]

    # group 1 pool = [1.0, 0.5] across the two samples
    pooled = torch.tensor([1.0, 0.5])
    std1 = pooled.std().item()
    assert s2.train_metadata["turn_adv"][0] == pytest.approx(0.25 / (std1 + 1e-6))
    assert s3.train_metadata["turn_adv"][0] == pytest.approx(-0.25 / (std1 + 1e-6))

    # each group's pooled normalized values are zero-mean
    for group in ([md0["turn_adv"]], [s2.train_metadata["turn_adv"], s3.train_metadata["turn_adv"]]):
        vals = [v for adv in group for v in adv]
        assert math.isclose(sum(vals) / len(vals), 0.0, abs_tol=1e-6)


def test_hook_zero_variance_turn_pool_ships_zeros(monkeypatch):
    monkeypatch.setenv(tr.TURN_REWARD_ENV, "next_sub")
    s0 = _sample(1.0, 0, _agentic([T0 + 10], [_sub(T0 + 50, 0.5)]))
    s1 = _sample(0.0, 0, _agentic([T0 + 10], [_sub(T0 + 50, 0.5)]))
    tr.post_process_rewards(_args(), [s0, s1])
    assert s0.train_metadata["turn_adv"] == [0.0]
    assert s1.train_metadata["turn_adv"] == [0.0]


def test_hook_null_sample_degrades(monkeypatch):
    monkeypatch.setenv(tr.TURN_REWARD_ENV, "next_sub")
    s0 = _sample(0.4, 0, _agentic([T0 + 10], [_sub(T0 + 50, 0.7)]))
    s1 = _sample(0.0, 0, {"exit_status": "ImageUnusable", "turns": 0})  # _ship_null shape
    raw, rewards = tr.post_process_rewards(_args(), [s0, s1])
    assert s1.train_metadata == {}
    assert "turn_adv" in s0.train_metadata


# ---------------------------------------------------------------------------
# train-side advantage painting (compute_turn_advantages)
# ---------------------------------------------------------------------------


def _rollout_data(kl_lens, rewards, metadata, response_lengths=None):
    kl = [torch.zeros(n) for n in kl_lens]
    response_lengths = response_lengths or list(kl_lens)
    return {
        "kl": kl,
        "rewards": rewards,
        "metadata": metadata,
        "response_lengths": response_lengths,
        "total_lengths": [n + 7 for n in response_lengths],  # arbitrary prompt len
    }


def test_advantages_paint_spans_and_broadcast_elsewhere(monkeypatch):
    monkeypatch.setenv(ta.TURN_MIX_ENV, "0.3")
    md = {"turn_spans": [[0, 3], [5, 8]], "turn_adv": [-1.0, 1.0]}
    data = _rollout_data([10], rewards=[0.5], metadata=[md])
    ta.compute_turn_advantages(SimpleNamespace(), data)
    adv = data["advantages"][0]
    expected = torch.full((10,), 0.5)
    expected[0:3] += 0.3 * -1.0
    expected[5:8] += 0.3 * 1.0
    assert torch.allclose(adv, expected)
    assert torch.equal(data["returns"][0], adv)


def test_advantages_without_metadata_match_builtin_grpo(monkeypatch):
    monkeypatch.delenv(ta.TURN_MIX_ENV, raising=False)
    data = _rollout_data([4, 6], rewards=[0.25, -1.5], metadata=None)
    ta.compute_turn_advantages(SimpleNamespace(), data)
    # bit-identical to get_grpo_returns: reward broadcast over kl-shaped ones
    assert torch.equal(data["advantages"][0], torch.full((4,), 0.25))
    assert torch.equal(data["advantages"][1], torch.full((6,), -1.5))


def test_advantages_omega_zero_is_pure_outcome(monkeypatch):
    monkeypatch.setenv(ta.TURN_MIX_ENV, "0")
    md = {"turn_spans": [[0, 4]], "turn_adv": [5.0]}
    data = _rollout_data([4], rewards=[1.0], metadata=[md])
    ta.compute_turn_advantages(SimpleNamespace(), data)
    assert torch.equal(data["advantages"][0], torch.ones(4))


def test_advantages_cp_mismatch_falls_back_to_broadcast(monkeypatch):
    # kl is CP-local (shorter than response) and megatron isn't importable here:
    # the sample must degrade to the outcome broadcast, never crash.
    monkeypatch.setenv(ta.TURN_MIX_ENV, "0.3")
    md = {"turn_spans": [[0, 8]], "turn_adv": [1.0]}
    data = _rollout_data([6], rewards=[0.5], metadata=[md], response_lengths=[12])
    ta.compute_turn_advantages(SimpleNamespace(), data)
    assert torch.equal(data["advantages"][0], torch.full((6,), 0.5))


def test_advantages_cp_slice_path(monkeypatch):
    # Inject a fake cp_utils so the CP branch (paint full, then slice) is exercised.
    import sys
    from types import ModuleType

    fake = ModuleType("slime.backends.megatron_utils.cp_utils")

    def fake_slice(full, total_length, response_length):
        assert len(full) == response_length
        return full[:6]  # pretend this rank owns the first 6 response tokens

    fake.slice_log_prob_with_cp = fake_slice
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.cp_utils", fake)
    monkeypatch.setenv(ta.TURN_MIX_ENV, "0.5")

    md = {"turn_spans": [[0, 4], [8, 12]], "turn_adv": [2.0, -2.0]}
    data = _rollout_data([6], rewards=[0.0], metadata=[md], response_lengths=[12])
    ta.compute_turn_advantages(SimpleNamespace(), data)
    expected = torch.zeros(6)
    expected[0:4] = 0.5 * 2.0  # only the first span lands in this rank's slice
    assert torch.allclose(data["advantages"][0], expected)


def test_span_clamped_to_response_length(monkeypatch):
    monkeypatch.setenv(ta.TURN_MIX_ENV, "1.0")
    md = {"turn_spans": [[3, 99]], "turn_adv": [1.0]}
    data = _rollout_data([5], rewards=[0.0], metadata=[md])
    ta.compute_turn_advantages(SimpleNamespace(), data)
    assert torch.allclose(data["advantages"][0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# RecordingModel span/timestamp recording
# ---------------------------------------------------------------------------


def test_recording_model_records_turn_spans_and_ts(monkeypatch):
    from ._fakes import FakeTokenizer
    import agentic_rl.core.model as model_mod
    from agentic_rl.core.model import RecordingModel

    def fake_parse(raw, **kwargs):
        uses = [{"name": "bash", "input": {"command": "echo hi"}}] if "TOOLCALL" in raw else []
        return SimpleNamespace(tool_uses=uses, text=raw, reasoning=None)

    monkeypatch.setattr(model_mod, "parse_model_output", fake_parse)

    tok = FakeTokenizer()
    m = RecordingModel(tok, {}, "http://fake:1", "s1", tool_parser=None, reasoning_parser=None)
    script = [("first TOOLCALL", "stop"), ("second TOOLCALL", "stop")]

    def fake_gen(input_ids, *, max_new_tokens=None):
        text, finish = script.pop(0)
        ids = tok.encode(text)
        return ids, [0.0] * len(ids), finish, "v1"

    m._generate = fake_gen

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    reply = m.query(messages)
    messages = messages + [reply, {"role": "user", "content": "obs one"}]
    m.query(messages)

    c = m.cur
    assert len(c.turn_spans) == 2 and len(c.turn_ts) == 2
    (s0, e0), (s1, e1) = c.turn_spans
    assert (s0, e0) == (c.prompt_len, c.prompt_len + len(tok.encode("first TOOLCALL")))
    assert e1 - s1 == len(tok.encode("second TOOLCALL"))
    assert s1 > e0  # observation delta sits between the turns, outside both spans
    assert e1 == len(c.tokens)
    assert c.turn_ts[0] <= c.turn_ts[1]
    # spans cover exactly the trained tokens (observation delta is mask-0)
    for (s, e) in c.turn_spans:
        assert all(c.loss_mask[s:e])
    # response-relative conversion used by generate.py stays in range
    rel = [[s - c.prompt_len, e - c.prompt_len] for s, e in c.turn_spans]
    assert rel[0][0] == 0 and rel[1][1] == len(c.tokens) - c.prompt_len
