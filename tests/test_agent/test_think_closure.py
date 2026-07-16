"""Forced think closure (``agentic_close_think_on_length``) in RecordingModel.

A turn that hits the per-turn cap (finish=length) with no tool call used to be
rolled back wholesale -> ContextLengthExceeded -> a reward-0 null episode (36-39%
of training samples in run 20260708-181706). With the flag on, the model keeps the
truncated generation, injects a loss-masked ``</think>`` closure, and continues the
SAME turn under a small budget so the agent can act on what it already worked out.

CPU-only: ``_generate`` is scripted per test and ``parse_model_output`` is stubbed
(a reply containing ``TOOLCALL`` parses to one bash call), so the real query loop,
token/mask bookkeeping, and rollback paths are what's under test.
"""

from types import SimpleNamespace

import pytest
from minisweagent.exceptions import LimitsExceeded

import agentic_rl.model as model_mod
from agentic_rl.model import RecordingModel

from ._fakes import FakeTokenizer

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]


def _make_model(tok, **kwargs) -> RecordingModel:
    return RecordingModel(
        tok,
        {"temperature": 1.0},
        "http://fake:1",
        "session-1",
        tool_parser=None,
        reasoning_parser=None,
        **kwargs,
    )


def _script_generate(model, tok, turns):
    """Replace ``_generate`` with a scripted queue of (text, finish); records the
    ``max_new_tokens`` override of every call so tests can assert the continuation
    budget."""
    queue = list(turns)
    calls = []

    def fake(input_ids, *, max_new_tokens=None):
        assert queue, "unexpected _generate call (script exhausted)"
        calls.append(max_new_tokens)
        text, finish = queue.pop(0)
        ids = tok.encode(text)
        return ids, [0.0] * len(ids), finish, f"v{len(calls)}"

    model._generate = fake
    return queue, calls


def _stub_parser(monkeypatch):
    """Tool call iff the decoded turn text contains ``TOOLCALL``."""

    def fake_parse(raw, **kwargs):
        uses = [{"name": "bash", "input": {"command": "echo hi"}}] if "TOOLCALL" in raw else []
        return SimpleNamespace(tool_uses=uses, text=raw, reasoning=None)

    monkeypatch.setattr(model_mod, "parse_model_output", fake_parse)


def test_closure_rescues_length_capped_think(monkeypatch):
    tok = FakeTokenizer()
    _stub_parser(monkeypatch)
    m = _make_model(tok, close_think_on_length=True, max_think_closures=2, think_closure_budget=4096)
    queue, calls = _script_generate(m, tok, [("<think> deep thought unfinished", "length"), ("done TOOLCALL", "stop")])

    msg = m.query(MESSAGES)

    assert msg["tool_calls"] and msg["tool_calls"][0]["function"]["name"] == "bash"
    assert m.n_think_closures == 1 and m.exit_status is None
    assert not queue, "both scripted generations consumed"
    # First call uses the default budget; the closure continuation is capped.
    assert calls == [None, 4096]

    c = m.cur
    gen1 = tok.encode("<think> deep thought unfinished")
    closure = tok.encode("\n</think>\n\n")
    gen2 = tok.encode("done TOOLCALL")
    assert c.tokens[c.prompt_len :] == gen1 + closure + gen2
    # The model's own tokens train; the injected closure never does.
    assert c.loss_mask[c.prompt_len :] == [1] * len(gen1) + [0] * len(closure) + [1] * len(gen2)
    assert len(c.versions) == 2  # one per generated segment


def test_giveup_rolls_back_whole_turn(monkeypatch):
    tok = FakeTokenizer()
    _stub_parser(monkeypatch)
    m = _make_model(tok, close_think_on_length=True, max_think_closures=2)
    _script_generate(
        m, tok, [("<think> still thinking", "length"), ("and thinking", "length"), ("more thinking", "length")]
    )

    with pytest.raises(LimitsExceeded):
        m.query(MESSAGES)

    assert m.exit_status == "ContextLengthExceeded"
    assert m.n_think_closures == 2
    c = m.cur
    # Every segment AND the injected closure ids are gone from the token stream.
    assert len(c.tokens) == c.prompt_len
    assert c.versions == []
    # The dashboard record spans the whole stitched turn (3 segments + 1 closure:
    # the second continuation sees a closed think, so nothing is injected again).
    assert c.truncated_tail["finish"] == "length"
    assert c.truncated_tail["tokens"] == len(tok.encode("<think> still thinking and thinking more thinking")) + 1


def test_flag_off_preserves_existing_rollback(monkeypatch):
    tok = FakeTokenizer()
    _stub_parser(monkeypatch)
    m = _make_model(tok)  # close_think_on_length defaults False
    queue, calls = _script_generate(m, tok, [("<think> runaway", "length")])

    with pytest.raises(LimitsExceeded):
        m.query(MESSAGES)

    assert m.exit_status == "ContextLengthExceeded"
    assert m.n_think_closures == 0
    assert len(calls) == 1, "no continuation attempted with the flag off"
    c = m.cur
    assert len(c.tokens) == c.prompt_len and c.versions == []


def test_closure_fires_for_template_seeded_think(monkeypatch):
    """Qwen3-style templates end the generation prompt with a pre-seeded '<think>':
    the model's own text then never contains the opener, but the turn is still
    mid-think and the closure must be injected (run 20260710-080841 showed 0
    injections across 31 closure attempts before this rule)."""
    tok = FakeTokenizer()
    _stub_parser(monkeypatch)
    m = _make_model(tok, close_think_on_length=True)
    # The rendered prompt tail ends with the opener (here via the user message; in
    # prod the chat template appends it after '<|im_start|>assistant\n').
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task <think>"}]
    _script_generate(m, tok, [("endless pondering no markers", "length"), ("act TOOLCALL", "stop")])

    m.query(messages)

    assert m.n_think_closures == 1
    c = m.cur
    gen1 = tok.encode("endless pondering no markers")
    closure = tok.encode("\n</think>\n\n")
    gen2 = tok.encode("act TOOLCALL")
    assert c.tokens[c.prompt_len :] == gen1 + closure + gen2
    assert c.loss_mask[c.prompt_len :] == [1] * len(gen1) + [0] * len(closure) + [1] * len(gen2)


def test_no_injection_without_unclosed_think(monkeypatch):
    tok = FakeTokenizer()
    _stub_parser(monkeypatch)
    m = _make_model(tok, close_think_on_length=True)
    _script_generate(m, tok, [("rambling cut mid sentence", "length"), ("act TOOLCALL", "stop")])

    m.query(MESSAGES)

    assert m.n_think_closures == 1
    c = m.cur
    gen1 = tok.encode("rambling cut mid sentence")
    gen2 = tok.encode("act TOOLCALL")
    # Continuation happened but no closure ids were injected (no dangling <think>).
    assert c.tokens[c.prompt_len :] == gen1 + gen2
    assert c.loss_mask[c.prompt_len :] == [1] * (len(gen1) + len(gen2))
