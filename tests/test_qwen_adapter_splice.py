"""Regression tests for the QwenOpenAIAdapter prompt-splice fix.

The qwen3_coder tool-call parser strips trailing whitespace from tool-call
arguments, so re-rendering the parsed assistant message is not token-identical to
the model's raw ``output_ids``. When ``_build_prompt`` re-rendered, that mismatch
made ``merge_turns`` log "prefix drift" and mask whole assistant turns out of
training. The fix splices the previous turn's raw ``output_ids`` into the next
prompt so prompt == training target by construction.

- ``test_splice_invariant_no_drift`` (unit): fake tokenizer, asserts every
  appended prompt starts with ``prompt_{i-1} + output_{i-1}`` and ``merge_turns``
  trains 100% of output tokens with zero drift warnings.
- ``test_real_template_trailing_whitespace_*`` (integration): real Qwen3.6
  tokenizer, reproduces the trailing-whitespace drift and shows the splice path
  eliminates it while the old re-render path does not.
"""
import json
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.adapters import openai as O
from slime.agent.adapters.common import AdapterChain, render_token_ids
from slime.agent.trajectory import TurnRecord, merge_turns
from async_rl_research.agent.adapters.qwen import _build_prompt as splice_build_prompt
from async_rl_research.agent.adapters.qwen import _dictify_tool_arguments

NUM_GPUS = 0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


class _Session:
    """Minimal stand-in for the adapter Session that ``_select_kind`` needs."""

    def __init__(self):
        self.main = AdapterChain()
        self.segments = []


class _DriftCapture:
    """Capture warnings emitted by ``merge_turns``."""

    def __enter__(self):
        self.msgs = []
        self._handler = logging.Handler()
        self._handler.emit = lambda record: self.msgs.append(record.getMessage())
        self._logger = logging.getLogger("slime.agent.trajectory")
        self._old_level = self._logger.level
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._old_level)


def _drive_episode(tok, raw_outputs, build_prompt, echo_for):
    """Run the adapter's per-turn loop. ``raw_outputs[i]`` is (output_ids, raw_text);
    ``echo_for(raw_text)`` returns the OpenAI assistant message handed back to the
    harness (models the parser). Returns (chain, merged_segment, drift_warnings)."""
    s = _Session()
    target = s.main
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can interact with a computer."},
        {"role": "user", "content": "Please solve the task. Use the bash tool."},
    ]
    for i, (output_ids, raw_text) in enumerate(raw_outputs):
        kind = O._select_kind(s, messages)
        prompt_ids = build_prompt(target, messages, TOOLS, kind, tok)
        target.turns.append(
            TurnRecord(
                prompt_ids=list(prompt_ids),
                output_ids=list(output_ids),
                finish_reason="tool_calls",
                output_log_probs=[-0.01] * len(output_ids),
            )
        )
        messages = messages + [
            echo_for(raw_text),
            {"role": "tool", "tool_call_id": "call_0", "content": f"<returncode>0</returncode>\n<output>{i}</output>"},
        ]
    with _DriftCapture() as cap:
        seg = merge_turns(target.turns)
    return target, seg, list(cap.msgs)


# --------------------------------------------------------------------------- #
# unit: fake tokenizer, splice invariant + clean merge                        #
# --------------------------------------------------------------------------- #

_IM_START, _IM_END, _NL = 800, 801, 802
_ROLE = {"system": 810, "user": 811, "assistant": 812, "tool": 813}


class _FakeTokenizer:
    """Prefix-consistent chat template: each message renders to a fixed block
    ending in ``<|im_end|>`` + ``\\n``; the generation prompt appends its own block.
    Models enough structure for ``_tool_continuation_ids``' sentinel-and-slice."""

    def _block(self, m):
        content = m.get("content") or ""
        ctoks = [(ord(c) % 50) + 100 for c in str(content)[:5]]
        calls = m.get("tool_calls") or []
        ctoks += [777] * len(calls)  # a fixed per-tool-call marker
        return [_IM_START, _ROLE.get(m.get("role"), 811)] + ctoks + [_IM_END, _NL]

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=True):
        ids = []
        for m in messages:
            ids += self._block(m)
        if add_generation_prompt:
            ids += [_IM_START, _ROLE["assistant"], 900]  # "<think>\n"
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return ""


@pytest.mark.unit
def test_splice_invariant_no_drift():
    tok = _FakeTokenizer()
    # three turns; output_ids are arbitrary but each ends in <|im_end|>
    raw_outputs = [([700 + i, 701 + i, 702 + i, _IM_END], f"raw{i}") for i in range(3)]

    def echo_for(_raw):
        return {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
        }

    target, seg, warns = _drive_episode(tok, raw_outputs, splice_build_prompt, echo_for)

    assert warns == [], f"splice path should not drift, got: {warns}"
    # every appended prompt must start with prompt_{i-1} + output_{i-1}
    for i in range(1, len(target.turns)):
        prev = list(target.turns[i - 1].prompt_ids) + list(target.turns[i - 1].output_ids)
        assert target.turns[i].prompt_ids[: len(prev)] == prev, f"turn {i} prompt does not extend prior turn"
    # 100% of generated output tokens are trained (mask=1); context is mask=0
    total_output = sum(len(o) for o, _ in raw_outputs)
    assert sum(seg.loss_mask) == total_output
    assert len(seg.loss_mask) == seg.response_ids.__len__()


# --------------------------------------------------------------------------- #
# integration: real Qwen3.6 template, trailing-whitespace drift               #
# --------------------------------------------------------------------------- #


def _real_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B")
    except Exception as exc:  # offline + not cached, gated, etc.
        pytest.skip(f"Qwen3.6 tokenizer unavailable: {exc}")


def _raw_output_text(reasoning, cmd):
    # cmd carries a trailing '\n' -> the parser strips it -> re-render drifts
    return (
        f"{reasoning}\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\n"
        f"{cmd}\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )


_TRAJECTORY = [
    ("Check the python version.", 'python3 -c "\nimport sys\nprint(sys.version)\n"\n'),
    ("Run the failing case.", 'cd /testbed && python3 -c "\nimport numpy as np\nprint(np.zeros((2,2)))\n"\n'),
    ("Apply the fix and re-run.", 'cd /testbed && python3 -c "\nprint(1 + 1)\n"\n'),
]


def _old_build_prompt(target, messages, tools_schema, kind, tok):
    """Pre-fix behavior: extend/replace, dict-ify, full re-render."""
    (O._extend_chat_messages if kind == "append" else O._replace_chat_messages)(target, messages, tools_schema)
    _dictify_tool_arguments(target.chat_messages)
    return render_token_ids(target, tok)


def _build_real_episode_inputs(tok):
    raw_outputs = []
    echo_map = {}
    for reasoning, cmd in _TRAJECTORY:
        raw_text = _raw_output_text(reasoning, cmd)
        raw_outputs.append((tok.encode(raw_text, add_special_tokens=False), raw_text))
        # model the qwen3_coder parser: tool value has its trailing whitespace
        # stripped, and arguments are returned to the harness as a JSON string
        # (exactly what _chat_message/_json_arguments produce).
        echo_map[raw_text] = {
            "role": "assistant",
            "content": None,
            "reasoning_content": reasoning,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd.rstrip()})},
                }
            ],
        }
    return raw_outputs, (lambda raw: echo_map[raw])


@pytest.mark.integration
def test_real_template_trailing_whitespace_old_path_drifts():
    tok = _real_tokenizer()
    raw_outputs, echo_for = _build_real_episode_inputs(tok)
    _, seg, warns = _drive_episode(tok, raw_outputs, _old_build_prompt, echo_for)
    total_output = sum(len(o) for o, _ in raw_outputs)
    # baseline: the old re-render path drifts and masks turns out of training
    assert any("prefix drift" in w for w in warns), "expected the pre-fix path to drift"
    assert sum(seg.loss_mask) < total_output, "expected the pre-fix path to mask some output"


@pytest.mark.integration
def test_real_template_trailing_whitespace_splice_is_clean():
    tok = _real_tokenizer()
    raw_outputs, echo_for = _build_real_episode_inputs(tok)
    target, seg, warns = _drive_episode(tok, raw_outputs, splice_build_prompt, echo_for)
    total_output = sum(len(o) for o, _ in raw_outputs)
    # the fix: no drift, every output token trained
    assert warns == [], f"splice path must not drift, got: {warns}"
    assert sum(seg.loss_mask) == total_output, "splice path must train 100% of output tokens"
    # and every appended prompt contains the prior turn's raw output verbatim
    for i in range(1, len(target.turns)):
        prev = list(target.turns[i - 1].prompt_ids) + list(target.turns[i - 1].output_ids)
        assert target.turns[i].prompt_ids[: len(prev)] == prev


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
