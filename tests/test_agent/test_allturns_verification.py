"""All-turns snapshot store: verification suite (design doc 2026-08-26).

Three properties, tested before any production wiring exists:

A. Hardlink staging (``turns.staging_command``) — every staged turn directory is
   an exact, immutable copy of the workspace as it stood at that turn; unchanged
   files hardlink against the PREVIOUS staged turn so storage is base + deltas.
   Runs against the real local ``rsync`` (dedup-specific assertions skip when
   the platform rsync doesn't hardlink; the Modal bench is authoritative there).

B. Trajectory faithfulness (``turns.TurnRecorder`` / ``derive_checkpoint``) —
   driven through the REAL ``RecordingModel.query`` loop with scripted
   generations: the checkpoint derived by truncating the FINAL checkpoint at a
   recorded boundary equals the checkpoint captured live at that boundary,
   field for field, including across rollback-and-retry turns; in-place history
   mutation (which also resets the chain) is detected, never silently replayed.

C. Score-trace attribution (``turns.score_trace_from_logs``) — a submission
   score belongs to the post-tool boundary where it first appeared, matching
   ``EventSelector.observe_log``'s cursor exactly, including multiple records
   appended by one tool action and malformed/out-of-range records.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The CPU CI job installs neither jinja2 nor mini-swe-agent (matching the other
# matrix files' stubbing pattern). core.model needs jinja2 only to render the
# format-error message, and minisweagent only for its two exception classes and
# the observation formatter (unused here) — stub the minimum, real classes for
# the exceptions since the query loop raises and catches them.
try:  # pragma: no cover - present in local dev envs
    import minisweagent  # noqa: F401
except ImportError:  # pragma: no cover - CI path
    _exc = types.ModuleType("minisweagent.exceptions")

    class _FormatError(Exception):
        pass

    class _LimitsExceeded(Exception):
        pass

    _exc.FormatError = _FormatError
    _exc.LimitsExceeded = _LimitsExceeded
    _actions = types.ModuleType("minisweagent.models.utils.actions_toolcall")
    _actions.format_toolcall_observation_messages = lambda **kwargs: []
    _actions.BASH_TOOL = {"name": "bash", "input_schema": {}}
    for _name, _mod in {
        "minisweagent": types.ModuleType("minisweagent"),
        "minisweagent.exceptions": _exc,
        "minisweagent.models": types.ModuleType("minisweagent.models"),
        "minisweagent.models.utils": types.ModuleType("minisweagent.models.utils"),
        "minisweagent.models.utils.actions_toolcall": _actions,
    }.items():
        _mod.__path__ = []
        sys.modules[_name] = _mod
try:  # pragma: no cover
    import jinja2  # noqa: F401
except ImportError:  # pragma: no cover - CI path
    _jinja = types.ModuleType("jinja2")

    class _Template:
        def __init__(self, source, **kwargs):
            self._source = source

        def render(self, **kwargs):
            return self._source

    _jinja.Template = _Template
    _jinja.StrictUndefined = object
    sys.modules["jinja2"] = _jinja

import agentic_rl.core.model as model_mod
from agentic_rl.core.model import RecordingModel
from agentic_rl.retro.backends.miniswe_checkpoint import (
    capture_checkpoint,
    restore_agent,
    restore_recording_model,
)
from agentic_rl.retro.selector import EventSelector, valid_score_trace
from agentic_rl.retro.turns import (
    PrefixViolation,
    TurnRecorder,
    derive_checkpoint,
    score_trace_from_logs,
    staging_command,
    turn_dir,
    verify_prefix,
)

from tests.test_agent._fakes import FakeTokenizer

# --------------------------------------------------------------------------
# Part A: hardlink staging
# --------------------------------------------------------------------------


def _sh(cmd: str) -> None:
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


_HAS_RSYNC = shutil.which("rsync") is not None


def _rsync_hardlinks(tmp: Path) -> bool:
    """Capability probe: does this platform's rsync --link-dest actually hardlink?"""
    src = tmp / "src"
    src.mkdir(parents=True)
    (src / "f").write_bytes(b"same")
    _sh(staging_command(str(src), str(tmp), 0))
    _sh(staging_command(str(src), str(tmp), 1, prev_turn_index=0))
    return (tmp / "0000" / "f").stat().st_ino == (tmp / "0001" / "f").stat().st_ino


pytestmark_rsync = pytest.mark.skipif(not _HAS_RSYNC, reason="rsync not installed")


class _Workspace:
    """Scripted per-turn workspace mutations + expected-state bookkeeping."""

    def __init__(self, tmp: Path):
        self.app = tmp / "app"
        self.staging = tmp / "staging"
        self.app.mkdir()
        self.staging.mkdir()
        self.expected: dict[int, dict[str, bytes]] = {}
        self.prev: int | None = None

    def mutate(self, turn: int) -> None:
        app = self.app
        if turn == 0:
            (app / "main.py").write_bytes(b"print('v0')\n")
            (app / "data.bin").write_bytes(os.urandom(4096))
            (app / "keep.txt").write_bytes(b"never changes\n")
            (app / "sub").mkdir()
            (app / "sub" / "notes.md").write_bytes(b"turn 0\n")
        elif turn == 1:
            (app / "main.py").write_bytes(b"print('v1')\n")  # modify
            (app / "new1.txt").write_bytes(b"added at 1\n")  # create
        elif turn == 2:
            with open(app / "sub" / "notes.md", "ab") as f:  # in-place append
                f.write(b"appended at 2\n")
            (app / "data.bin").unlink()  # delete
        elif turn == 3:
            (app / "new1.txt").rename(app / "renamed1.txt")  # rename
            (app / "big.bin").write_bytes(os.urandom(64 * 1024))
        elif turn == 4:
            os.chmod(app / "main.py", 0o755)  # metadata-only change
        else:
            (app / f"turn{turn}.log").write_bytes(f"log {turn}\n".encode())

    def stage(self, turn: int) -> None:
        self.mutate(turn)
        self.expected[turn] = _tree(self.app)
        _sh(staging_command(str(self.app), str(self.staging), turn, prev_turn_index=self.prev))
        self.prev = turn

    def staged(self, turn: int) -> Path:
        return Path(turn_dir(str(self.staging), turn))


@pytestmark_rsync
def test_staged_state_matches_workspace_at_every_turn(tmp_path):
    ws = _Workspace(tmp_path)
    for turn in range(10):
        ws.stage(turn)
    for turn in range(10):
        assert _tree(ws.staged(turn)) == ws.expected[turn], f"turn {turn} staged tree diverged"


@pytestmark_rsync
def test_unchanged_files_hardlink_and_changed_files_do_not(tmp_path):
    if not _rsync_hardlinks(tmp_path / "probe"):
        pytest.skip("platform rsync does not hardlink via --link-dest (Modal bench covers linux)")
    ws = _Workspace(tmp_path)
    for turn in range(3):
        ws.stage(turn)
    ino = lambda turn, rel: (ws.staged(turn) / rel).stat().st_ino  # noqa: E731
    # keep.txt never changes: one inode shared by all three staged turns
    assert ino(0, "keep.txt") == ino(1, "keep.txt") == ino(2, "keep.txt")
    # main.py changed at turn 1 then not at turn 2
    assert ino(0, "main.py") != ino(1, "main.py")
    assert ino(1, "main.py") == ino(2, "main.py")
    # in-place append at turn 2 must break the link (this is the cp -al hazard)
    assert ino(1, "sub/notes.md") != ino(2, "sub/notes.md")


@pytestmark_rsync
def test_later_workspace_mutation_cannot_alter_staged_history(tmp_path):
    ws = _Workspace(tmp_path)
    for turn in range(3):
        ws.stage(turn)
    # every in-place mutation style against the LIVE workspace
    with open(ws.app / "keep.txt", "ab") as f:
        f.write(b"tamper-append\n")
    (ws.app / "main.py").write_bytes(b"tamper-rewrite\n")
    with open(ws.app / "sub" / "notes.md", "r+b") as f:
        f.truncate(1)
    for turn in range(3):
        assert _tree(ws.staged(turn)) == ws.expected[turn], f"turn {turn} history mutated"


@pytestmark_rsync
def test_deletions_propagate_between_turns(tmp_path):
    ws = _Workspace(tmp_path)
    for turn in range(3):
        ws.stage(turn)
    assert (ws.staged(1) / "data.bin").exists()
    assert not (ws.staged(2) / "data.bin").exists()


@pytestmark_rsync
def test_storage_is_base_plus_deltas_not_turns_times_base(tmp_path):
    if not _rsync_hardlinks(tmp_path / "probe"):
        pytest.skip("platform rsync does not hardlink via --link-dest (Modal bench covers linux)")
    ws = _Workspace(tmp_path)
    turns = 10
    for turn in range(turns):
        ws.stage(turn)
    logical = physical = 0
    seen: set[int] = set()
    for path in ws.staging.rglob("*"):
        if not path.is_file():
            continue
        st = path.stat()
        logical += st.st_size
        if st.st_ino not in seen:
            seen.add(st.st_ino)
            physical += st.st_size
    assert physical < logical / 3, f"expected >3x dedup, got {logical}/{physical}"


def _rsync_propagates_mode_only_change(tmp: Path) -> bool:
    """GNU rsync --link-dest refuses to link when perms differ (mode-only change
    propagates); some platform rsyncs (macOS openrsync) link anyway and lose the
    chmod. Production sandboxes run GNU rsync — the Modal bench asserts this hard."""
    src = tmp / "src"
    src.mkdir(parents=True)
    (src / "f").write_bytes(b"same")
    _sh(staging_command(str(src), str(tmp), 0))
    os.chmod(src / "f", 0o755)
    _sh(staging_command(str(src), str(tmp), 1, prev_turn_index=0))
    return (Path(turn_dir(str(tmp), 1)) / "f").stat().st_mode & 0o777 == 0o755


@pytestmark_rsync
def test_rsync_archive_preserves_mode_and_mtime(tmp_path):
    ws = _Workspace(tmp_path)
    for turn in range(5):
        ws.stage(turn)
    live, staged = (ws.app / "main.py").stat(), (ws.staged(4) / "main.py").stat()
    assert int(staged.st_mtime) == int(live.st_mtime)
    if _rsync_propagates_mode_only_change(tmp_path / "modeprobe"):
        assert staged.st_mode == live.st_mode  # 0o755 from turn 4
    else:
        pytest.skip("platform rsync links across a mode-only change (GNU rsync does not; Modal bench asserts it)")


@pytestmark_rsync
def test_staging_command_handles_spaces_in_paths(tmp_path):
    src = tmp_path / "work space"
    staging = tmp_path / "stage dir"
    src.mkdir()
    staging.mkdir()
    (src / "a file.txt").write_bytes(b"x")
    _sh(staging_command(str(src), str(staging), 0))
    assert (Path(turn_dir(str(staging), 0)) / "a file.txt").read_bytes() == b"x"


# --------------------------------------------------------------------------
# Part B: trajectory faithfulness through the real RecordingModel
# --------------------------------------------------------------------------

_MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]


def _stub_parser(monkeypatch):
    def fake_parse(raw, **kwargs):
        uses = [{"name": "bash", "input": {"command": "echo hi"}}] if "TOOLCALL" in raw else []
        return SimpleNamespace(tool_uses=uses, text=raw, reasoning=None)

    monkeypatch.setattr(model_mod, "parse_model_output", fake_parse)


def _make_model(tok) -> RecordingModel:
    return RecordingModel(
        tok, {"temperature": 1.0}, "http://fake:1", "s1", tool_parser=None, reasoning_parser=None
    )


def _script_generate(model, tok, replies: list[str]):
    queue = list(replies)

    def fake(input_ids, *, max_new_tokens=None):
        assert queue, "unexpected _generate call (script exhausted)"
        text = queue.pop(0)
        ids = tok.encode(text)
        return ids, [-0.25] * len(ids), "stop", f"v{len(queue)}"

    model._generate = fake


class _Episode:
    """Drive the real query loop; capture live checkpoints + TurnRecords per boundary."""

    def __init__(self, tok, monkeypatch):
        _stub_parser(monkeypatch)
        self.model = _make_model(tok)
        self.agent = SimpleNamespace(
            messages=list(_MESSAGES), n_calls=0, cost=0.0, extra_template_vars={"workdir": "/app"}
        )
        self.recorder = TurnRecorder()
        self.live: list = []  # (TurnRecord, ChainCheckpoint) per boundary

    def turn(self, turn_index: int, *, reply: str | None = None):
        msg = self.model.query(self.agent.messages)
        self.agent.messages.append(msg)
        self.agent.n_calls += 1
        self.agent.cost += 0.01
        self.agent.messages.append({"role": "user", "content": f"observation {turn_index}"})
        record = self.recorder.record(
            self.agent, self.model, turn_index=turn_index, elapsed_seconds=float(turn_index), ts=1e9 + turn_index
        )
        self.live.append((record, capture_checkpoint(self.agent, self.model)))

    def final(self):
        return capture_checkpoint(self.agent, self.model)


def _assert_checkpoints_equal(derived, captured, context: str):
    assert derived.tokens == captured.tokens, context
    assert derived.seen_msgs == captured.seen_msgs, context
    assert derived.msg_hashes == captured.msg_hashes, context
    assert derived.messages == captured.messages, context
    assert derived.n_calls == captured.n_calls, context
    assert derived.cost == pytest.approx(captured.cost), context
    assert derived.extra_template_vars == captured.extra_template_vars, context
    assert derived.source_weight_version == captured.source_weight_version, context


def test_derived_checkpoint_equals_captured_at_every_boundary(monkeypatch):
    tok = FakeTokenizer()
    ep = _Episode(tok, monkeypatch)
    n_turns = 8
    _script_generate(ep.model, tok, [f"<think>step {t}</think> do TOOLCALL {t}" for t in range(n_turns)])
    for t in range(n_turns):
        ep.turn(t + 1)
    final = ep.final()
    for record, captured in ep.live:
        derived = derive_checkpoint(final, record)
        _assert_checkpoints_equal(derived, captured, f"turn {record.turn_index}")
        assert verify_prefix(final, record)


def test_restore_from_derived_equals_restore_from_captured(monkeypatch):
    tok = FakeTokenizer()
    ep = _Episode(tok, monkeypatch)
    _script_generate(ep.model, tok, [f"go TOOLCALL {t}" for t in range(5)])
    for t in range(5):
        ep.turn(t + 1)
    final = ep.final()
    record, captured = ep.live[2]  # a mid-trajectory branch point
    derived = derive_checkpoint(final, record)

    restored = []
    for checkpoint in (derived, captured):
        model = _make_model(tok)
        agent = SimpleNamespace(messages=[], n_calls=0, cost=0.0, extra_template_vars={})
        restore_recording_model(model, checkpoint)
        restore_agent(agent, checkpoint)
        restored.append((model.cur, agent))
    chain_a, agent_a = restored[0]
    chain_b, agent_b = restored[1]
    assert chain_a.tokens == chain_b.tokens
    assert chain_a.loss_mask == chain_b.loss_mask == [0] * len(chain_a.tokens)  # fully masked prefix
    assert chain_a.prompt_len == chain_b.prompt_len == len(chain_a.tokens)
    assert chain_a.seen_msgs == chain_b.seen_msgs
    assert chain_a.msg_hashes == chain_b.msg_hashes
    assert chain_a.full_prompt == chain_b.full_prompt
    assert agent_a.messages == agent_b.messages
    assert (agent_a.n_calls, agent_a.cost) == (agent_b.n_calls, agent_b.cost)


def test_rollback_and_retry_turn_stays_faithful(monkeypatch):
    """A format-error turn rewrites the current segment; earlier and later
    boundaries must still derive exactly (the recorder re-extends its hasher
    over the rewritten region)."""
    from minisweagent.exceptions import FormatError

    tok = FakeTokenizer()
    ep = _Episode(tok, monkeypatch)
    _script_generate(
        ep.model,
        tok,
        ["fine TOOLCALL 1", "no tool call here", "retried TOOLCALL 2", "fine TOOLCALL 3"],
    )
    ep.turn(1)
    with pytest.raises(FormatError):
        ep.model.query(ep.agent.messages)  # rolled back, no boundary recorded
    ep.agent.messages.append({"role": "user", "content": "format error, use the tool"})
    ep.turn(2)
    ep.turn(3)
    final = ep.final()
    assert ep.model.n_format_errors == 1
    for record, captured in ep.live:
        _assert_checkpoints_equal(derive_checkpoint(final, record), captured, f"turn {record.turn_index}")


def test_history_mutation_is_detected_not_silently_replayed(monkeypatch):
    tok = FakeTokenizer()
    ep = _Episode(tok, monkeypatch)
    _script_generate(ep.model, tok, [f"work TOOLCALL {t}" for t in range(6)])
    for t in (1, 2, 3):
        ep.turn(t)
    # In-place mutation of history: also forces a chain reset on the next query
    ep.agent.messages[1]["content"] = "task (edited in place)"
    for t in (4, 5, 6):
        ep.turn(t)
    final = ep.final()
    assert ep.recorder.chain_seq == 1, "the non-append query must have reset the chain"
    for record, _ in ep.live[:3]:
        assert not verify_prefix(final, record), f"pre-mutation turn {record.turn_index} must be refused"
        with pytest.raises(PrefixViolation):
            derive_checkpoint(final, record)
    for record, captured in ep.live[3:]:
        _assert_checkpoints_equal(derive_checkpoint(final, record), captured, f"turn {record.turn_index}")


def test_75_turn_episode_derives_everywhere_and_records_stats(monkeypatch):
    tok = FakeTokenizer()
    ep = _Episode(tok, monkeypatch)
    n_turns = 75
    _script_generate(ep.model, tok, [f"<think>{'x ' * 40}</think> act TOOLCALL {t}" for t in range(n_turns)])
    started = time.perf_counter()
    for t in range(n_turns):
        ep.turn(t + 1)
    record_seconds = time.perf_counter() - started
    final = ep.final()
    for record, captured in ep.live:
        _assert_checkpoints_equal(derive_checkpoint(final, record), captured, f"turn {record.turn_index}")
    # boundary metadata is monotone and per-turn stats are populated
    lens = [record.token_len for record, _ in ep.live]
    assert lens == sorted(lens) and len(set(lens)) == n_turns
    assert all(record.gen_tokens > 0 for record, _ in ep.live)
    assert all(record.mean_logprob == pytest.approx(-0.25) for record, _ in ep.live)
    assert record_seconds < 5.0, f"75 boundary records took {record_seconds:.2f}s"


# --------------------------------------------------------------------------
# Part C: score-trace attribution
# --------------------------------------------------------------------------


def _log(*records: dict) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def _done(uuid: str, score) -> dict:
    return {"submission_uuid": uuid, "ts": f"t-{uuid}", "status": "done", "score": score}


def test_score_trace_attribution_matches_event_selector_cursor():
    logs_by_turn = [
        (1, ""),  # nothing yet
        (2, _log(_done("a", 0.2))),
        (3, _log(_done("a", 0.2), _done("b", 0.4), _done("c", 0.45))),  # 2 in one tool action
        (4, _log(_done("a", 0.2), _done("b", 0.4), _done("c", 0.45),
                 {"submission_uuid": "d", "ts": "t-d", "status": "started"})),  # unscored
        (5, _log(_done("a", 0.2), _done("b", 0.4), _done("c", 0.45),
                 {"submission_uuid": "d", "ts": "t-d", "status": "started"}, _done("e", 0.1))),
        (6, _log(_done("a", 0.2), _done("b", 0.4), _done("c", 0.45),
                 {"submission_uuid": "d", "ts": "t-d", "status": "started"}, _done("e", 0.1))),
    ]
    trace = score_trace_from_logs(logs_by_turn)
    assert [(e["turn_index"], e["submission_index"], e["score"]) for e in trace] == [
        (2, 0, 0.2),
        (3, 1, 0.4),
        (3, 2, 0.45),
        (5, 3, 0.1),
    ]
    # parity with the in-episode selector's cursor: same scores consumed, in the
    # same per-turn batches (observe_log returns an event only for the FINAL new
    # record of a boundary; the cursor advance is what we assert)
    selector = EventSelector()
    for turn_index, text in logs_by_turn:
        selector.observe_log(
            text, turn_index=turn_index, max_steps=75, elapsed_seconds=float(turn_index), wall_time_seconds=1800
        )
        assert selector._seen_scored == len([e for e in trace if e["turn_index"] <= turn_index])
    assert selector._seen_scored == len(trace)
    # positions match the canonical parser order
    final_scores = valid_score_trace(logs_by_turn[-1][1])
    for event in trace:
        assert final_scores[event["submission_index"]] == event["score"]


def test_score_trace_ignores_malformed_and_out_of_range_records():
    logs_by_turn = [
        (1, _log(_done("a", 50.0))),  # raw 0-100 units: excluded (selection uses [0,1] only)
        (2, _log(_done("a", 50.0), _done("b", "abc"))),  # malformed: excluded
        (3, _log(_done("a", 50.0), _done("b", "abc"), _done("c", 0.7))),
    ]
    trace = score_trace_from_logs(logs_by_turn)
    assert [(e["turn_index"], e["score"]) for e in trace] == [(3, 0.7)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
