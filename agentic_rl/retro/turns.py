"""Per-turn boundary records for all-turns retro capture.

Design doc: "All-Turns Snapshot Store" (2026-08-26). The expensive halves of a
resumable agent state stay once-per-trajectory — ONE workspace snapshot (the
staging root, every turn a subdirectory) and ONE final :class:`ChainCheckpoint`
— while a :class:`TurnRecord` is the ~100-byte per-turn complement: enough to
(a) name the staged workspace directory for its turn and (b) reconstruct the
exact turn-t checkpoint by truncating the final one.

Faithfulness is asserted, not assumed. ``Chain`` is append-only at post-tool
boundaries (rollbacks rewrite only the *current* turn; a non-append ``query``
starts a new chain), so the chain state at turn t is a prefix of the final
state. Each record therefore carries a digest of the token/message prefix as it
existed when the turn completed; :func:`derive_checkpoint` recomputes that
digest from the final checkpoint and raises :class:`PrefixViolation` on any
mismatch (chain reset with a diverging re-render, in-place message mutation,
template-var drift). A violation degrades to "this turn is not branchable",
never to a corrupt replay.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any

from typing import TYPE_CHECKING

from .selector import valid_score_trace

if TYPE_CHECKING:  # the model/checkpoint stack is imported lazily so that the
    # staging + score-trace half stays importable without jinja2/mini-swe
    # (profiles/retro_snapshot_bench runs with the bare Modal SDK only)
    from .backends.miniswe_checkpoint import ChainCheckpoint


class PrefixViolation(ValueError):
    """The final chain does not contain this turn's recorded prefix."""


@dataclass(frozen=True)
class TurnRecord:
    """One post-tool boundary: where to cut the final checkpoint for turn t.

    ``token_len`` ends immediately after turn t's last assistant action;
    ``messages_len`` includes turn t's tool observation(s) — mirroring
    ``capture_checkpoint`` at the same boundary.
    """

    turn_index: int
    token_len: int
    seen_msgs: int
    messages_len: int
    n_calls: int
    cost: float
    source_weight_version: str
    elapsed_seconds: float
    ts: float
    digest: str
    chain_seq: int = 0
    gen_tokens: int = 0
    mean_logprob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TurnRecord":
        return cls(**value)


@dataclass
class TurnRecorder:
    """Per-episode recorder; O(new tokens) per boundary via an incremental hasher.

    A chain reset (non-append ``query``) re-seeds the hasher from the new
    chain's full token stream; the digest check at derive time — not chain
    identity — is what decides whether a record is still branchable, because a
    reset that re-renders the identical prefix is still faithfully truncatable.
    """

    records: list[TurnRecord] = field(default_factory=list)
    chain_seq: int = 0
    _chain_id: int | None = None
    _hashed_len: int = 0
    _hasher: Any = None

    def record(self, agent: Any, model: Any, *, turn_index: int, elapsed_seconds: float, ts: float) -> TurnRecord:
        chain = model.cur
        if self._hasher is None or id(chain) != self._chain_id or len(chain.tokens) < self._hashed_len:
            if self._hasher is not None:
                self.chain_seq += 1
            self._hasher = hashlib.sha256()
            self._chain_id = id(chain)
            self._hashed_len = 0
        self._hasher.update(_token_bytes(chain.tokens[self._hashed_len :]))
        self._hashed_len = len(chain.tokens)

        span = chain.turn_spans[-1] if getattr(chain, "turn_spans", None) else (0, len(chain.tokens))
        gen_logps = [
            lp
            for lp, mask in zip(chain.logprobs[span[0] : span[1]], chain.loss_mask[span[0] : span[1]])
            if mask == 1
        ]
        record = TurnRecord(
            turn_index=turn_index,
            token_len=len(chain.tokens),
            seen_msgs=int(chain.seen_msgs),
            messages_len=len(agent.messages),
            n_calls=int(agent.n_calls),
            cost=float(agent.cost),
            source_weight_version=_last_version(chain),
            elapsed_seconds=float(elapsed_seconds),
            ts=float(ts),
            digest=_combine_digest(
                self._hasher.copy().digest(),
                chain.msg_hashes,
                agent.extra_template_vars,
            ),
            chain_seq=self.chain_seq,
            gen_tokens=max(0, span[1] - span[0]),
            mean_logprob=(sum(gen_logps) / len(gen_logps)) if gen_logps else None,
        )
        self.records.append(record)
        return record


def derive_checkpoint(final: "ChainCheckpoint", record: TurnRecord) -> "ChainCheckpoint":
    """Exact turn-t checkpoint by truncating the final one; digest-verified."""

    from agentic_rl.core.model import _stable_hash

    from .backends.miniswe_checkpoint import ChainCheckpoint

    if record.token_len > len(final.tokens) or record.messages_len > len(final.messages):
        raise PrefixViolation(
            f"turn {record.turn_index}: recorded prefix (tokens={record.token_len}, "
            f"messages={record.messages_len}) exceeds the final checkpoint "
            f"(tokens={len(final.tokens)}, messages={len(final.messages)})"
        )
    tokens = final.tokens[: record.token_len]
    messages = copy.deepcopy(final.messages[: record.messages_len])
    msg_hashes = [_stable_hash(m) for m in messages[: record.seen_msgs]]
    derived_digest = _combine_digest(
        hashlib.sha256(_token_bytes(tokens)).digest(),
        msg_hashes,
        final.extra_template_vars,
    )
    if derived_digest != record.digest:
        raise PrefixViolation(
            f"turn {record.turn_index}: final chain does not contain this turn's recorded "
            "prefix (chain reset with diverging re-render, in-place message mutation, or "
            "template-var drift) — this turn is not branchable from the final checkpoint"
        )
    return ChainCheckpoint(
        tokens=tokens,
        seen_msgs=record.seen_msgs,
        msg_hashes=msg_hashes,
        messages=messages,
        n_calls=record.n_calls,
        cost=record.cost,
        extra_template_vars=copy.deepcopy(final.extra_template_vars),
        source_weight_version=record.source_weight_version,
    )


def verify_prefix(final: "ChainCheckpoint", record: TurnRecord) -> bool:
    try:
        derive_checkpoint(final, record)
        return True
    except PrefixViolation:
        return False


def staging_command(
    source_dir: str,
    staging_root: str,
    turn_index: int,
    prev_turn_index: int | None = None,
) -> str:
    """rsync the live workspace into an immutable per-turn staging directory.

    ``--link-dest`` hardlinks unchanged files against the PREVIOUS staged turn
    (never against the live workspace, so later in-place writes to the source
    cannot mutate staged history); only deltas materialize.

    ``--checksum`` is load-bearing, not an optimization knob: rsync's default
    quick-check (size + mtime) silently hardlinks the OLD content for a
    same-size edit landing in the same mtime granule — a pattern agents hit
    constantly (sed-style in-place rewrites). Content comparison costs one read
    of the workspace per turn (~10–50 ms at 10–50 MiB) and makes staged state
    exact. Caught by test_allturns_verification.py::test_staged_state_matches.
    """

    dest = turn_dir(staging_root, turn_index)
    link = ""
    if prev_turn_index is not None:
        link = f" --link-dest={shlex.quote(turn_dir(staging_root, prev_turn_index) + '/')}"
    return (
        f"mkdir -p {shlex.quote(dest)} && "
        f"rsync -a --checksum{link} {shlex.quote(source_dir.rstrip('/') + '/')} {shlex.quote(dest + '/')}"
    )


def turn_dir(staging_root: str, turn_index: int) -> str:
    return f"{staging_root.rstrip('/')}/{turn_index:04d}"


class ScoreTraceBuilder:
    """Attribute submission scores to the post-tool boundary where they first appeared.

    A score belongs to the turn at whose boundary it first appeared — the same
    attribution ``EventSelector.observe_log`` uses (its ``_seen_scored``
    cursor), so lease-time policies replaying a stored trace see exactly what
    in-episode selection saw. Scores come from the sandbox log and stay
    selection-only; reward remains server-side.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._seen = 0

    def observe(self, turn_index: int, log_text: str) -> None:
        scores = valid_score_trace(log_text)
        for submission_index in range(self._seen, len(scores)):
            self.events.append(
                {
                    "turn_index": int(turn_index),
                    "submission_index": submission_index,
                    "score": float(scores[submission_index]),
                }
            )
        self._seen = max(self._seen, len(scores))


def score_trace_from_logs(turn_logs: list[tuple[int, str]]) -> list[dict[str, Any]]:
    """One-shot form of :class:`ScoreTraceBuilder` over recorded boundary logs."""

    builder = ScoreTraceBuilder()
    for turn_index, text in turn_logs:
        builder.observe(turn_index, text)
    return builder.events


def probe_capture_tools(sandbox: Any) -> dict[str, bool]:
    """Which staging/packing tools the task image actually ships.

    Task images are converter-built app images, not ours — rsync in particular
    is not guaranteed. all_turns capture requires rsync + tar (+gzip); when the
    probe fails the episode degrades to winner-mode capture and a metric flags
    it, never a crash.
    """

    tools = ("rsync", "tar", "gzip", "zstd")
    checks = " ; ".join(f"command -v {tool} >/dev/null 2>&1 && echo {tool}" for tool in tools)
    rc, out, _ = sandbox.exec(checks, check=False, timeout=30)
    present = set((out or "").split())
    return {tool: tool in present for tool in tools}


def branch_budget(
    record: TurnRecord, *, total_turns: int | None, total_seconds: float | None
) -> tuple[int, int]:
    """(remaining_steps, remaining_seconds) when branching at ``record``."""

    steps = max(1, int(total_turns or 0) - record.turn_index) if total_turns else 1
    seconds = max(1, int((total_seconds or 0.0) - record.elapsed_seconds)) if total_seconds else 1
    return steps, seconds


def _token_bytes(tokens: list[int]) -> bytes:
    return b"".join(int(token).to_bytes(8, "little", signed=True) for token in tokens)


def _combine_digest(token_digest: bytes, msg_hashes: list[str], extra_template_vars: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(token_digest)
    h.update(b"\x00msgs\x00")
    h.update("\x00".join(msg_hashes).encode("utf-8"))
    h.update(b"\x00etv\x00")
    h.update(json.dumps(extra_template_vars, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def _last_version(chain: Any) -> str:
    versions = [version for version in getattr(chain, "versions", ()) if version is not None]
    return str(versions[-1]) if versions else ""
