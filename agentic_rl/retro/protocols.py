"""The retro coupling surface: three protocols (RUNBOOK §7.1, P4/P6).

An env is retro-capturable iff it can provide all three plus a turn callback —
that is the whole ``Capturable`` contract, and it is what makes retro on a new
task family a per-family adapter instead of a rewrite:

* :class:`ScoreTrace` — what "progress" means. Frontier-CS reads the judge
  submissions log (``selector.EventSelector.observe_log``); a harbor-style
  family would adapt its per-step ``reward.json`` trace.
* :class:`SnapshotBackend` — what a snapshot is. Today:
  :mod:`.backends.modal_snapshot` (Modal directory-snapshot Images). A
  volume-tarball backend, or the git-in-workspace all-turns design (§7.1),
  would slot in here.
* :class:`AgentCheckpoint` — what agent state is. Today:
  :mod:`.backends.miniswe_checkpoint` (mini-swe ``Chain`` capture/restore).

These are structural (``typing.Protocol``): the backends never import this
module, and nothing here imports Modal or mini-swe.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ScoreTrace(Protocol):
    """A stream of scored progress events observed during a fresh episode."""

    def observe_log(
        self,
        log_text: str | None,
        *,
        turn_index: int,
        max_steps: int,
        elapsed_seconds: float,
        wall_time_seconds: float,
    ) -> Any | None:
        """Feed the newest progress log; return a stage-worthy event or None."""
        ...


@runtime_checkable
class SnapshotBackend(Protocol):
    """Creates, restores, and deletes workspace snapshots by opaque id."""

    def snapshot_sandbox(self, sandbox: Any, *, path: str, ttl_seconds: int | None) -> Any:
        """Photograph ``path`` inside ``sandbox``; result carries ``snapshot_id``."""
        ...

    def restore_directory(self, sandbox: Any, snapshot_id: str, *, target_path: str) -> Any:
        """Materialize a snapshot into ``target_path`` of a fresh sandbox."""
        ...

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Permanently delete a snapshot (idempotent on empty ids)."""
        ...


@runtime_checkable
class AgentCheckpoint(Protocol):
    """Serializable agent + recording-model state at a turn boundary."""

    def to_dict(self) -> dict[str, Any]: ...

    @property
    def n_calls(self) -> int:
        """Turns consumed at capture — branch ``step_limit`` = n_calls + budget."""
        ...
