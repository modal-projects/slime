"""Modal directory-snapshot adapter for retro replay.

All Modal imports are lazy so the metadata and selector tests remain CPU-only.
The adapter accepts either the local ``agentic_rl.core.sandbox.Sandbox`` wrapper or a
raw ``modal.Sandbox`` handle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..manifest import SnapshotKind

DEFAULT_TTL_SECONDS = 48 * 60 * 60
DEFAULT_TIMEOUT_SECONDS = 55
DEFAULT_DIRECTORY_MOUNT = "/mnt/retro_snapshot"


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_id: str
    kind: SnapshotKind
    path: str
    latency_seconds: float
    ttl_seconds: int | None
    ttl_enforced_by_sdk: bool
    estimated_bytes: int | None = None
    estimated_files: int | None = None


@dataclass(frozen=True)
class RestoreResult:
    latency_seconds: float
    copied_to_writable: bool
    mount_path: str
    target_path: str


def snapshot_sandbox(
    sandbox: Any,
    *,
    kind: SnapshotKind = SnapshotKind.DIRECTORY,
    path: str = "/app",
    ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> SnapshotResult:
    """Create one snapshot after an agent/tool turn has fully completed."""

    raw = _raw_sandbox(sandbox)
    estimated_bytes, estimated_files = workspace_stats(sandbox, path)
    started = time.perf_counter()
    try:
        if kind != SnapshotKind.DIRECTORY:  # pragma: no cover - enum has one member
            raise ValueError(f"unsupported snapshot kind {kind!r}")
        try:
            image = raw.snapshot_directory(path, timeout=timeout_seconds, ttl=ttl_seconds)
            ttl_enforced = True
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            image = raw.snapshot_directory(path)
            ttl_enforced = False
        snapshot_id = image.object_id
    except Exception as exc:  # noqa: BLE001 - normalize an external SDK boundary
        raise RuntimeError(f"snapshot_directory: {exc}") from exc
    if not snapshot_id:
        raise RuntimeError(f"snapshot_{kind.value}: Modal returned an empty image id")
    return SnapshotResult(
        snapshot_id=snapshot_id,
        kind=kind,
        path=path,
        latency_seconds=time.perf_counter() - started,
        ttl_seconds=ttl_seconds,
        ttl_enforced_by_sdk=ttl_enforced,
        estimated_bytes=estimated_bytes,
        estimated_files=estimated_files,
    )


def restore_directory(
    sandbox: Any,
    snapshot_id: str,
    *,
    target_path: str = "/app",
    mount_path: str = DEFAULT_DIRECTORY_MOUNT,
    copy_to_writable: bool = True,
) -> RestoreResult:
    """Mount a directory snapshot, optionally copying it into writable space.

    The smoke test decides whether direct writes through a mounted Image are
    supported reliably.  Production defaults to a writable copy so sibling
    sandboxes cannot mutate or contend on the mounted snapshot layer.
    """

    if not snapshot_id:
        raise ValueError("restore_directory requires snapshot_id")
    modal = _modal()
    raw = _raw_sandbox(sandbox)
    started = time.perf_counter()
    try:
        raw.mount_image(mount_path, modal.Image.from_id(snapshot_id))
        if copy_to_writable:
            _exec_checked(
                sandbox,
                f"rm -rf {_q(target_path)} && mkdir -p {_q(target_path)} "
                f"&& cp -a {_q(mount_path)}/. {_q(target_path)}/",
            )
            try:
                raw.unmount_image(mount_path)
            except Exception:  # noqa: BLE001 - copied state is already usable
                pass
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"restore_directory: {exc}") from exc
    return RestoreResult(
        latency_seconds=time.perf_counter() - started,
        copied_to_writable=copy_to_writable,
        mount_path=mount_path,
        target_path=target_path,
    )


def pack_staging(
    sandbox: Any,
    staging_root: str,
    *,
    archive_dir: str = "/tmp/retro_artifact",
    compression: str = "gzip",
    timeout_seconds: int = 300,
) -> dict[str, str]:
    """Pack the per-turn staging root into ONE tarball ready to snapshot.

    The bench (profiles/retro_snapshot_bench) showed hardlink dedup does NOT
    survive snapshot_directory → mount (23x expansion); tar stores hardlinks as
    link entries and compresses, so the trajectory artifact must be a single
    opaque file. Returns the manifest ``artifact`` dict.
    """

    if compression == "zstd":
        flag, ext = "--zstd", "tzst"
    elif compression == "gzip":
        flag, ext = "-z", "tgz"
    else:
        raise ValueError(f"unsupported artifact compression {compression!r}")
    root = staging_root.rstrip("/")
    parent, member_root = root.rsplit("/", 1)
    member = f"turns.{ext}"
    _exec_checked(
        sandbox,
        f"rm -rf {_q(archive_dir)} && mkdir -p {_q(archive_dir)} && "
        f"tar {flag} -cf {_q(archive_dir + '/' + member)} -C {_q(parent or '/')} {_q(member_root)}",
        timeout=timeout_seconds,
    )
    return {"member": member, "compression": compression, "member_root": member_root}


def restore_turn_from_artifact(
    sandbox: Any,
    snapshot_id: str,
    *,
    artifact: dict[str, Any],
    turn_index: int,
    target_path: str = "/app",
    mount_path: str = DEFAULT_DIRECTORY_MOUNT,
    timeout_seconds: int = 300,
) -> RestoreResult:
    """Mount the artifact snapshot, unpack the tarball, install one turn dir.

    Bench timings: mount+cp of the single tarball ~0.9 s, unpack ~0.5 s, turn
    copy ~0.15 s — faster than restoring an expanded staging directory.
    """

    if not snapshot_id:
        raise ValueError("restore_turn_from_artifact requires snapshot_id")
    member = str(artifact.get("member") or "")
    member_root = str(artifact.get("member_root") or "")
    if not member or not member_root:
        raise ValueError(f"artifact dict is incomplete: {artifact!r}")
    modal = _modal()
    raw = _raw_sandbox(sandbox)
    turn_dir = f"/tmp/retro_extract/{member_root}/{turn_index:04d}"
    started = time.perf_counter()
    try:
        raw.mount_image(mount_path, modal.Image.from_id(snapshot_id))
        _exec_checked(
            sandbox,
            f"rm -rf /tmp/retro_extract && mkdir -p /tmp/retro_extract && "
            f"tar -xf {_q(mount_path + '/' + member)} -C /tmp/retro_extract && "
            f"test -d {_q(turn_dir)} && "
            f"rm -rf {_q(target_path)} && mkdir -p {_q(target_path)} && "
            f"cp -a {_q(turn_dir)}/. {_q(target_path)}/ && "
            f"rm -rf /tmp/retro_extract",
            timeout=timeout_seconds,
        )
        try:
            raw.unmount_image(mount_path)
        except Exception:  # noqa: BLE001 - copied state is already usable
            pass
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"restore_turn_from_artifact(turn={turn_index}): {exc}") from exc
    return RestoreResult(
        latency_seconds=time.perf_counter() - started,
        copied_to_writable=True,
        mount_path=mount_path,
        target_path=target_path,
    )


def delete_snapshot(snapshot_id: str) -> None:
    if not snapshot_id:
        return
    try:
        import modal.experimental

        modal.experimental.image_delete(snapshot_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"image_delete: {exc}") from exc


def workspace_stats(sandbox: Any, path: str = "/app") -> tuple[int | None, int | None]:
    """Best-effort logical workspace bytes/file count for cost diagnostics."""

    try:
        rc, out, _ = _exec(sandbox, f"du -sk {_q(path)} 2>/dev/null | awk '{{print $1}}'")
        size = int(out.strip().splitlines()[-1]) * 1024 if rc == 0 and out.strip() else None
        rc, out, _ = _exec(sandbox, f"find {_q(path)} -xdev | wc -l")
        files = int(out.strip().splitlines()[-1]) if rc == 0 and out.strip() else None
        return size, files
    except (TypeError, ValueError, IndexError):
        return None, None


def sandbox_compute_cost(
    seconds: float,
    *,
    cpu_cores: float = 4.0,
    memory_gib: float = 4.0,
    cpu_dollars_per_core_second: float = 0.00003942,
    memory_dollars_per_gib_second: float = 0.00000667,
) -> float:
    """Public Modal Sandbox pricing estimate as of 2026-08-01."""

    return max(0.0, seconds) * (
        cpu_cores * cpu_dollars_per_core_second + memory_gib * memory_dollars_per_gib_second
    )


def _modal():
    try:
        import modal
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError("import modal: install the Modal SDK to use retro snapshots") from exc
    return modal


def _raw_sandbox(sandbox: Any) -> Any:
    return getattr(sandbox, "sb", sandbox)


def _exec(sandbox: Any, command: str) -> tuple[int, str, str]:
    if hasattr(sandbox, "exec") and hasattr(sandbox, "sb"):
        return sandbox.exec(command, check=False)
    process = sandbox.exec("bash", "-lc", command, text=True)
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    return process.wait(), stdout, stderr


def _exec_checked(sandbox: Any, command: str, timeout: int | None = None) -> None:
    if hasattr(sandbox, "exec") and hasattr(sandbox, "sb"):
        if timeout is None:
            sandbox.exec(command, cwd="/", check=True)
        else:
            sandbox.exec(command, cwd="/", check=True, timeout=timeout)
        return
    process = sandbox.exec("bash", "-lc", command, text=True)
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"exec in restored sandbox (rc={returncode}): {(stderr or stdout)[-500:]}")


def _q(value: str) -> str:
    import shlex

    return shlex.quote(value)
