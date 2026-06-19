"""Modal sandbox backend for agent rollouts.

``ModalSandbox`` is a drop-in ``Sandbox`` protocol impl backed by
``modal.Sandbox`` (the local analog of ``E2BSandbox``). Pure infrastructure, so
the env glue and agent runtimes build on it. Image is a registry ref or a
host-side Dockerfile build (``DockerfileImage``). ``modal`` is imported lazily
so this stays importable without Modal installed. Boot concurrency and
create-retry live here so every sandbox creation is gated/retried uniformly.

Env knobs
---------
    MODAL_BOOT_CONCURRENCY            max concurrent sandbox creates (default 8)
    MODAL_BOOT_RETRIES               transient-create retries (default 2)
    MODAL_BOOT_TIMEOUT_SEC           cap on sandbox boot/image-pull (default 600)
    MODAL_RPC_RETRIES                transient-exec retries (default 2)
    SLIME_AGENT_SANDBOX_LIFETIME_SEC sandbox max lifetime (default 3600)
        (legacy alias: MODAL_SANDBOX_LIFETIME_SEC)
    SLIME_AGENT_SANDBOX_MODAL_APP    Modal app name (default slime-agent-sandboxes)
        (legacy alias: MODAL_SANDBOX_APP_NAME)
    SLIME_AGENT_SANDBOX_BLOCK_NETWORK 1 to cut sandbox outbound network
        (legacy alias: MODAL_SANDBOX_BLOCK_NETWORK)
    SLIME_AGENT_SANDBOX_CPU          fractional cpu cores (optional)
    SLIME_AGENT_SANDBOX_MEMORY_MB    memory in MB (optional)
    SLIME_AGENT_SANDBOX_GPU          gpu spec, e.g. "a10g" (optional)
    SLIME_AGENT_SANDBOX_VM_RUNTIME   1 to boot a VM sandbox instead of gVisor
        (real kernel, allowlisted workspaces only; VM memory is static, floored
        at MODAL_VM_MEMORY_FLOOR_MB, default 2048)
    SLIME_AGENT_SANDBOX_ADD_PYTHON   add a python to the image (optional)
    MODAL_REGISTRY_SECRET            modal.Secret name for a private registry/ECR
    MODAL_ENVIRONMENT                modal environment name (optional)
"""

from __future__ import annotations

import asyncio
import codecs
import logging
import os
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


class SandboxBootTimeout(Exception):
    """Sandbox create/image-pull exceeded its boot budget (MODAL_BOOT_TIMEOUT_SEC)."""

    def __init__(self, timeout_sec: int, image: str = "") -> None:
        self.timeout_sec = timeout_sec
        super().__init__(f"sandbox boot exceeded {timeout_sec}s: {image}")


@dataclass(frozen=True)
class DockerfileImage:
    """Build-from-Dockerfile image spec (harbor-style task environments).

    ``context_dir`` defaults to the Dockerfile's dir. Modal content-hashes the
    Dockerfile + context so identical task files cache-hit across boots. FROM
    pull uses Modal's default builder auth (no private base images).
    """

    path: str
    context_dir: str | None = None

    @property
    def description(self) -> str:
        return f"dockerfile:{self.path}"


# Modal validates exec argv against ARG_MAX (64 KiB) client-side; larger
# commands are staged as a script file. Under the limit to leave room for the
# bash/runuser wrapper argv.
_EXEC_ARGV_LIMIT_BYTES = 32_768


def _normalize_image_ref(ref: str) -> str:
    """Lowercase a registry ref's repository name; preserve tag/digest.

    OCI repo names must be lowercase, but some SWE-bench dataset images carry
    mixed-case orgs/repos. Tag/digest are case-sensitive and left untouched.
    """
    if not ref:
        return ref
    name, sep, suffix = ref, "", ""
    if "@" in ref:  # digest pin: name@sha256:...
        name, _, digest = ref.partition("@")
        sep, suffix = "@", digest
    else:
        slash = ref.rfind("/")
        colon = ref.rfind(":")
        if colon > slash:  # a tag colon, not a registry :port
            name, sep, suffix = ref[:colon], ":", ref[colon + 1 :]
    return name.lower() + sep + suffix


def _getenv(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


def _getenv_int(*names: str, default: int) -> int:
    raw = _getenv(*names)
    return int(raw) if raw else default


# Process-wide create gate + cached App, lazily created on the running loop.
_BOOT_SEM: asyncio.Semaphore | None = None
_APP_CACHE: dict[str, Any] = {}
_APP_LOCK: asyncio.Lock | None = None


def _boot_sem() -> asyncio.Semaphore:
    global _BOOT_SEM
    if _BOOT_SEM is None:
        _BOOT_SEM = asyncio.Semaphore(_getenv_int("MODAL_BOOT_CONCURRENCY", default=8))
    return _BOOT_SEM


def _app_lock() -> asyncio.Lock:
    global _APP_LOCK
    if _APP_LOCK is None:
        _APP_LOCK = asyncio.Lock()
    return _APP_LOCK


class ModalSandbox:
    """Async context manager around ``modal.Sandbox`` (the ``Sandbox`` protocol).

    Command failures surface as exit codes; transient Modal transport errors are
    retried so infra problems are never scored as a failed test.
    """

    default_lifetime_sec = 3600
    default_boot_timeout_sec = 600
    default_app_name = "slime-agent-sandboxes"
    default_create_retries = 2
    default_rpc_retries = 2
    # Main process that keeps the sandbox alive (exec runs separate processes).
    # Required for images that blank their entrypoint (see __aenter__).
    keepalive_command = ("sleep", "infinity")
    rpc_backoff_base_sec = 1.0
    # Per-stream output cap so a runaway command can't balloon host memory.
    output_cap_chars = _getenv_int("MODAL_EXEC_OUTPUT_CAP", default=1_000_000)

    def __init__(
        self,
        image: str | DockerfileImage,
        *,
        timeout: int | None = None,
        block_network: bool | None = None,
        cpu: float | None = None,
        memory_mb: int | None = None,
        gpu: str | None = None,
        registry_secret: str | None = None,
        rpc_retries: int | None = None,
        create_retries: int | None = None,
        app_name: str | None = None,
        add_python: str | None = None,
        workdir: str | None = None,
        vm_runtime: bool | None = None,
        boot_timeout: int | None = None,
    ) -> None:
        if isinstance(image, DockerfileImage):
            self.image_spec: DockerfileImage | None = image
            self.image = image.description  # label only
        else:
            self.image_spec = None
            self.image = _normalize_image_ref(image)
        self.timeout = timeout if timeout is not None else self._lifetime_from_env()
        self.boot_timeout = boot_timeout if boot_timeout is not None else self._boot_timeout_from_env()
        self.block_network = block_network if block_network is not None else self._block_network_from_env()
        self.cpu = cpu if cpu is not None else self._float_from_env("SLIME_AGENT_SANDBOX_CPU", "MODAL_SANDBOX_CPU")
        self.memory_mb = (
            memory_mb
            if memory_mb is not None
            else self._int_from_env("SLIME_AGENT_SANDBOX_MEMORY_MB", "MODAL_SANDBOX_MEMORY_MB")
        )
        self.gpu = gpu or (_getenv("SLIME_AGENT_SANDBOX_GPU", "MODAL_SANDBOX_GPU") or None)
        self.vm_runtime = vm_runtime if vm_runtime is not None else self._vm_runtime_from_env()
        if self.vm_runtime and self.memory_mb is None:
            # VM memory is static; Modal's 128MB default OOMs a VM.
            self.memory_mb = _getenv_int("MODAL_VM_MEMORY_FLOOR_MB", default=2048)
        self.registry_secret = registry_secret or (_getenv("MODAL_REGISTRY_SECRET") or None)
        self.rpc_retries = (
            rpc_retries
            if rpc_retries is not None
            else _getenv_int("MODAL_RPC_RETRIES", "SLIME_AGENT_SANDBOX_RPC_RETRIES", default=self.default_rpc_retries)
        )
        self.create_retries = (
            create_retries
            if create_retries is not None
            else _getenv_int("MODAL_BOOT_RETRIES", default=self.default_create_retries)
        )
        self.app_name = app_name or _getenv(
            "SLIME_AGENT_SANDBOX_MODAL_APP", "MODAL_SANDBOX_APP_NAME", default=self.default_app_name
        )
        self.add_python = add_python or (_getenv("SLIME_AGENT_SANDBOX_ADD_PYTHON", "MODAL_SANDBOX_ADD_PYTHON") or None)
        self.workdir = workdir or (_getenv("SLIME_AGENT_SANDBOX_WORKDIR", "MODAL_SANDBOX_WORKDIR") or None)
        self._modal = None
        self._sb = None
        self.sandbox_id = ""

    # -- env helpers --------------------------------------------------------
    @classmethod
    def _lifetime_from_env(cls) -> int:
        return _getenv_int(
            "SLIME_AGENT_SANDBOX_LIFETIME_SEC", "MODAL_SANDBOX_LIFETIME_SEC", default=cls.default_lifetime_sec
        )

    @classmethod
    def _boot_timeout_from_env(cls) -> int:
        return _getenv_int("MODAL_BOOT_TIMEOUT_SEC", default=cls.default_boot_timeout_sec)

    @staticmethod
    def _vm_runtime_from_env() -> bool:
        return _getenv("SLIME_AGENT_SANDBOX_VM_RUNTIME", "MODAL_SANDBOX_VM_RUNTIME").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    @staticmethod
    def _block_network_from_env() -> bool:
        return _getenv("SLIME_AGENT_SANDBOX_BLOCK_NETWORK", "MODAL_SANDBOX_BLOCK_NETWORK").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    @staticmethod
    def _float_from_env(*names: str) -> float | None:
        raw = _getenv(*names)
        return float(raw) if raw else None

    @staticmethod
    def _int_from_env(*names: str) -> int | None:
        raw = _getenv(*names)
        return int(raw) if raw else None

    # -- transient-error classification ------------------------------------
    @staticmethod
    def _is_transient(e: BaseException) -> bool:
        """True if ``e`` is a Modal transport flap safe to retry (command-level
        timeouts are NOT transient)."""
        name = type(e).__name__
        if "SandboxTimeout" in name or name == "TimeoutError":
            return False
        if name in {
            "ConnectionError",
            "ConnectionResetError",
            "ConnectionAbortedError",
            "GRPCError",
            "StreamTerminatedError",
            "InternalError",
            "ServerError",
            "RemoteError",
        }:
            return True
        msg = str(e).lower()
        return any(s in msg for s in ("connection", "unavailable", "stream terminated", "goaway", "reset"))

    async def _retry(self, op_name: str, attempts: int, coro_factory):
        last_err: BaseException | None = None
        for attempt in range(max(1, attempts)):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient(e) or attempt + 1 >= max(1, attempts):
                    raise
                last_err = e
                backoff = self.rpc_backoff_base_sec * (2**attempt)
                logger.debug(
                    "[modal_sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                    op_name,
                    type(e).__name__,
                    attempt + 1,
                    attempts,
                    backoff,
                    str(e)[:160],
                )
                await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    # -- lifecycle ----------------------------------------------------------
    async def _get_app(self):
        environment_name = _getenv("MODAL_ENVIRONMENT") or None
        key = f"{self.app_name}\0{environment_name or ''}"
        async with _app_lock():
            app = _APP_CACHE.get(key)
            if app is None:
                kwargs: dict[str, Any] = {"create_if_missing": True}
                if environment_name:
                    kwargs["environment_name"] = environment_name
                app = await self._modal.App.lookup.aio(self.app_name, **kwargs)
                _APP_CACHE[key] = app
            return app

    def _build_image(self):
        modal = self._modal
        kwargs: dict[str, Any] = {}
        if self.add_python:
            kwargs["add_python"] = self.add_python
        if self.image_spec is not None:
            spec = self.image_spec
            context_dir = spec.context_dir or str(Path(spec.path).parent)
            return modal.Image.from_dockerfile(spec.path, context_dir=context_dir, **kwargs)
        secret = None
        if self.registry_secret:
            secret = modal.Secret.from_name(self.registry_secret)
        if ".dkr.ecr." in self.image and secret is not None:
            return modal.Image.from_aws_ecr(self.image, secret=secret, **kwargs)
        return modal.Image.from_registry(self.image, secret=secret, **kwargs)

    async def __aenter__(self) -> ModalSandbox:
        import modal  # lazy

        self._modal = modal
        app = await self._get_app()
        image = self._build_image()

        create_kwargs: dict[str, Any] = {
            "app": app,
            "image": image,
            "timeout": self.timeout,
            "block_network": self.block_network,
        }
        if self.cpu is not None:
            create_kwargs["cpu"] = self.cpu
        if self.memory_mb is not None:
            create_kwargs["memory"] = self.memory_mb
        if self.gpu:
            create_kwargs["gpu"] = self.gpu
        if self.workdir:
            create_kwargs["workdir"] = self.workdir
        if self.vm_runtime:
            create_kwargs["experimental_options"] = {"vm_runtime": True}

        async def _create():
            async with _boot_sem():
                # Explicit keepalive command. Some task images (SWE-bench-Pro)
                # blank the entrypoint (`ENTRYPOINT []`) expecting an external
                # `sleep infinity` from docker-compose; with no command Modal runs
                # the now-empty entrypoint and the container exits immediately
                # (rc 128) -> every later exec hits "sandbox already shut down".
                # `exec` spawns its own processes, so this is inert for images
                # that already stay up (SWE-bench verified).
                return await modal.Sandbox.create.aio(*self.keepalive_command, **create_kwargs)

        try:
            async with asyncio.timeout(self.boot_timeout):
                self._sb = await self._retry(f"create({self.image[:48]!r})", self.create_retries, _create)
        except TimeoutError:
            raise SandboxBootTimeout(self.boot_timeout, self.image[:48]) from None
        self.sandbox_id = str(getattr(self._sb, "object_id", "") or "")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        sb = self._sb
        if sb is None:
            return
        try:
            await sb.terminate.aio()
        except Exception as e:
            logger.warning("[modal_sandbox] terminate %s failed: %s", self.sandbox_id[:8], e)
        try:
            await sb.wait.aio(raise_on_termination=False)
        except Exception:
            pass

    # -- protocol surface ---------------------------------------------------
    def _require_sandbox(self):
        if self._sb is None:
            raise RuntimeError("ModalSandbox has not been entered")
        return self._sb

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        sb = self._require_sandbox()
        # Honor user= for agents that drop privileges.
        if user and user != "root":
            inner = f"runuser -u {shlex.quote(user)} -- bash -lc {shlex.quote(cmd)}"
        else:
            inner = cmd
        if len(inner.encode("utf-8", errors="ignore")) > _EXEC_ARGV_LIMIT_BYTES:
            # Too big for exec argv: stage as a script. Left behind so _retry can
            # re-run; sandboxes are ephemeral.
            script = f"/tmp/.modal_exec_{uuid.uuid4().hex}.sh"
            await self.write_file(script, inner)
            inner = f"bash {shlex.quote(script)}"
        secrets = [self._modal.Secret.from_dict({str(k): str(v) for k, v in env.items()})] if env else []

        async def _run() -> ExecResult:
            # text=False: Modal's text mode decodes strictly, so one non-UTF8
            # byte kills the rollout. Take bytes, decode host-side with replace.
            proc = await sb.exec.aio("bash", "-lc", inner, timeout=timeout, secrets=secrets, text=False)
            # Drain both streams BEFORE wait(): a full pipe buffer deadlocks wait().
            out_task = asyncio.create_task(_read_stream_capped(proc.stdout, self.output_cap_chars))
            err_task = asyncio.create_task(_read_stream_capped(proc.stderr, self.output_cap_chars))
            exit_code = int(await proc.wait.aio())
            stdout, stderr = await asyncio.gather(out_task, err_task)
            return exit_code, stdout, stderr

        exit_code, stdout, stderr = await self._retry(f"exec({cmd[:48]!r})", self.rpc_retries, _run)
        if check and exit_code != 0:
            raise RuntimeError(f"modal exec failed (exit={exit_code}): {cmd[:120]}\n{stderr[-1000:]}")
        return exit_code, stdout, stderr

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        sb = self._require_sandbox()
        fs = sb.filesystem

        async def _write():
            if isinstance(content, Path):
                await fs.copy_from_local.aio(str(content), sandbox_path)
            elif isinstance(content, bytes):
                await fs.write_bytes.aio(content, sandbox_path)
            else:
                await fs.write_text.aio(str(content), sandbox_path)

        await self._retry(f"write_file({sandbox_path})", self.rpc_retries, _write)
        if user and user != "root":
            quoted = shlex.quote(user)
            await self.exec(f"chown {quoted}:{quoted} {shlex.quote(sandbox_path)}", timeout=30, check=False)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        del user  # advisory: Modal reads as the sandbox owner
        sb = self._require_sandbox()
        try:
            return await self._retry(
                f"read_file({sandbox_path})",
                self.rpc_retries,
                lambda: sb.filesystem.read_text.aio(sandbox_path),
            )
        except Exception:
            return ""


async def _read_stream_capped(stream: Any, cap: int) -> str:
    """Drain ``stream`` fully but keep only the first ``cap`` chars (tail dropped
    with a marker). Decodes byte chunks incrementally with ``errors="replace"``
    so a split multibyte char doesn't mojibake and non-UTF8 never raises.
    """
    if stream is None:
        return ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    parts: list[str] = []
    total = 0
    truncated = False
    async for chunk in stream:
        if chunk is None:
            continue
        text = decoder.decode(chunk) if isinstance(chunk, (bytes, bytearray)) else chunk
        if total < cap:
            take = text[: cap - total]
            parts.append(take)
            total += len(take)
            if total >= cap:
                truncated = True
        else:
            truncated = True
    out = "".join(parts)
    if truncated:
        out += "\n...[truncated]"
    return out
