"""Modal sandbox booted from the task image; also mini-swe-agent's bash
``Environment`` (duck-typed). One per rollout, run synchronously in a worker
thread. The sandbox is a bare bash executor — the agent loop runs in-process on
the head node, so nothing is provisioned inside.

Beyond agentic_rl's original 65-line executor this keeps the hardening the
general task families need: Dockerfile-built images, per-sandbox env injection
(FrontierCS judge URL), cpu/memory sizing, the gVisor/vm_runtime toggle, boot
retries, and separate stdout/stderr with an output cap.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass

import modal

from minisweagent.exceptions import Submitted

from .prompts import SUBMIT_SENTINEL

DEFAULT_APP = "agentic-rl-sandboxes"
OUTPUT_CAP = 1_000_000  # per-stream char cap; oversized output is head+tail elided


@dataclass(frozen=True)
class DockerfileImage:
    """Build the image from a task Dockerfile instead of a registry ref."""

    path: str
    context_dir: str


class SandboxBootError(RuntimeError):
    """Sandbox creation failed after all retries; the sample aborts + recycles."""


def _build_image(image: str | DockerfileImage):
    if isinstance(image, DockerfileImage):
        return modal.Image.from_dockerfile(image.path, context_dir=image.context_dir)
    return modal.Image.from_registry(image)


class Sandbox:
    config = None  # mini-swe Environment protocol

    def __init__(
        self,
        image: str | DockerfileImage,
        *,
        cwd: str = "/",
        lifetime: int = 1800,
        exec_timeout: int = 120,
        app_name: str = DEFAULT_APP,
        env: dict[str, str] | None = None,
        cpu: float | None = None,
        memory_mb: int | None = None,
        vm_runtime: bool = False,
        boot_retries: int = 2,
    ):
        app = modal.App.lookup(app_name, create_if_missing=True)
        kwargs: dict = {"image": _build_image(image), "app": app, "timeout": lifetime}
        if env:
            kwargs["secrets"] = [modal.Secret.from_dict(dict(env))]
        if cpu:
            kwargs["cpu"] = float(cpu)
        if memory_mb:
            kwargs["memory"] = int(memory_mb)
        if vm_runtime:
            kwargs["experimental_options"] = {"vm_runtime": True}

        t0 = time.perf_counter()
        self.sb = self._create_with_retry(kwargs, boot_retries)
        self.boot_time = time.perf_counter() - t0
        self.cwd = cwd
        self.exec_timeout = exec_timeout
        self.exec_time = 0.0  # cumulative bash wall-time

    @staticmethod
    def _create_with_retry(kwargs: dict, retries: int):
        # Keepalive "sleep infinity": some task images blank ENTRYPOINT, so a
        # sandbox with no foreground command exits rc128 on boot.
        last = None
        for attempt in range(retries + 1):
            try:
                return modal.Sandbox.create("sleep", "infinity", **kwargs)
            except Exception as e:  # noqa: BLE001 - transient boot errors retried, then surfaced
                last = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        raise SandboxBootError(f"sandbox boot failed after {retries + 1} attempts: {last}")

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Run ``command`` in a login shell; return ``(returncode, stdout, stderr)``."""
        t0 = time.perf_counter()
        prefix = "".join(f"export {shlex.quote(k)}={shlex.quote(v)}; " for k, v in (env or {}).items())
        # text=False: commands can emit non-UTF-8 (binary diffs) that str-decode would raise on.
        p = self.sb.exec(
            "bash", "-lc", f"{prefix}cd {shlex.quote(cwd or self.cwd)} && {command}", timeout=timeout or self.exec_timeout, text=False
        )
        out = _cap(p.stdout.read().decode("utf-8", errors="replace"))
        err = _cap(p.stderr.read().decode("utf-8", errors="replace"))
        rc = p.wait()
        self.exec_time += time.perf_counter() - t0
        if check and rc != 0:
            raise RuntimeError(f"command failed (rc={rc}): {command[:120]}\n{err[-500:]}")
        return rc, out, err

    def write_file(self, path: str, content) -> None:
        # RPC, not a shell arg: patches/tarballs exceed ARG_MAX.
        if isinstance(content, (bytes, bytearray)):
            self.sb.filesystem.write_bytes(bytes(content), path)
        elif hasattr(content, "read") or self._is_pathlike(content):
            with open(content, "rb") as fh:
                self.sb.filesystem.write_bytes(fh.read(), path)
        else:
            self.sb.filesystem.write_text(content, path)

    @staticmethod
    def _is_pathlike(content) -> bool:
        import os

        return isinstance(content, (str, os.PathLike)) and len(str(content)) < 4096 and os.path.exists(str(content))

    def read_file(self, path: str) -> str:
        try:
            return self.sb.filesystem.read_text(path)
        except Exception:  # noqa: BLE001 - missing file -> empty, the env decides
            return ""

    # mini-swe Environment protocol -----------------------------------------
    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        rc, out, err = self.exec(action.get("command", ""), cwd=cwd or self.cwd, timeout=timeout or self.exec_timeout)
        output = out if not err else (out + ("\n" if out else "") + err)
        lines = output.lstrip().splitlines()
        if lines and lines[0].strip() == SUBMIT_SENTINEL and rc == 0:
            raise Submitted({"role": "exit", "content": "", "extra": {"exit_status": "Submitted", "submission": ""}})
        return {"output": output, "returncode": rc, "exception_info": ""}

    def get_template_vars(self, **kwargs) -> dict:
        return {"system": "Linux", "release": "", "version": "", "machine": "x86_64", "cwd": self.cwd}

    def serialize(self) -> dict:
        return {}

    def terminate(self) -> None:
        try:
            self.sb.terminate()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.terminate()


def _cap(text: str) -> str:
    if len(text) <= OUTPUT_CAP:
        return text
    half = OUTPUT_CAP // 2
    return f"{text[:half]}\n...[{len(text) - OUTPUT_CAP} chars elided]...\n{text[-half:]}"
