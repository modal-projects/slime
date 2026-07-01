"""Modal sandbox booted from the task image; also mini-swe-agent's bash Environment
(duck-typed). One per rollout, run synchronously in a worker thread."""

import shlex
import time

import modal

from minisweagent.exceptions import Submitted

from .prompts import SUBMIT_SENTINEL


class Sandbox:
    config = None  # mini-swe Environment protocol

    def __init__(
        self,
        image: str,
        *,
        cwd: str = "/",
        lifetime: int = 1800,
        exec_timeout: int = 120,
        app_name: str = "agentic-rl-sandboxes",
    ):
        app = modal.App.lookup(app_name, create_if_missing=True)
        t0 = time.perf_counter()
        # Sandbox v2 (_experimental_create): higher create throughput + lower/tighter time-to-interactive,
        # designed for our >256-concurrent-sandbox regime where v1's control plane flaked. We use only the
        # v2-supported surface (create/exec/wait/terminate/stdout/stderr/stdin) — see write_file, which streams
        # via stdin because v2 drops the v1 `.filesystem` API. Fall back to v1 if the image's modal lacks v2.
        _create = getattr(modal.Sandbox, "_experimental_create", modal.Sandbox.create)
        self.sb = _create("sleep", "infinity", image=modal.Image.from_registry(image), app=app, timeout=lifetime)
        self.boot_time = time.perf_counter() - t0
        self.cwd = cwd
        self.exec_timeout = exec_timeout
        self.exec_time = 0.0  # cumulative bash wall-time

    def exec(self, command: str, *, cwd: str | None = None, timeout: int = 120) -> tuple[int, str]:
        t0 = time.perf_counter()
        command = command.replace("\x00", "")  # a NUL is never valid in a shell command; Modal's exec rejects it
        # text=False: commands can emit non-UTF-8 (binary diffs); Modal's str decode would raise.
        p = self.sb.exec("bash", "-lc", f"cd {cwd or self.cwd} && {command}", timeout=timeout, text=False)
        out = (p.stdout.read() + p.stderr.read()).decode("utf-8", errors="replace")
        rc = p.wait()
        self.exec_time += time.perf_counter() - t0
        return rc, out

    def write_file(self, path: str, content: str) -> None:
        # Stream via stdin (not a shell arg) so large patches don't hit ARG_MAX. Avoids the `.filesystem`
        # RPC, which Sandbox v2 does not expose — this path works on both v1 and v2 sandboxes.
        p = self.sb.exec("bash", "-lc", f"cat > {shlex.quote(path)}", text=False)
        data = content.encode()
        chunk = 1 << 20  # 1 MiB: drain per chunk so large patches don't exceed Modal's stdin buffer limit
        for i in range(0, len(data), chunk):
            p.stdin.write(data[i : i + chunk])
            p.stdin.drain()
        p.stdin.write_eof()
        p.stdin.drain()
        p.wait()

    # mini-swe Environment protocol
    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        rc, output = self.exec(action.get("command", ""), cwd=cwd or self.cwd, timeout=timeout or self.exec_timeout)
        lines = output.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_SENTINEL and rc == 0:
            submission = "".join(lines[1:])  # the curated patch the agent cat-ed after the sentinel
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return {"output": output, "returncode": rc, "exception_info": ""}

    def get_template_vars(self, **kwargs) -> dict:
        return {"system": "Linux", "release": "", "version": "", "machine": "x86_64", "cwd": self.cwd}

    def serialize(self) -> dict:
        return {}

    def terminate(self) -> None:
        try:
            self.sb.terminate()
        except Exception:
            pass
