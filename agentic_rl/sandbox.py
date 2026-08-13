"""Modal Sandbox adapter for mini-swe-agent's bash environment protocol."""

import os
import shlex
import threading
import time

import modal

from minisweagent.exceptions import Submitted

from .prompts import SUBMIT_SENTINEL

_EXEC_GRACE_SEC = 30


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
        cpu: float | None = None,
        memory_mb: int | None = None,
        boot_retries: int = 2,
    ):
        if cpu is None and os.environ.get("SLIME_AGENT_SANDBOX_CPU"):
            cpu = float(os.environ["SLIME_AGENT_SANDBOX_CPU"])
        if memory_mb is None and os.environ.get("SLIME_AGENT_SANDBOX_MEMORY_MB"):
            memory_mb = int(os.environ["SLIME_AGENT_SANDBOX_MEMORY_MB"])

        app = modal.App.lookup(app_name, create_if_missing=True)
        t0 = time.perf_counter()
        _create = getattr(modal.Sandbox, "_experimental_create", modal.Sandbox.create)
        kwargs = {
            "image": modal.Image.from_registry(image),
            "app": app,
            "timeout": lifetime,
        }
        if cpu is not None:
            kwargs["cpu"] = cpu
        if memory_mb is not None:
            kwargs["memory"] = memory_mb
        self.sb = self._create_with_retry(_create, kwargs, boot_retries)
        self.boot_time = time.perf_counter() - t0
        self.cwd = cwd
        self.exec_timeout = exec_timeout
        self.exec_time = 0.0
        self.exec_timeouts = 0
        # Armed for the duration of an agent leg so a single command cannot
        # outlive the episode's remaining wall-clock budget.
        self.deadline: float | None = None

    @staticmethod
    def _create_with_retry(create, kwargs: dict, retries: int):
        last_error = None
        for attempt in range(retries + 1):
            try:
                return create("sleep", "infinity", **kwargs)
            except Exception as error:
                last_error = error
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"sandbox boot after {retries + 1} attempts: {last_error}")

    def exec(self, command: str, *, cwd: str | None = None, timeout: int | None = None) -> tuple[int, str]:
        t0 = time.perf_counter()
        command = command.replace("\x00", "")  # a NUL is never valid in a shell command; Modal's exec rejects it
        budget = timeout or self.exec_timeout
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                return 124, "command not run: agent time budget exhausted"
            budget = min(budget, max(1, int(remaining)))
        workdir = shlex.quote(cwd or self.cwd)

        def run() -> tuple[int, str]:
            # text=False: task commands can emit non-UTF-8 or binary diffs.
            process = self.sb.exec(
                "bash",
                "-lc",
                f"cd {workdir} && {command}",
                timeout=budget,
                text=False,
            )
            output = (process.stdout.read() + process.stderr.read()).decode(
                "utf-8",
                errors="replace",
            )
            return process.wait(), output

        try:
            rc, out = _run_with_timeout(
                run,
                budget + _EXEC_GRACE_SEC,
                f"exec({command[:80]})",
            )
        except TimeoutError:
            self.exec_timeouts += 1
            rc, out = 124, f"command timed out after {budget}s (sandbox unresponsive)"
        self.exec_time += time.perf_counter() - t0
        return rc, out

    def write_file(self, path: str, content: str) -> None:
        def write() -> None:
            # Stream via stdin so large patches do not hit ARG_MAX and the same
            # implementation works with both Modal Sandbox APIs.
            process = self.sb.exec(
                "bash",
                "-lc",
                f"cat > {shlex.quote(path)}",
                text=False,
            )
            data = content.encode()
            chunk = 1 << 20
            for offset in range(0, len(data), chunk):
                process.stdin.write(data[offset : offset + chunk])
                process.stdin.drain()
            process.stdin.write_eof()
            process.stdin.drain()
            process.wait()

        _run_with_timeout(write, self.exec_timeout, f"write_file({path})")

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


def _run_with_timeout(fn, timeout_sec: float, operation: str):
    """Bound a blocking Modal RPC whose stream has no client deadline."""

    result: dict = {}
    done = threading.Event()

    def run() -> None:
        try:
            result["value"] = fn()
        except BaseException as error:
            result["error"] = error
        finally:
            done.set()

    threading.Thread(target=run, name="modal-rpc", daemon=True).start()
    if not done.wait(timeout_sec):
        raise TimeoutError(f"{operation} exceeded {timeout_sec}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")
