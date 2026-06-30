"""Modal sandbox booted from the task image; also mini-swe-agent's bash Environment
(duck-typed). One per rollout, run synchronously in a worker thread."""

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
        self.sb = modal.Sandbox.create(
            "sleep", "infinity", image=modal.Image.from_registry(image), app=app, timeout=lifetime
        )
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
        self.sb.filesystem.write_text(content, path)  # RPC, not a shell arg (patches exceed ARG_MAX)

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
