"""Boot the verifier server (Node + go-judge) once per worker.

``ensure_started()`` is called by ``FrontierCsEnv`` on its first rollout. It:
  * returns ``FRONTIER_CS_JUDGE_URL`` immediately if a pre-deployed judge is set
    (the standalone-deploy path), else
  * boots the vendored ``server/`` as a ``vm_runtime`` Modal Sandbox, mounts the
    ``slime-data`` Volume at ``/data`` and points the go-judge's ``problemsRoot`` at
    ``/data/frontier_cs/problems`` (``PROBLEMS_ROOT``), waits for ``/health``, exports
    ``FRONTIER_CS_JUDGE_URL`` to the tunnel URL, and returns it.

The problem testdata lives on the SAME ``slime-data`` Volume as the prompt jsonl +
task trees (pulled by the config's ``download_data``: ``frontier_cs/problems/**``),
so there is no separate volume or populate step. The judge boots INSIDE the training
job (one per worker), shares the run's lifecycle, and dies with it. go-judge needs a
real kernel; ``vm_runtime`` provides it (allowlisted workspaces) — if the workspace
isn't allowlisted the boot raises loudly; fall back to a pre-deployed
``FRONTIER_CS_JUDGE_URL``.

Env knobs
---------
    FRONTIER_CS_JUDGE_URL              pre-deployed judge; if set, autostart is skipped
    FRONTIER_CS_DATA_VOLUME            Volume holding frontier_cs/problems/ (default slime-data)
    FRONTIER_CS_PROBLEMS_ROOT          problemsRoot inside the judge (default /data/frontier_cs/problems)
    FRONTIER_CS_JUDGE_APP              Modal app name (default frontier-cs-judge)
    FRONTIER_CS_JUDGE_CPU / _MEMORY_MB sandbox resources (default 64 / 131072; gives go-judge
                                       64 workers — max-headroom for burst grading (best latency
                                       + throughput, 0 errors with the gojudge.js queue-full
                                       retry). 2GB/worker. Profiled knee is ~32; 16 is a thriftier
                                       option.)
    FRONTIER_CS_JUDGE_LIFETIME_SEC     sandbox max lifetime (default 86400)
    FRONTIER_CS_JUDGE_BOOT_TIMEOUT_SEC wait for /health (default 900; covers first image build)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

URL_ENV = "FRONTIER_CS_JUDGE_URL"
_JUDGE_PORT = 8081
_DATA_MOUNT = "/data"
_SERVER_DIR = Path(__file__).parent / "server"

# Module-global so the judge boots once per worker and the handle outlives the
# call; guarded by a lock so concurrent first-rollouts don't double-boot.
_JUDGE_SANDBOX = None
_LOCK = threading.Lock()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def ensure_started() -> str:
    """Return the verifier-server URL, booting it once per worker if needed.

    Idempotent and thread-safe. Raises if the boot fails (e.g. vm_runtime not
    allowlisted) so the caller aborts loudly rather than rolling out blind.
    """
    url = os.environ.get(URL_ENV, "").strip()
    if url:
        return url
    global _JUDGE_SANDBOX
    with _LOCK:
        url = os.environ.get(URL_ENV, "").strip()  # re-check under lock
        if url:
            return url
        if _JUDGE_SANDBOX is None:
            _JUDGE_SANDBOX = _boot_judge_sandbox()
        return os.environ[URL_ENV]


def _boot_judge_sandbox():
    import modal

    app_name = os.environ.get("FRONTIER_CS_JUDGE_APP", "frontier-cs-judge")
    volume_name = os.environ.get("FRONTIER_CS_DATA_VOLUME", "slime-data")
    problems_root = os.environ.get("FRONTIER_CS_PROBLEMS_ROOT", "/data/frontier_cs/problems")
    environment_name = os.environ.get("MODAL_ENVIRONMENT") or None

    # Mount slime-data and point the go-judge's problemsRoot into it (download_data
    # pulls frontier_cs/problems/** here), so no separate problems Volume is needed.
    image = modal.Image.from_dockerfile(str(_SERVER_DIR / "Dockerfile"), context_dir=str(_SERVER_DIR)).env(
        {"PROBLEMS_ROOT": problems_root}
    )
    data_vol = modal.Volume.from_name(volume_name)  # must exist + hold frontier_cs/problems/ (download_data)
    app_kwargs = {"create_if_missing": True}
    if environment_name:
        app_kwargs["environment_name"] = environment_name
    app = modal.App.lookup(app_name, **app_kwargs)

    logger.info(
        "[verifier_server] booting judge sandbox (app=%s, volume=%s, problems_root=%s, vm_runtime)…",
        app_name, volume_name, problems_root,
    )
    sb = modal.Sandbox.create(
        "/app/entrypoint.sh",  # go-judge + node server (image ENTRYPOINT); foreground node keeps it up
        app=app,
        image=image,
        volumes={_DATA_MOUNT: data_vol},
        cpu=float(_int_env("FRONTIER_CS_JUDGE_CPU", 64)),
        memory=_int_env("FRONTIER_CS_JUDGE_MEMORY_MB", 131072),
        timeout=_int_env("FRONTIER_CS_JUDGE_LIFETIME_SEC", 86400),
        encrypted_ports=[_JUDGE_PORT],
        experimental_options={"vm_runtime": True},
    )
    try:
        url = sb.tunnels()[_JUDGE_PORT].url
        _wait_healthy(url, timeout_sec=_int_env("FRONTIER_CS_JUDGE_BOOT_TIMEOUT_SEC", 900))
    except BaseException:
        sb.terminate()
        raise
    os.environ[URL_ENV] = url
    logger.info("[verifier_server] judge sandbox %s ready at %s", (sb.object_id or "")[:12], url)
    return sb


def _wait_healthy(base_url: str, *, timeout_sec: int) -> None:
    """Poll <base_url>/health until it returns ok, or raise after timeout_sec."""
    health = base_url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout_sec
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=10) as resp:
                if resp.status == 200 and json.loads(resp.read() or b"{}").get("ok"):
                    return
                last = f"HTTP {resp.status}"
        except (urllib.error.URLError, ValueError, TimeoutError) as e:
            last = type(e).__name__
        time.sleep(5)
    raise RuntimeError(f"verifier server not healthy within {timeout_sec}s ({health}; last={last})")
