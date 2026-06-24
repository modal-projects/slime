"""Modal web app for browsing slime SWE rollout debug dumps.

Serves the Bun/TypeScript dashboard over the ``slime-checkpoints`` volume so
each run's ``rollout_N.pt`` dumps can be inspected as agent conversations.

Deploy/iterate (volume lives in the ``junlin-dev`` environment)::

    MODAL_ENVIRONMENT=junlin-dev modal deploy async_rl_research/dashboard/app.py
    MODAL_ENVIRONMENT=junlin-dev modal serve  async_rl_research/dashboard/app.py  # hot-reload
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import modal

PORT = 3000
DUMP_ROOT = "/vol/swe_rollout_dumps"
DASHBOARD_DIR = Path(__file__).parent

app = modal.App("swe-rollout-dashboard")
volume = modal.Volume.from_name("slime-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip", "ca-certificates")
    .run_commands("curl -fsSL https://bun.sh/install | bash")
    # CPU wheel only: convert.py just unpickles sample dicts.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    # Local code last so edits don't bust the bun/torch layers.
    .add_local_dir(
        DASHBOARD_DIR,
        remote_path="/dashboard",
        ignore=["__pycache__", "node_modules", ".DS_Store", "*.pyc"],
    )
)


@app.function(
    image=image,
    volumes={"/vol": volume},
    # One container so the /tmp .pt -> JSON cache is shared.
    max_containers=1,
    scaledown_window=20 * 60,
)
@modal.web_server(port=PORT, startup_timeout=120)
def dashboard():
    def reload_volume_forever():
        while True:
            time.sleep(20)
            try:
                volume.reload()
            except Exception as e:
                print(f"[dashboard] volume reload skipped: {e}")

    threading.Thread(target=reload_volume_forever, daemon=True).start()

    env = dict(
        os.environ,
        PORT=str(PORT),
        DUMP_ROOT=DUMP_ROOT,
        PYTHON_BIN="python3",
        NODE_ENV="production",
    )
    subprocess.Popen(["/root/.bun/bin/bun", "run", "/dashboard/server.ts"], env=env)
