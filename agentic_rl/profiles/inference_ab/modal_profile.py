"""Inference A/B profile: SGLang engine flag-identical to the Frontier-CS retro
runs, driven by a simulated 5-step agentic rollout workload.

One Modal container = one production engine slice (2x H200, TP2) booted with the
exact ServerArgs the retro launch config produces (slime maps every ``sglang_*``
arg onto ``ServerArgs``; see slime/backends/sglang_utils/sglang_engine.py
``_compute_server_args``). The only intentional variable is
``--enable-deterministic-inference`` (arm "on" mirrors the retro runs, arm "off"
mirrors the pre-retro baselines).

Launch (both arms in parallel, detached):

    export MODAL_ENVIRONMENT=junlin-dev
    uv run --with modal modal run -d agentic_rl/profiles/inference_ab/modal_profile.py --arm on
    uv run --with modal modal run -d agentic_rl/profiles/inference_ab/modal_profile.py --arm off

Results land in the ``slime-data`` volume under ``profiles/inference_ab/`` and
are printed as JSON to the app logs.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import modal

DOCKER_IMAGE = "slimerl/slime:nightly-dev-20260529a"
HF_MODEL = "Qwen/Qwen3.6-27B"
PORT = 30000
DATA_PATH = "/data"
RESULT_DIR = f"{DATA_PATH}/profiles/inference_ab"

image = (
    modal.Image.from_registry(DOCKER_IMAGE)
    .entrypoint([])
    # The base image ships a populated HF cache; clear it or the volume mount
    # fails with "cannot mount volume on non-empty path" (same as launch_config).
    .run_commands("rm -rf /root/.cache/huggingface", "uv pip install --system modal")
    .add_local_dir(str(Path(__file__).parent), remote_path="/root/inference_ab", copy=True, ignore=["**/__pycache__"])
)

# create_if_missing=False on purpose: fail loudly if MODAL_ENVIRONMENT is wrong
# instead of minting empty throwaway volumes in `main`.
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=False)

app = modal.App("inference-ab-profile")


def _engine_cmd(model_path: str, deterministic: bool) -> list[str]:
    """The retro engine's ServerArgs as launch_server CLI flags.

    Mirrors RetroSlimeConfig + slime's _compute_server_args base kwargs
    (minus multi-node/router plumbing, which doesn't apply to one engine).
    """
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--trust-remote-code",
        "--random-seed",
        "20260802",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--tp-size",
        "2",
        "--mem-fraction-static",
        "0.85",
        "--cuda-graph-bs",
        *[str(b) for b in [1, 2, 4, 8, 16] + list(range(24, 257, 8))],
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--skip-server-warmup",
        "--enable-draft-weights-cpu-backup",
        "--enable-metrics",
    ]
    if deterministic:
        cmd.append("--enable-deterministic-inference")
    return cmd


def _wait_healthy(proc: subprocess.Popen, log_path: str, timeout_s: int = 1800) -> float:
    import requests

    t0 = time.monotonic()
    while True:
        if proc.poll() is not None:
            tail = Path(log_path).read_text(errors="replace")[-8000:]
            raise RuntimeError(f"engine died during boot (rc={proc.returncode}):\n{tail}")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health_generate", timeout=5).status_code == 200:
                return time.monotonic() - t0
        except Exception:
            pass
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError("engine did not become healthy in time")
        time.sleep(5)


@app.function(
    image=image,
    gpu="H200:2",
    volumes={"/root/.cache/huggingface": hf_cache_volume, DATA_PATH: data_volume},
    timeout=4 * 60 * 60,
)
def profile(
    arm: str,
    steps: int = 5,
    episodes: int = 16,
    concurrency: int = 8,
    turns: int = 12,
    max_new: int = 4096,
    pause: float = 4.0,
) -> dict:
    import sys

    sys.path.insert(0, "/root/inference_ab")
    import requests
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from workload import Workload

    deterministic = arm == "on"
    hf_cache_volume.reload()
    data_volume.reload()
    model_path = snapshot_download(HF_MODEL, local_files_only=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # The jsonl "prompt" is a messages list ([{"role": "user", "content": ...}],
    # see convert2slime/harbor.py); the agent renders it through the chat
    # template. Do the same, and skip outsized statements so a 12-turn episode
    # still fits the 65536 context.
    needed = steps * max(1, episodes // 8)
    prompts = []
    with open(f"{DATA_PATH}/frontier_cs/train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if len(prompts) >= needed:
                break
            if not line.strip():
                continue
            messages = json.loads(line)["prompt"]
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            if sum(len(str(m.get("content", ""))) for m in messages) > 40_000:
                continue
            prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

    # Mirror the ray runtime env the production engines inherit.
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "1"

    log_path = "/tmp/sglang_engine.log"
    with open(log_path, "wb") as log_fh:
        proc = subprocess.Popen(_engine_cmd(model_path, deterministic), stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        boot_s = _wait_healthy(proc, log_path)
        print(f"[boot] engine healthy in {boot_s:.0f}s (deterministic={deterministic})", flush=True)
        server_info = requests.get(f"http://127.0.0.1:{PORT}/get_server_info", timeout=30).json()

        wl = Workload(
            base_url=f"http://127.0.0.1:{PORT}",
            tokenizer=tokenizer,
            prompts=prompts,
            deterministic=deterministic,
            steps=steps,
            episodes=episodes,
            concurrency=concurrency,
            turns=turns,
            max_new=max_new,
            pause=pause,
        )
        import asyncio

        result = asyncio.run(wl.run())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    result["boot_s"] = boot_s
    result["arm"] = arm
    interesting = (
        "attention_backend",
        "decode_attention_backend",
        "prefill_attention_backend",
        "sampling_backend",
        "disable_radix_cache",
        "enable_deterministic_inference",
        "disable_custom_all_reduce",
        "speculative_algorithm",
        "mamba_scheduler_strategy",
        "max_total_num_tokens",
        "version",
    )
    result["server_info"] = {k: server_info.get(k) for k in interesting if k in server_info}

    stamp = os.environ.get("LAUNCH_STAMP") or f"{datetime.now():%Y%m%d-%H%M%S}"
    out_path = f"{RESULT_DIR}/{stamp}-det-{arm}.json"
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)
    data_volume.commit()

    brief = {"arm": arm, "boot_s": boot_s, "result_path": out_path, **result["summary"], "server_info": result["server_info"]}
    print(f"[done] {json.dumps(brief, default=float)}", flush=True)
    return brief


@app.local_entrypoint()
def main(
    arm: str = "on",
    steps: int = 5,
    episodes: int = 16,
    concurrency: int = 8,
    turns: int = 12,
    max_new: int = 4096,
    pause: float = 4.0,
) -> None:
    if arm not in ("on", "off"):
        raise SystemExit("--arm must be 'on' (deterministic, mirrors retro runs) or 'off'")
    brief = profile.remote(arm, steps=steps, episodes=episodes, concurrency=concurrency, turns=turns, max_new=max_new, pause=pause)
    print(json.dumps(brief, indent=1, default=float))
