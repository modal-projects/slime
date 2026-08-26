"""Self-contained Modal launcher for Frontier-CS asynchronous retro replay.

Examples:

    # One-step engineering smoke (1 train node + 1 rollout node).
    RETRO_REWARD_ARM=final RETRO_TARGET_TRAJECTORY_FRACTION=0.50 \
      uv run --with modal modal run agentic_rl/retro/modal_train.py::train

    # Full 32-group, 20-update P75 Arm-A pilot.
    RETRO_REWARD_ARM=final RETRO_TARGET_TRAJECTORY_FRACTION=0.75 \
    RETRO_PHASE2_GROUPS=32 RETRO_PHASE2_ROLLOUTS=20 \
      uv run --with modal modal run -d agentic_rl/retro/modal_train.py::train

Held-out avg@3 eval (RUNBOOK §7 step 4 — no guide repo involved; use
``python -m agentic_rl.eval.frontier_cs.plan`` to print the full per-arm
commands):

    ROLLOUT_MODE=eval FRONTIER_CS_EVAL_ARM=p50 \
      uv run --with modal modal run -d agentic_rl/retro/modal_train.py::train
    ROLLOUT_MODE=eval FRONTIER_CS_EVAL_ARM=p50 FRONTIER_CS_EVAL_ID=<same id> \
      uv run --with modal modal run agentic_rl/retro/modal_train.py::post_process_data

Bootstrap hooks use the same file:

    uv run --with modal modal run agentic_rl/retro/modal_train.py::download_model
    uv run --with modal modal run agentic_rl/retro/modal_train.py::download_data
    uv run --with modal modal run agentic_rl/retro/modal_train.py::convert_hf_to_megatron_checkpoint
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import modal
import modal.experimental

from agentic_rl.retro.launch_config import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    SLIME_ROOT,
    build_launch_configs,
    materialize_yaml_configs,
)

# Locally this file sits at <repo>/agentic_rl/retro/modal_train.py and REPO_ROOT
# feeds add_local_dir. In the container Modal >= 1.0 imports the entrypoint at
# /root/modal_train.py (no package path, only two parents), where the repo copy
# lives at SLIME_ROOT — REPO_ROOT is never used there beyond this definition.
REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path(SLIME_ROOT)
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
FINAL_COMMIT_BARRIER = "retro-final-commit-barrier"
FINAL_COMMIT_NAMESPACE = "retro-launcher"
FINAL_COMMIT_TIMEOUT_SEC = 3600
PERIODIC_COMMIT_SEC = 60

modal_cfg, slime_cfg = build_launch_configs()

image = (
    modal.Image.from_registry(modal_cfg.docker_image)
    .entrypoint([])
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=SLIME_ROOT,
        copy=True,
        ignore=[
            "**/__pycache__",
            "**/*.pyc",
            "**/.git",
            "**/.venv",
            "agentic_rl/profiles/**",
        ],
    )
    .run_commands(*modal_cfg.image_run_commands)
    .env(modal_cfg.image_env)
)

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)
modal_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
}

if slime_cfg.rollout_mode == "eval":
    _default_app_name = f"frontier-cs-heldout-{slime_cfg.arm}"
else:
    _default_app_name = f"frontier-cs-retro-{slime_cfg.reward_arm}"
app_name = os.environ.get("MODAL_APP_NAME") or _default_app_name
app = modal.App(app_name)


@app.local_entrypoint()
def show_config() -> None:
    """Print the resolved run identity and ablation controls without launching."""

    print(f"mode={slime_cfg.rollout_mode}")
    print(f"run_tag={slime_cfg.run_tag}")
    print(f"state_tag={slime_cfg.state_tag}")
    print(f"nodes={slime_cfg.total_nodes()}")
    if slime_cfg.rollout_mode == "eval":
        print(f"arm={slime_cfg.arm}")
        print(f"eval_id={slime_cfg.eval_id}")
        print(f"checkpoint={slime_cfg.load} (step={slime_cfg.ckpt_step})")
        return
    print(f"reward_arm={slime_cfg.reward_arm}")
    print(f"target_fraction={slime_cfg.target_fraction}")
    print(f"groups={slime_cfg.rollout_batch_size}")
    print(f"rollouts={slime_cfg.num_rollout}")
    for key, value in sorted(slime_cfg.environment.items()):
        if key.startswith("ASYNC_RL_RETRO_"):
            print(f"{key}={value}")


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model() -> None:
    hf_cache_volume.reload()
    slime_cfg.download_model()
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_data() -> None:
    data_volume.reload()
    slime_cfg.download_data()
    data_volume.commit()


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume, str(CHECKPOINTS_PATH): checkpoints_volume},
    timeout=2 * 60 * 60,
)
def post_process_data() -> None:
    """Aggregate a finished held-out eval dump into strict avg@3 summary.json."""

    if slime_cfg.rollout_mode != "eval":
        raise ValueError("post_process_data is only meaningful with ROLLOUT_MODE=eval")
    data_volume.reload()
    checkpoints_volume.reload()
    slime_cfg.post_process_data()
    checkpoints_volume.commit()


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}",
    volumes=modal_volumes,
    timeout=4 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
def convert_hf_to_megatron_checkpoint() -> None:
    """Convert Qwen3.6-27B from HF format into Slime's torch_dist reference."""

    from huggingface_hub import snapshot_download

    hf_cache_volume.reload()
    checkpoints_volume.reload()
    hf_path = snapshot_download(slime_cfg.hf_checkpoint, local_files_only=True)
    world_size = slime_cfg.tensor_model_parallel_size * slime_cfg.pipeline_model_parallel_size
    if world_size > slime_cfg.actor_num_gpus_per_node:
        raise ValueError(
            f"conversion world size {world_size} exceeds one {slime_cfg.actor_num_gpus_per_node}-GPU node"
        )
    extra_args = []
    for attr, flag in (
        ("decoder_first_pipeline_num_layers", "decoder-first-pipeline-num-layers"),
        ("decoder_last_pipeline_num_layers", "decoder-last-pipeline-num-layers"),
        ("mtp_num_layers", "mtp-num-layers"),
        ("make_vocab_size_divisible_by", "make-vocab-size-divisible-by"),
    ):
        if value := getattr(slime_cfg, attr, None):
            extra_args.extend((f"--{flag}", str(value)))
    command = (
        f"source {SLIME_ROOT}/{slime_cfg.slime_model_script} && "
        f"torchrun --nproc-per-node={world_size} {SLIME_ROOT}/tools/convert_hf_to_torch_dist.py "
        f"${{MODEL_ARGS[@]}} {shlex.join(extra_args)} "
        f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(str(slime_cfg.ref_load))}"
    )
    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env={**os.environ, **slime_cfg.environment},
    )
    checkpoints_volume.commit()


def _resolve_checkpoint(value: str) -> str:
    if value.startswith("/"):
        return value
    from huggingface_hub import snapshot_download

    return snapshot_download(value, local_files_only=True)


def _prepare_config() -> None:
    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if value := getattr(slime_cfg, attr, None):
            setattr(slime_cfg, attr, _resolve_checkpoint(str(value)))
    materialize_yaml_configs(slime_cfg, tempfile.mkdtemp(prefix="retro-slime-"))


def _build_train_cmd() -> str:
    train_script = f"{SLIME_ROOT}/{'train_async.py' if slime_cfg.async_mode else 'train.py'}"
    inner = (
        f"source {SLIME_ROOT}/{slime_cfg.slime_model_script} && "
        f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(slime_cfg.cli_args())}"
    )
    return f"bash -c {shlex.quote(inner)}"


def _cluster_context() -> tuple[int, str, str, int]:
    expected = slime_cfg.total_nodes()
    if expected == 1:
        return 0, "127.0.0.1", "127.0.0.1", 1
    info = modal.experimental.get_cluster_info()
    actual = len(info.container_ipv4_ips)
    if actual != expected:
        raise RuntimeError(f"cluster size mismatch: expected {expected}, got {actual}")
    return (
        info.rank,
        info.container_ipv4_ips[0],
        info.container_ipv4_ips[info.rank],
        actual,
    )


def _start_ray_head(my_ip: str, expected_nodes: int) -> None:
    import ray

    ray.shutdown()
    subprocess.Popen(
        [
            "ray",
            "start",
            "--head",
            f"--node-ip-address={my_ip}",
            "--dashboard-host=0.0.0.0",
        ]
    )
    for _ in range(30):
        try:
            ray.init(address="auto")
            break
        except ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("ray head node did not start")

    timeout = int(os.environ.get("MODAL_RAY_JOIN_TIMEOUT", "180"))
    for _ in range(timeout):
        alive = [node for node in ray.nodes() if node["Alive"]]
        print(f"Waiting for workers: {len(alive)}/{expected_nodes} alive", flush=True)
        if len(alive) == expected_nodes:
            return
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for {expected_nodes} Ray nodes")


async def _periodic_volume_commit() -> None:
    """Bound the uncommitted-upload backlog behind the checkpoints mount.

    A single torch_dist save writes hundreds of GB through the FUSE mount and
    Modal's background commits lag far behind; anything still uncommitted when
    the cluster dies is lost (observed as truncated .distcp shards in the
    staleness cohort's final saves). Best-effort — a failed commit only defers
    flushing to the next tick or the final barrier.
    """

    while True:
        await asyncio.sleep(PERIODIC_COMMIT_SEC)
        try:
            await checkpoints_volume.commit.aio()
        except Exception as exc:
            print(f"Periodic checkpoints commit failed (will retry): {exc}", flush=True)


async def _cancel_task(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def _park_worker_and_commit_on_signal() -> None:
    """Park a non-rank-0 node; flush its checkpoint mount when rank 0 signals.

    Modal tears the whole cluster down the moment rank 0 returns, so worker
    containers never exit gracefully: volume writes newer than the last
    commit — the final save's shard data still uploading plus its trailing
    files (.metadata, metadata.json, latest_checkpointed_iteration.txt), which
    the Megatron save coordinator may write from any train node — would be
    lost.
    """

    import ray

    ray.shutdown()
    committer = asyncio.create_task(_periodic_volume_commit())
    connected = False
    while True:
        await asyncio.sleep(5)
        if not connected:
            try:
                ray.init(address="auto", ignore_reinit_error=True)
                connected = True
            except Exception:
                continue
        try:
            barrier = ray.get_actor(FINAL_COMMIT_BARRIER, namespace=FINAL_COMMIT_NAMESPACE)
        except ValueError:
            continue
        except Exception:
            # Stale client from a retried attempt against a restarted cluster.
            ray.shutdown()
            connected = False
            continue
        await _cancel_task(committer)
        ok = True
        try:
            await checkpoints_volume.commit.aio()
        except Exception as exc:
            ok = False
            print(f"Checkpoints volume commit failed on worker: {exc}", flush=True)
        ray.get(barrier.ack.remote(ok))
        break
    while True:
        await asyncio.sleep(10)


async def _commit_checkpoints_cluster_wide(nodes: int) -> None:
    """Commit every node's checkpoint mount before rank 0 returns (= teardown)."""

    import ray

    @ray.remote(num_cpus=0)
    class FinalCommitBarrier:
        def __init__(self) -> None:
            self.acks = 0
            self.failures = 0

        def ack(self, ok: bool) -> None:
            self.acks += 1
            if not ok:
                self.failures += 1

        def counts(self) -> tuple[int, int]:
            return self.acks, self.failures

    barrier = FinalCommitBarrier.options(
        name=FINAL_COMMIT_BARRIER,
        namespace=FINAL_COMMIT_NAMESPACE,
        lifetime="detached",
    ).remote()
    await checkpoints_volume.commit.aio()
    expected = nodes - 1
    acks = failures = 0
    deadline = time.time() + FINAL_COMMIT_TIMEOUT_SEC
    while acks < expected and time.time() < deadline:
        await asyncio.sleep(2)
        acks, failures = ray.get(barrier.counts.remote())
    if acks < expected or failures:
        print(
            f"WARNING: checkpoint commit barrier incomplete: {acks}/{expected} worker acks, "
            f"{failures} failed commits — the last save may be missing files",
            flush=True,
        )
    else:
        print(f"Checkpoints volume committed on all {nodes} nodes", flush=True)


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory,
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    ephemeral_disk=modal_cfg.ephemeral_disk,
    volumes=modal_volumes,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=60.0),
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(slime_cfg.total_nodes(), rdma=True)
async def train() -> None:
    """Launch the resolved retro experiment on a clustered Modal Ray job."""

    from ray.job_submission import JobSubmissionClient

    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    os.environ.setdefault("RAY_health_check_initial_delay_ms", "60000")
    os.environ.setdefault("RAY_health_check_period_ms", "15000")
    os.environ.setdefault("RAY_health_check_timeout_ms", "120000")
    os.environ.setdefault("RAY_health_check_failure_threshold", "20")

    await asyncio.gather(
        hf_cache_volume.reload.aio(),
        data_volume.reload.aio(),
        checkpoints_volume.reload.aio(),
    )
    rank, master_addr, my_ip, nodes = _cluster_context()
    os.environ["SLIME_HOST_IP"] = my_ip
    os.environ["SGLANG_HOST_IP"] = my_ip
    os.environ["HOST_IP"] = my_ip

    if rank != 0:
        subprocess.Popen(
            [
                "ray",
                "start",
                f"--node-ip-address={my_ip}",
                "--address",
                f"{master_addr}:{RAY_PORT}",
            ]
        )
        await _park_worker_and_commit_on_signal()

    _start_ray_head(my_ip, nodes)
    _prepare_config()
    command = _build_train_cmd()
    runtime_env = {
        "env_vars": {
            "no_proxy": f"127.0.0.1,{master_addr}",
            "MASTER_ADDR": master_addr,
            **slime_cfg.environment,
        }
    }
    if wandb_key := os.environ.get("WANDB_API_KEY", ""):
        runtime_env["env_vars"]["WANDB_API_KEY"] = wandb_key

    client = JobSubmissionClient(f"http://127.0.0.1:{RAY_DASHBOARD_PORT}")
    job_id = client.submit_job(entrypoint=command, runtime_env=runtime_env)
    if slime_cfg.rollout_mode == "eval":
        detail = f"(held-out avg@3, arm={slime_cfg.arm}, ckpt_step={slime_cfg.ckpt_step})"
    else:
        detail = f"(arm={slime_cfg.reward_arm}, target={slime_cfg.target_fraction})"
    print(f"Running {slime_cfg.run_tag} on {nodes} {modal_cfg.gpu} nodes {detail}", flush=True)
    display_env = {"env_vars": dict(runtime_env["env_vars"])}
    if "WANDB_API_KEY" in display_env["env_vars"]:
        display_env["env_vars"]["WANDB_API_KEY"] = "<redacted>"
    print(f"Command: {command}, runtime_env: {display_env}", flush=True)

    committer = asyncio.create_task(_periodic_volume_commit())
    try:
        async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
            print(f"Ray dashboard: {tunnel.url}", flush=True)
            async for line in client.tail_job_logs(job_id):
                print(line, end="", flush=True)
    finally:
        await _cancel_task(committer)

    status = client.get_job_status(job_id)
    print(f"Ray job {job_id} finished with status: {status}", flush=True)
    # Flush even on failure so interval saves survive; must finish before this
    # function returns, because returning tears down the worker containers.
    await _commit_checkpoints_cluster_wide(nodes)
    if str(status) != "SUCCEEDED":
        raise RuntimeError(f"Ray job {job_id} ended with status {status}")
