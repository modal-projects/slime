"""Production-scale rollout inference simulator (one rollout node).

Replays real Frontier-CS episode traces (built by ``prepare_trace.py``) against
a locally-booted fleet of SGLang engines that mirrors one production rollout
node: 8x H200 = 4 engines x TP2, with a sticky-routed in-flight pool of
episodes. Production runs 4 such nodes with a 256-episode batch (queue up to
the staleness window), i.e. ~64 in-flight episodes per node — the default
here.

Replay semantics (shape-faithful and cache-faithful):

- each turn POSTs ``/generate`` with the episode's full ``input_ids`` and
  ``max_new_tokens = <real turn length>`` + ``ignore_eos`` so every config
  generates exactly the same token counts;
- the freshly generated tokens are appended (radix cache behaves like
  production), then the real recorded observation block, then a "sandbox
  exec" pause using the episode's measured per-turn non-LLM time.

Usage (baseline, deterministic off):

    export MODAL_ENVIRONMENT=junlin-dev
    uv run --with modal modal run -d agentic_rl/profiles/rollout_sim/modal_sim.py \
        --trace p50_rlag4_r4 --label baseline

Knobs for optimization passes: ``--engines/--tp/--inflight/--episodes/
--pause-scale/--deterministic/--mem-fraction/--spec-steps/--spec-topk/
--spec-draft (0 disables EAGLE)/--attention-backend/--server-extra`` (raw
flags appended to launch_server). Results land in
``slime-data:/data/profiles/rollout_sim/results/`` and print to the app log.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import random
import subprocess
import time
from datetime import datetime

import modal

# Standardized on upstream SGLang (newest stable) as of 2026-08-22. The harness
# needs no slime/megatron code -- it only boots engines and drives them over HTTP
# -- so the official image is both cleaner and lets us track SGLang releases.
# Override with ROLLOUT_SIM_IMAGE to reproduce older results on the slime image
# (slimerl/slime:nightly-dev-20260529a ships sglang 0.5.12.post1).
DOCKER_IMAGE = os.environ.get("ROLLOUT_SIM_IMAGE", "lmsysorg/sglang:v0.5.18-cu129")
HF_MODEL = "Qwen/Qwen3.6-27B"
DATA_PATH = "/data"
TRACE_DIR = f"{DATA_PATH}/profiles/rollout_sim/traces"
RESULT_DIR = f"{DATA_PATH}/profiles/rollout_sim/results"
BASE_PORT = 30000
CTX_LIMIT = 65536

image = (
    modal.Image.from_registry(DOCKER_IMAGE)
    .entrypoint([])
    # Clear any baked HF cache or the volume mount fails on a non-empty path.
    .run_commands(
        "rm -rf /root/.cache/huggingface",
        "pip install --no-cache-dir modal httpx requests huggingface_hub",
    )
    # Bake the resolved image name so the remote DOCKER_IMAGE (re-read from env
    # in the container) reports the image that actually ran, not the default.
    .env({"ROLLOUT_SIM_IMAGE": DOCKER_IMAGE})
)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=False)

app = modal.App("rollout-sim")


def _engine_cmd(
    model_path: str,
    *,
    port: int,
    base_gpu_id: int,
    tp: int,
    deterministic: bool,
    mem_fraction: float,
    spec_steps: int,
    spec_topk: int,
    spec_draft: int,
    spec_algo: str,
    spec_draft_model: str,
    dflash_block_size: int,
    attention_backend: str,
    server_extra: str,
    mamba_strategy: str,
) -> list[str]:
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--trust-remote-code",
        "--random-seed",
        str(20260802 + base_gpu_id),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--base-gpu-id",
        str(base_gpu_id),
        "--tp-size",
        str(tp),
        "--mem-fraction-static",
        str(mem_fraction),
        "--cuda-graph-bs",
        *[str(b) for b in [1, 2, 4, 8, 16] + list(range(24, 257, 8))],
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--skip-server-warmup",
        "--enable-draft-weights-cpu-backup",
        "--enable-metrics",
    ]
    if mamba_strategy:
        cmd += ["--mamba-scheduler-strategy", mamba_strategy]
    algo = (spec_algo or "").upper()
    if algo == "EAGLE" and spec_draft > 0:
        cmd += [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            str(spec_steps),
            "--speculative-eagle-topk",
            str(spec_topk),
            "--speculative-num-draft-tokens",
            str(spec_draft),
        ]
    elif algo == "DFLASH":
        # DFLASH needs its own trained draft checkpoint; block size is DFLASH's
        # alias for the verify-window length (--speculative-num-draft-tokens).
        if not spec_draft_model:
            raise ValueError("DFLASH requires --spec-draft-model (e.g. z-lab/Qwen3.6-27B-DFlash)")
        cmd += [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            spec_draft_model,
            "--speculative-dflash-block-size",
            str(dflash_block_size),
        ]
    if attention_backend:
        cmd += ["--attention-backend", attention_backend]
    if deterministic:
        cmd.append("--enable-deterministic-inference")
    if server_extra:
        cmd += server_extra.split()
    return cmd


def _wait_healthy(procs: dict[int, subprocess.Popen], ports: list[int], timeout_s: int = 2400) -> float:
    import requests

    t0 = time.monotonic()
    pending = set(ports)
    while pending:
        for port in list(pending):
            proc = procs[port]
            if proc.poll() is not None:
                tail = open(f"/tmp/engine_{port}.log", errors="replace").read()[-6000:]
                raise RuntimeError(f"engine :{port} died during boot (rc={proc.returncode}):\n{tail}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/health_generate", timeout=5).status_code == 200:
                    pending.discard(port)
            except Exception:
                pass
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(f"engines not healthy in time; pending: {sorted(pending)}")
        if pending:
            time.sleep(5)
    return time.monotonic() - t0


def _pctl(xs, q):
    if not xs:
        return 0.0
    ys = sorted(xs)
    return float(ys[min(len(ys) - 1, int(q * len(ys)))])


def _summary(xs):
    return (
        {"n": len(xs), "mean": sum(xs) / len(xs), "p50": _pctl(xs, 0.5), "p90": _pctl(xs, 0.9), "max": max(xs)}
        if xs
        else {}
    )


async def _episode(
    client, sem, url: str, trace: dict, ep_idx: int, pause_scale: float, bucket_tokens: dict, live: dict
):
    """Fault-isolated: any failure ends THIS episode (error recorded) — it must
    never propagate, or the shared client closes and every in-flight episode
    dies with 'client has been closed'.

    ``live`` carries the occupancy counters the sampler reads: ``eps`` (episodes
    holding the in-flight slot) and ``reqs`` (episodes actually awaiting
    /generate — the rest are in a tool pause).
    """
    import httpx

    rng = random.Random(11701 + ep_idx)
    ids = list(trace["prompt"])
    pause = float(trace["pause_s"]) * pause_scale
    turn_recs = []
    error = None
    async with sem:
        live["eps"] += 1
        t_start = time.perf_counter()
        for turn in trace["turns"]:
            gen = min(int(turn["gen"]), CTX_LIMIT - 1024 - len(ids))
            if gen <= 0:
                break
            sp = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": gen,
                "ignore_eos": True,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
            }
            t0 = time.perf_counter()
            live["reqs"] += 1
            try:
                resp = await client.post(url, json={"input_ids": ids, "sampling_params": sp, "return_logprob": True})
                latency = time.perf_counter() - t0
                resp.raise_for_status()
                meta = resp.json().get("meta_info") or {}
            except httpx.HTTPStatusError as exc:
                error = f"turn {len(turn_recs)}: HTTP {exc.response.status_code}: {exc.response.text[:300]}"
                break
            except Exception as exc:  # noqa: BLE001 - connection/protocol/JSON failures
                error = f"turn {len(turn_recs)}: {type(exc).__name__}: {exc}"
                break
            finally:
                live["reqs"] -= 1
            lps = meta.get("output_token_logprobs") or []
            out_ids = [t[1] for t in lps]
            turn_recs.append(
                {
                    "latency": latency,
                    "completion_tokens": meta.get("completion_tokens", len(out_ids)),
                    "prompt_tokens": meta.get("prompt_tokens", len(ids)),
                    "cached_tokens": meta.get("cached_tokens", 0),
                    "spec_verify_ct": meta.get("spec_verify_ct", 0),
                }
            )
            bucket = int(time.monotonic() // 60)
            bucket_tokens[bucket] = bucket_tokens.get(bucket, 0) + turn_recs[-1]["completion_tokens"]
            ids += out_ids
            obs = turn.get("obs") or []
            if len(ids) + len(obs) >= CTX_LIMIT - 1024:
                break
            ids += obs
            if pause > 0:
                await asyncio.sleep(pause * rng.uniform(0.7, 1.3))
        elapsed = time.perf_counter() - t_start
        live["eps"] -= 1
    gen_time = sum(r["latency"] for r in turn_recs)
    out_tok = sum(r["completion_tokens"] for r in turn_recs)
    return {
        "error": error,
        "episode": ep_idx,
        "turns": len(turn_recs),
        "elapsed": elapsed,
        "gen_time": gen_time,
        "output_tokens": out_tok,
        "final_ctx": len(ids),
        "decode_tok_per_s": out_tok / gen_time if gen_time > 0 else 0.0,
        "cached_tokens": sum(r["cached_tokens"] for r in turn_recs),
        "prompt_tokens": sum(r["prompt_tokens"] for r in turn_recs),
        "spec_verify_ct": sum(r["spec_verify_ct"] for r in turn_recs),
        "turn_latencies": [round(r["latency"], 3) for r in turn_recs],
    }


def _latency_means(engine_stats: list[dict]) -> dict:
    """Mean TTFT / inter-token / e2e latency from Prometheus histogram deltas.

    Each engine exports monotonically increasing ``_sum``/``_count`` pairs;
    (last-first) of each gives the run's mean without needing per-request
    timestamps, which non-streaming /generate cannot provide.
    """
    out: dict = {}
    pairs = (
        ("ttft_s", "time_to_first_token_seconds"),
        ("inter_token_s", "inter_token_latency_seconds"),
        ("e2e_request_s", "e2e_request_latency_seconds"),
    )
    for label, metric in pairs:
        totals_sum = 0.0
        totals_count = 0.0
        for key, acc in ((f"{metric}_sum", "sum"), (f"{metric}_count", "count")):
            first: dict = {}
            last: dict = {}
            for row in engine_stats:
                for port, value in (row.get(key) or {}).items():
                    first.setdefault(port, value)
                    last[port] = value
            delta = sum(last[p] - first.get(p, 0.0) for p in last)
            if acc == "sum":
                totals_sum = delta
            else:
                totals_count = delta
        if totals_count > 0:
            out[f"engine_{label}_mean"] = totals_sum / totals_count
            out[f"engine_{label}_samples"] = totals_count
    return out


def _metrics_line(result: dict) -> str:
    """The standard probe report line. Keep this set stable across probes so
    configs are comparable at a glance."""
    s = result["summary"]
    c = result["config"]
    kv = s.get("engine_token_usage") or {}
    spec = c.get("spec_algo") or "none"
    if (c.get("spec_algo") or "").upper() == "DFLASH":
        spec += f"(block={c.get('dflash_block_size')})"
    elif (c.get("spec_algo") or "").upper() == "EAGLE":
        spec += f"({'/'.join(str(x) for x in c.get('spec') or [])})"
    return (
        f"[metrics] {c.get('label')} | {c.get('engines')}x TP{c.get('tp')} inflight={c.get('inflight')} "
        f"spec={spec} | "
        f"gpu_tok/s={s.get('gpu_tok_per_s_saturated', 0):.0f} "
        f"decode/stream={(s.get('decode_tok_per_s') or {}).get('mean', 0):.1f} "
        f"ttft={s.get('engine_ttft_s_mean', float('nan')):.2f}s "
        f"kv_p50/p90/max={kv.get('p50', 0):.2f}/{kv.get('p90', 0):.2f}/{kv.get('max', 0):.2f} "
        f"kv_pool={(result.get('server_info') or {}).get('max_total_num_tokens')} "
        f"cache_hit={s.get('prefix_cache_hit_rate', 0):.3f} "
        f"accept={s.get('spec_accept_length', 0):.2f} "
        f"| eps={c.get('episodes_completed')} err={s.get('episode_errors')}"
    )


def _build_result(
    res: dict,
    *,
    cfg: dict,
    boot_s: float,
    server_info: dict,
    engines: int,
    tp: int,
    inflight: int,
) -> dict:
    """Assemble the result document from raw run state.

    Shared by the final write and the periodic partial dumps, so a run that is
    killed mid-flight still yields metrics computed exactly the same way.
    """
    all_eps_recs = res["episode_recs"]
    errors = [e for e in all_eps_recs if e.get("error")]
    eps = [e for e in all_eps_recs if not e.get("error")]  # clean episodes drive the metrics
    out_tok = sum(e["output_tokens"] for e in eps)
    verify = sum(e["spec_verify_ct"] for e in eps)
    prompt_tok = sum(e["prompt_tokens"] for e in eps)
    buckets = sorted(res["bucket_tokens"].items())
    timeline = [v for _, v in buckets]

    # Saturated-window throughput. The pool is closed (a fixed episode list), so
    # the run ends with a long straggler drain during which the pool is far below
    # its cap -- real training refills it, so averaging the drain in understates
    # sustained capability. Keep only the minutes where measured occupancy was
    # >= 90% of the in-flight cap; fall back to >=50%-of-peak-throughput minutes
    # when occupancy sampling is unavailable.
    occ = res.get("occupancy") or []
    minute_occ: dict[int, list[int]] = {}
    for t, eps_live, _reqs in occ:
        minute_occ.setdefault(int(t // 60), []).append(eps_live)
    saturated_minutes = {
        m for m, vals in minute_occ.items() if vals and (sum(vals) / len(vals)) >= 0.9 * inflight
    }
    if saturated_minutes:
        saturated = [v for i, (_, v) in enumerate(buckets) if i in saturated_minutes]
    else:
        peak = max(timeline) if timeline else 0
        saturated = [v for v in timeline if v >= 0.5 * peak]
    saturated = saturated or timeline
    result = {
        "config": {**cfg, "episodes_completed": len(eps)},
        "summary": {
            "boot_s": boot_s,
            "wall_s": res["wall_s"],
            "output_tokens": out_tok,
            # Headline: sustained throughput while the pool was full.
            "fleet_tok_per_s_saturated": (sorted(saturated)[len(saturated) // 2] / 60) if saturated else 0.0,
            "gpu_tok_per_s_saturated": (
                (sorted(saturated)[len(saturated) // 2] / 60 / (engines * tp)) if saturated else 0.0
            ),
            "saturated_minutes": len(saturated),
            "total_minutes": len(timeline),
            # Whole-run averages (include ramp-up + straggler drain; closed-pool artifact).
            "fleet_tok_per_s_run_avg": out_tok / res["wall_s"],
            "gpu_tok_per_s_run_avg": out_tok / res["wall_s"] / (engines * tp),
            "inflight_episodes": _summary([e for _, e, _ in occ]),
            "inflight_requests": _summary([r for _, _, r in occ]),
            "engine_running_reqs": _summary(
                [v for r in (res.get("engine_stats") or []) for v in (r.get("num_running_reqs") or {}).values()]
            ),
            "engine_queued_reqs": _summary(
                [v for r in (res.get("engine_stats") or []) for v in (r.get("num_queue_reqs") or {}).values()]
            ),
            # Server-side latency means, from histogram sum/count deltas over the
            # run (first -> last sample), averaged across engines. TTFT here is
            # prefill + queue wait; with a ~98% prefix-cache hit it is dominated
            # by queueing, so it is the metric that exposes over-subscription.
            **_latency_means(res.get("engine_stats") or []),
            "engine_token_usage": _summary(
                [v for r in (res.get("engine_stats") or []) for v in (r.get("token_usage") or {}).values()]
            ),
            "inflight_requests_per_engine": (
                (sum(r for _, _, r in occ) / len(occ) / engines) if occ else 0.0
            ),
            "decode_tok_per_s": _summary([e["decode_tok_per_s"] for e in eps]),
            "episode_elapsed": _summary([e["elapsed"] for e in eps]),
            "turns_total": sum(e["turns"] for e in eps),
            "spec_accept_length": out_tok / verify if verify else 0.0,
            "prefix_cache_hit_rate": sum(e["cached_tokens"] for e in eps) / prompt_tok if prompt_tok else 0.0,
            "episode_errors": len(errors),
            "first_errors": [e["error"] for e in errors[:5]],
        },
        "server_info": {
            k: server_info.get(k)
            for k in (
                "attention_backend",
                "sampling_backend",
                "disable_radix_cache",
                "enable_deterministic_inference",
                "disable_custom_all_reduce",
                "speculative_algorithm",
                "speculative_num_steps",
                "speculative_num_draft_tokens",
                "mem_fraction_static",
                "max_total_num_tokens",
                "version",
            )
            if k in server_info
        },
        "tok_per_min_timeline": timeline,
        "occupancy_samples": res.get("occupancy") or [],
        "engine_stats": res.get("engine_stats") or [],
        "episode_recs": all_eps_recs,
    }
    return result


@app.function(
    image=image,
    gpu="H200:8",
    volumes={"/root/.cache/huggingface": hf_cache_volume, DATA_PATH: data_volume},
    timeout=5 * 60 * 60,
    memory=(262144, 2097152),
)
def simulate(
    trace: str = "p50_rlag4_r4",
    label: str = "baseline",
    engines: int = 4,
    tp: int = 2,
    inflight: int = 64,
    episodes: int = 0,
    pause_scale: float = 1.0,
    deterministic: bool = False,
    mem_fraction: float = 0.85,
    spec_steps: int = 3,
    spec_topk: int = 1,
    spec_draft: int = 4,
    spec_algo: str = "EAGLE",
    spec_draft_model: str = "",
    dflash_block_size: int = 8,
    mamba_strategy: str = "extra_buffer",
    boot_only: bool = False,
    attention_backend: str = "",
    server_extra: str = "",
    engine_stagger: float = 45.0,
) -> dict:
    import httpx
    import requests
    from huggingface_hub import snapshot_download

    assert engines * tp <= 8, "engines*tp must fit one 8-GPU node"
    hf_cache_volume.reload()
    data_volume.reload()
    model_path = snapshot_download(HF_MODEL, local_files_only=True)

    with gzip.open(f"{TRACE_DIR}/{trace}.json.gz", "rt", encoding="utf-8") as fh:
        trace_data = json.load(fh)
    all_eps = trace_data["episodes"]
    n_eps = episodes or len(all_eps)
    rng = random.Random(20260820)
    pool = [all_eps[i % len(all_eps)] for i in range(n_eps)]
    rng.shuffle(pool)

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "1"

    cfg = {
        "trace": trace,
        "label": label,
        "engines": engines,
        "tp": tp,
        "inflight": inflight,
        "episodes_requested": n_eps,
        "pause_scale": pause_scale,
        "deterministic": deterministic,
        "mem_fraction": mem_fraction,
        "spec": [spec_steps, spec_topk, spec_draft],
        "spec_algo": spec_algo,
        "spec_draft_model": spec_draft_model or None,
        "dflash_block_size": dflash_block_size if (spec_algo or "").upper() == "DFLASH" else None,
        "mamba_strategy": mamba_strategy or None,
        "image": DOCKER_IMAGE,
        "attention_backend": attention_backend or None,
        "server_extra": server_extra or None,
        "engine_stagger": engine_stagger,
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    partial_path = f"{RESULT_DIR}/partial-{label}.json"

    ports = [BASE_PORT + i for i in range(engines)]
    procs: dict[int, subprocess.Popen] = {}
    for i, port in enumerate(ports):
        # Stagger startup: N engines x TP loading a 54GB bf16 checkpoint at once
        # spikes host RAM (page cache + per-rank RSS) and gets a rank SIGKILLed
        # (rc=-9, empty log because stdout never flushed). Costs boot time only.
        if i and engine_stagger:
            time.sleep(engine_stagger)
        cmd = _engine_cmd(
            model_path,
            port=port,
            base_gpu_id=i * tp,
            tp=tp,
            deterministic=deterministic,
            mem_fraction=mem_fraction,
            spec_steps=spec_steps,
            spec_topk=spec_topk,
            spec_draft=spec_draft,
            spec_algo=spec_algo,
            spec_draft_model=spec_draft_model,
            dflash_block_size=dflash_block_size,
            mamba_strategy=mamba_strategy,
            attention_backend=attention_backend,
            server_extra=server_extra,
        )
        log_fh = open(f"/tmp/engine_{port}.log", "wb")
        procs[port] = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

    try:
        boot_s = _wait_healthy(procs, ports)
        print(f"[boot] {engines} engines healthy in {boot_s:.0f}s", flush=True)
        server_info = requests.get(f"http://127.0.0.1:{ports[0]}/get_server_info", timeout=30).json()
        if boot_only:
            keys = (
                "attention_backend", "sampling_backend", "speculative_algorithm",
                "speculative_num_draft_tokens", "speculative_dflash_block_size",
                "disable_radix_cache", "max_total_num_tokens", "tp_size",
                "mem_fraction_static", "version",
            )
            info = {k: server_info.get(k) for k in keys if k in server_info}
            print(f"[boot-only] image={DOCKER_IMAGE} boot_s={boot_s:.0f} {json.dumps(info, default=str)}", flush=True)
            return {"boot_only": True, "boot_s": boot_s, "server_info": info, "image": DOCKER_IMAGE}

        async def run() -> dict:
            sem = asyncio.Semaphore(inflight)
            bucket_tokens: dict[int, int] = {}
            live = {"eps": 0, "reqs": 0}
            occupancy: list[tuple[float, int, int]] = []
            engine_stats: list[dict] = []
            stop = asyncio.Event()

            async def sampler(t_zero: float) -> None:
                while not stop.is_set():
                    occupancy.append((time.perf_counter() - t_zero, live["eps"], live["reqs"]))
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=5.0)
                    except TimeoutError:
                        pass

            async def engine_sampler(t_zero: float) -> None:
                """Engine-side truth: running/queued requests and KV occupancy.

                Client-side counters can't distinguish 'decoding' from 'queued
                behind a full KV pool', which is exactly the failure mode when
                concurrency is pushed past the pool's capacity.
                """
                import re as _re

                # Gauges plus histogram sum/count pairs. SGLang exports TTFT and
                # inter-token latency only as histograms, and the client can't see
                # TTFT at all on non-streaming /generate, so this is the only
                # source for it. Deltas of sum/count give windowed means.
                want = (
                    "num_running_reqs",
                    "num_queue_reqs",
                    "token_usage",
                    "cache_hit_rate",
                    "time_to_first_token_seconds_sum",
                    "time_to_first_token_seconds_count",
                    "inter_token_latency_seconds_sum",
                    "inter_token_latency_seconds_count",
                    "e2e_request_latency_seconds_sum",
                    "e2e_request_latency_seconds_count",
                )
                while not stop.is_set():
                    row: dict = {"t": time.perf_counter() - t_zero}
                    for port in ports:
                        try:
                            text = (await client.get(f"http://127.0.0.1:{port}/metrics", timeout=10.0)).text
                        except Exception:  # noqa: BLE001 - telemetry must never break the run
                            continue
                        for key in want:
                            m = _re.search(rf"^sglang:{key}\{{[^}}]*\}}\s+([0-9.eE+-]+)$", text, _re.M)
                            if m:
                                row.setdefault(key, {})[port] = float(m.group(1))
                    engine_stats.append(row)
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=15.0)
                    except TimeoutError:
                        pass

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(None), limits=httpx.Limits(max_connections=inflight * 2 + 16)
            ) as client:
                tasks = []
                for ep_idx, ep in enumerate(pool):
                    # Round-robin, sticky per episode. NOT hash(): str hashing is
                    # PYTHONHASHSEED-randomized, so engine assignment (and thus
                    # load balance) would differ run to run and break A/B compares.
                    port = ports[ep_idx % engines]
                    url = f"http://127.0.0.1:{port}/generate"
                    tasks.append(_episode(client, sem, url, ep, ep_idx, pause_scale, bucket_tokens, live))
                t0 = time.perf_counter()
                last_partial = [0.0]
                sampler_task = asyncio.create_task(sampler(t0))
                engine_task = asyncio.create_task(engine_sampler(t0))
                done: list[dict] = []
                n_err = 0
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    rec = await coro
                    done.append(rec)
                    if rec.get("error"):
                        n_err += 1
                        if n_err <= 5:
                            print(f"[episode-error] ep {rec['episode']}: {rec['error']}", flush=True)
                    if (i + 1) % 16 == 0 or i + 1 == len(tasks):
                        el = time.perf_counter() - t0
                        tok = sum(r["output_tokens"] for r in done)
                        print(
                            f"[progress] {i + 1}/{len(tasks)} eps, {el:.0f}s, fleet {tok / el:.0f} tok/s",
                            flush=True,
                        )
                        # Flush a partial result every ~5 min. These runs are
                        # ~1 GPU-hour; a mid-flight kill (client disconnect,
                        # preemption) must not throw all of it away.
                        if el - last_partial[0] >= 300:
                            last_partial[0] = el
                            try:
                                snapshot = _build_result(
                                    {
                                        "episode_recs": list(done),
                                        "wall_s": el,
                                        "bucket_tokens": dict(bucket_tokens),
                                        "occupancy": list(occupancy),
                                        "engine_stats": list(engine_stats),
                                    },
                                    cfg={**cfg, "partial": True},
                                    boot_s=boot_s,
                                    server_info=server_info,
                                    engines=engines,
                                    tp=tp,
                                    inflight=inflight,
                                )
                                with open(partial_path, "w", encoding="utf-8") as fh:
                                    json.dump(snapshot, fh, default=float)
                                data_volume.commit()
                                print(f"[partial] wrote {partial_path} at {i + 1} eps", flush=True)
                                print(_metrics_line(snapshot), flush=True)
                            except Exception as exc:  # noqa: BLE001 - never break the run
                                print(f"[partial] failed: {type(exc).__name__}: {exc}", flush=True)
                wall = time.perf_counter() - t0
                stop.set()
                await asyncio.gather(sampler_task, engine_task)
            return {
                "episode_recs": done,
                "wall_s": wall,
                "bucket_tokens": bucket_tokens,
                "occupancy": occupancy,
                "engine_stats": engine_stats,
            }

        res = asyncio.run(run())
    finally:
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    result = _build_result(
        res,
        cfg=cfg,
        boot_s=boot_s,
        server_info=server_info,
        engines=engines,
        tp=tp,
        inflight=inflight,
    )

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = f"{RESULT_DIR}/{stamp}-{label}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)
    data_volume.commit()
    print(_metrics_line(result), flush=True)
    brief = {"result_path": out_path, **result["summary"], "server_info": result["server_info"]}
    print(f"[done] {json.dumps(brief, default=float)}", flush=True)
    return brief


@app.local_entrypoint()
def main(
    trace: str = "p50_rlag4_r4",
    label: str = "baseline",
    engines: int = 4,
    tp: int = 2,
    inflight: int = 64,
    episodes: int = 0,
    pause_scale: float = 1.0,
    deterministic: bool = False,
    mem_fraction: float = 0.85,
    spec_steps: int = 3,
    spec_topk: int = 1,
    spec_draft: int = 4,
    spec_algo: str = "EAGLE",
    spec_draft_model: str = "",
    dflash_block_size: int = 8,
    mamba_strategy: str = "extra_buffer",
    boot_only: bool = False,
    attention_backend: str = "",
    server_extra: str = "",
    engine_stagger: float = 45.0,
) -> None:
    brief = simulate.remote(
        trace=trace,
        label=label,
        engines=engines,
        tp=tp,
        inflight=inflight,
        episodes=episodes,
        pause_scale=pause_scale,
        deterministic=deterministic,
        mem_fraction=mem_fraction,
        spec_steps=spec_steps,
        spec_topk=spec_topk,
        spec_draft=spec_draft,
        spec_algo=spec_algo,
        spec_draft_model=spec_draft_model,
        dflash_block_size=dflash_block_size,
        mamba_strategy=mamba_strategy,
        boot_only=boot_only,
        attention_backend=attention_backend,
        server_extra=server_extra,
        engine_stagger=engine_stagger,
    )
    print(json.dumps(brief, indent=1, default=float))


@app.local_entrypoint()
def spawn(
    trace: str = "p50_rlag4_r4",
    label: str = "baseline",
    engines: int = 4,
    tp: int = 2,
    inflight: int = 64,
    episodes: int = 0,
    pause_scale: float = 1.0,
    deterministic: bool = False,
    mem_fraction: float = 0.85,
    spec_steps: int = 3,
    spec_topk: int = 1,
    spec_draft: int = 4,
    spec_algo: str = "EAGLE",
    spec_draft_model: str = "",
    dflash_block_size: int = 8,
    mamba_strategy: str = "extra_buffer",
    boot_only: bool = False,
    attention_backend: str = "",
    server_extra: str = "",
    engine_stagger: float = 45.0,
) -> None:
    """Fire-and-forget launch. Prefer this over ``modal run -d ...::main``:
    ``main`` blocks on ``.remote()``, so a local client crash (network blip,
    laptop sleep) cancels the running function. ``.spawn()`` returns a handle
    immediately and the run survives the client exiting.

    Poll with ``modal app logs <app-id>``; partial results are flushed to
    ``profiles/rollout_sim/results/partial-<label>.json`` every ~5 minutes.
    """
    handle = simulate.spawn(
        trace=trace,
        label=label,
        engines=engines,
        tp=tp,
        inflight=inflight,
        episodes=episodes,
        pause_scale=pause_scale,
        deterministic=deterministic,
        mem_fraction=mem_fraction,
        spec_steps=spec_steps,
        spec_topk=spec_topk,
        spec_draft=spec_draft,
        spec_algo=spec_algo,
        spec_draft_model=spec_draft_model,
        dflash_block_size=dflash_block_size,
        mamba_strategy=mamba_strategy,
        boot_only=boot_only,
        attention_backend=attention_backend,
        server_extra=server_extra,
        engine_stagger=engine_stagger,
    )
    print(f"spawned label={label} function_call_id={handle.object_id}")


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_draft(repo: str = "z-lab/Qwen3.6-27B-DFlash") -> str:
    """Fetch a speculative draft checkpoint (e.g. DFlash) into the HF cache volume."""
    from huggingface_hub import snapshot_download

    hf_cache_volume.reload()
    path = snapshot_download(repo)
    hf_cache_volume.commit()
    print(f"[draft] {repo} -> {path}", flush=True)
    return path


@app.local_entrypoint()
def fetch_draft(repo: str = "z-lab/Qwen3.6-27B-DFlash") -> None:
    print(download_draft.remote(repo))
