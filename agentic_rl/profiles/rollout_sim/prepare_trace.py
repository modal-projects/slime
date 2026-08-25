"""Build replayable episode traces from a real rollout dump.

Reads ``slime-checkpoints:/checkpoints/swe_rollout_dumps/<tag>/rollout_<k>.pt``
(the ``save_debug_rollout_data`` format: ``{"rollout_id", "samples":
[Sample.to_dict()]}``) and writes a compact trace to
``slime-data:/data/profiles/rollout_sim/traces/<name>.json.gz``.

A trace episode preserves the *shape* of the real episode, which is what the
load simulator needs:

- ``prompt``: the real prompt token ids;
- ``turns``: alternating real assistant-turn lengths (contiguous
  ``loss_mask==1`` runs) and real observation token blocks (``loss_mask==0``
  runs) from the response region;
- ``pause_s``: the episode's measured non-LLM agent time per turn
  (timing.agent - timing.generate, from metadata.agentic).

Replay appends the freshly *generated* tokens (not the recorded ones) so the
radix cache behaves like production, and forces each turn to the recorded
length with ``ignore_eos`` so the workload shape is identical across engine
configs.

Usage:

    export MODAL_ENVIRONMENT=junlin-dev
    # inspect a dump's structure first
    uv run --with modal modal run agentic_rl/profiles/rollout_sim/prepare_trace.py::peek \
        --dump swe_rollout_dumps/qwen3.6-27b-frontier-cs-retro-a-final-p50-rlag4-20260818-020000/rollout_4.pt
    # extract traces
    uv run --with modal modal run agentic_rl/profiles/rollout_sim/prepare_trace.py::extract \
        --dump swe_rollout_dumps/qwen3.6-27b-frontier-cs-retro-a-final-p50-rlag4-20260818-020000/rollout_4.pt \
        --name p50_rlag4_r4
"""

from __future__ import annotations

import gzip
import json
import os

import modal

DOCKER_IMAGE = "slimerl/slime:nightly-dev-20260529a"
CHECKPOINTS_PATH = "/checkpoints"
DATA_PATH = "/data"
TRACE_DIR = f"{DATA_PATH}/profiles/rollout_sim/traces"

image = modal.Image.from_registry(DOCKER_IMAGE).entrypoint([]).run_commands("uv pip install --system modal")
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=False)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=False)

app = modal.App("rollout-sim-prepare")

_VOLUMES = {CHECKPOINTS_PATH: checkpoints_volume, DATA_PATH: data_volume}


def _load_samples(dump_rel: str) -> list[dict]:
    import torch

    data = torch.load(f"{CHECKPOINTS_PATH}/{dump_rel}", map_location="cpu", weights_only=False)
    return data["samples"]


def _mask_runs(mask: list[int]) -> list[tuple[int, int, int]]:
    """(value, start, length) for contiguous runs."""
    runs = []
    i = 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        runs.append((mask[i], i, j - i))
        i = j
    return runs


def _episode_trace(sample: dict) -> dict | None:
    tokens = sample.get("tokens") or []
    response_length = int(sample.get("response_length") or 0)
    mask = sample.get("loss_mask")
    if not tokens or response_length <= 0 or not mask:
        return None
    prompt = tokens[: len(tokens) - response_length]
    response = tokens[len(tokens) - response_length :]
    if len(mask) == len(tokens):
        mask = mask[len(tokens) - response_length :]
    if len(mask) != len(response):
        return None

    turns: list[dict] = []
    for value, start, length in _mask_runs(mask):
        if value == 1:
            turns.append({"gen": length})
        else:
            obs = response[start : start + length]
            if turns and "obs" not in turns[-1]:
                turns[-1]["obs"] = obs
            elif turns:
                turns[-1]["obs"] = turns[-1]["obs"] + obs
            # leading mask-0 run (rare): fold into the prompt
            else:
                prompt = prompt + obs
    turns = [t for t in turns if t["gen"] > 0]
    if not turns:
        return None

    agentic = (sample.get("metadata") or {}).get("agentic") or {}
    timing = agentic.get("timing") or {}
    n_turns = max(1, int(agentic.get("turns") or len(turns)))
    pause_total = max(0.0, float(timing.get("agent") or 0.0) - float(timing.get("generate") or 0.0))
    return {
        "prompt": prompt,
        "turns": turns,
        "pause_s": round(min(30.0, pause_total / n_turns), 3),
        "meta": {
            "real_turns": agentic.get("turns"),
            "real_output_tokens": agentic.get("output_tokens"),
            "real_elapsed_sec": agentic.get("elapsed_sec"),
            "real_generate_s": timing.get("generate"),
            "exit_status": agentic.get("exit_status"),
        },
    }


@app.function(image=image, volumes=_VOLUMES, timeout=1800, memory=32768)
def peek_remote(dump_rel: str) -> None:
    checkpoints_volume.reload()
    samples = _load_samples(dump_rel)
    print(f"samples: {len(samples)}")
    s = samples[0]
    print("fields:", sorted(s.keys()))
    print(
        "tokens:",
        len(s.get("tokens") or []),
        "response_length:",
        s.get("response_length"),
        "loss_mask:",
        len(s.get("loss_mask") or []),
    )
    agentic = (s.get("metadata") or {}).get("agentic") or {}
    print("agentic keys:", sorted(agentic.keys()))
    print("timing:", agentic.get("timing"))
    trace = _episode_trace(s)
    if trace:
        print(
            "trace: prompt",
            len(trace["prompt"]),
            "turns",
            len(trace["turns"]),
            "gen",
            [t["gen"] for t in trace["turns"]][:8],
            "obs",
            [len(t.get("obs") or []) for t in trace["turns"]][:8],
            "pause_s",
            trace["pause_s"],
        )


@app.function(image=image, volumes=_VOLUMES, timeout=3600, memory=65536)
def extract_remote(dump_rel: str, name: str, max_episodes: int = 256) -> str:
    checkpoints_volume.reload()
    data_volume.reload()
    samples = _load_samples(dump_rel)
    traces = []
    skipped = 0
    for s in samples:
        t = _episode_trace(s)
        if t is None:
            skipped += 1
            continue
        traces.append(t)
        if len(traces) >= max_episodes:
            break
    if not traces:
        raise RuntimeError("no usable episodes in dump")
    gens = [sum(t["gen"] for t in tr["turns"]) for tr in traces]
    summary = {
        "source_dump": dump_rel,
        "episodes": len(traces),
        "skipped": skipped,
        "mean_turns": sum(len(tr["turns"]) for tr in traces) / len(traces),
        "mean_gen_tokens": sum(gens) / len(gens),
        "max_gen_tokens": max(gens),
        "mean_prompt_tokens": sum(len(tr["prompt"]) for tr in traces) / len(traces),
        "mean_pause_s": sum(tr["pause_s"] for tr in traces) / len(traces),
    }
    os.makedirs(TRACE_DIR, exist_ok=True)
    out = f"{TRACE_DIR}/{name}.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump({"summary": summary, "episodes": traces}, fh)
    data_volume.commit()
    print(json.dumps(summary, indent=1))
    return out


@app.local_entrypoint()
def peek(dump: str) -> None:
    peek_remote.remote(dump)


@app.local_entrypoint()
def extract(dump: str, name: str, max_episodes: int = 256) -> None:
    print(extract_remote.remote(dump, name, max_episodes=max_episodes))
