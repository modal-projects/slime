"""W&B metric pruning (2026-08-27 audit) — installed from agentic_rl, zero slime edits.

Rating scheme: 3 = decision metrics (rewards pre/post filter, lane-split
outcomes, rollout/train/exec time, KV + throughput, retro-flag telemetry);
2 = diagnostics kept (bug canaries like think_closure_frac and format_errors,
straggler_ratio, spec accept); 1 = redundant tails and internals, dropped here.

Mechanism: ``install()`` wraps ``slime.utils.logging_utils.log`` (every
step-keyed metric in slime and agentic_rl funnels through it) and
``slime.utils.wandb_utils._parse_prometheus_text`` (the sgl_engine scraper).
It is invoked at import time of ``agentic_rl.obs.metrics`` — the module slime's
RolloutManager loads via ``custom_rollout_log_function_path`` before it logs
the core rollout/perf dict, so both wrappers are active in the process that
owns the W&B run. The scraper resolves ``_parse_prometheus_text`` by module
attribute each tick, so the allowlist applies even though scraping starts
before the first rollout log. Not covered (by design, they never import
agentic_rl): train-actor emissions — all rated 2+ anyway (train/*, perf/*_time).

Set SLIME_WANDB_LOG_ALL=1 to bypass both filters (full firehose for debugging).
"""

import os
from fnmatch import fnmatchcase

DROPPED_METRIC_PATTERNS = (
    # Episode-latency tails: mean/p90 (+ combined p50/max) tell the story; the
    # per-lane p50/p99/max triplicates it.
    "agentic/elapsed_sec/p99",
    "agentic/*/elapsed_sec/p50",
    "agentic/*/elapsed_sec/p99",
    "agentic/*/elapsed_sec/max",
    "agentic/exec_call_sec/p50",
    "agentic/exec_call_sec/p99",
    "agentic/exec_time_sec/p50",
    "agentic/exec_time_sec/p99",
    "agentic/*/exec_time_sec/max",
    # Redundant with exec_call_sec (same quantity, different denominator).
    "agentic/exec_time_per_call_sec/*",
    "agentic/exec_count/max",
    "agentic/exec_count/p90",
    # Counts whose frac twin is the useful signal.
    "agentic/exec_timeouts/mean",
    "agentic/think_closures/mean",
    "agentic/chains/mean",
    "agentic/output_tokens/max",
    "agentic/context_length_wasted_tokens/max",
    "agentic/turns/max",
    "agentic/submissions/max",
    # Phase-timing maxes: episode/max (the rollout critical path) is kept; the
    # rest duplicate what the p90s already show.
    "agentic/timing/boot/max",
    "agentic/timing/prep/max",
    "agentic/timing/agent/max",
    "agentic/timing/generate/max",
    "agentic/timing/verifier/max",
    "agentic/fresh/timing/*/max",
    "agentic/fresh/timing/*/p90",
    "agentic/retro/timing/*/max",
    "agentic/retro/timing/*/p90",
    # version_span is versions_in_batch + version_lag restated; age tail unused.
    "async/version_span/*",
    "async/*/version_span/*",
    "async/sample_age_sec/max",
    "async/*/sample_age_sec/max",
    # Snapshot-store internals from the all-turns bring-up; mean/p90 kept.
    "retro/snapshot_capture/fraction_error/*",
    "retro/snapshot_capture/trajectory_fraction/p*",
    "retro/snapshot_capture/trajectory_fraction/max",
    "retro/snapshot_capture/turn_index/p*",
    "retro/snapshot_capture/turn_index/max",
    "retro/snapshot_capture/latency_seconds/p50",
    "retro/snapshot_capture/latency_seconds/p99",
    "retro/snapshot_capture/latency_seconds/max",
    "retro/snapshot_restore/latency_seconds/p50",
    "retro/snapshot_restore/latency_seconds/p99",
    "retro/snapshot_restore/latency_seconds/max",
    "retro/snapshot_capture/estimated_files/mean",
    # dynamic_sampling/{completed,dropped}_groups carry the filter story.
    "rollout/dynamic_filter/*",
    "rollout/response_len/min",
    "rollout/response_len/median",
    "perf/longest_effective_sample_tokens_per_sec",
)

# The router's /engine_metrics dump mirrors sglang's full Prometheus registry
# (~80 series including histogram buckets, HTTP counters, and startup one-offs).
# Only these gauges earn a W&B panel: concurrency (running/queue/retracted/
# aborted), capacity (token/KV/mamba/SWA usage, utilization), cache hit rate,
# decode throughput, and speculative-decoding acceptance.
ENGINE_METRICS_KEEP = frozenset(
    {
        "sglang_num_running_reqs",
        "sglang_num_queue_reqs",
        "sglang_num_retracted_reqs",
        "sglang_num_aborted_requests_total",
        "sglang_token_usage",
        "sglang_full_token_usage",
        "sglang_mamba_usage",
        "sglang_swa_token_usage",
        "sglang_utilization",
        "sglang_cache_hit_rate",
        "sglang_gen_throughput",
        "sglang_spec_accept_rate",
        "sglang_spec_accept_length",
    }
)


def _log_all() -> bool:
    return os.environ.get("SLIME_WANDB_LOG_ALL") == "1"


def drop_low_value_metrics(metrics: dict, step_key: str | None = None) -> dict:
    if _log_all():
        return metrics
    return {
        k: v
        for k, v in metrics.items()
        if k == step_key or not any(fnmatchcase(k, pat) for pat in DROPPED_METRIC_PATTERNS)
    }


_installed = False


def install() -> None:
    """Idempotently wrap slime's logging choke points with the rating-1 filter."""
    global _installed
    if _installed:
        return
    _installed = True

    from slime.utils import logging_utils, wandb_utils

    inner_log = logging_utils.log

    def filtered_log(args, metrics, step_key):
        return inner_log(args, drop_low_value_metrics(metrics, step_key=step_key), step_key=step_key)

    logging_utils.log = filtered_log

    inner_parse = wandb_utils._parse_prometheus_text

    def filtered_parse(text):
        metrics = inner_parse(text)
        if _log_all():
            return metrics
        return {name: value for name, value in metrics.items() if name in ENGINE_METRICS_KEEP}

    wandb_utils._parse_prometheus_text = filtered_parse
