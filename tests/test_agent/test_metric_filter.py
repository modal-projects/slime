"""Rating-1 metric cut (2026-08-27 W&B audit), implemented agentic_rl-side:
agentic_rl.obs.metric_filter drop patterns + engine-scrape allowlist + the
install() monkeypatch over slime's logging choke points. Keys below are real
ones observed on the round-3 runs."""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _import_with_stubs(modname: str):
    """Permissively stub missing heavy deps (torch, wandb, ...) so the pure-python
    filter logic is testable on a CPU env — same pattern as test_agentic_metrics."""
    for _ in range(60):
        try:
            return importlib.import_module(modname)
        except ImportError as exc:
            missing = exc.name
            if not missing or missing in sys.modules:
                raise
            stub = types.ModuleType(missing)
            stub.__path__ = []
            stub.__getattr__ = lambda _name: MagicMock()  # noqa: B023
            sys.modules[missing] = stub
    return importlib.import_module(modname)


try:
    metric_filter = _import_with_stubs("agentic_rl.obs.metric_filter")
except Exception as exc:  # pragma: no cover - unsatisfiable import env
    pytest.skip(f"agentic_rl.obs.metric_filter unimportable: {exc}", allow_module_level=True)

drop_low_value_metrics = metric_filter.drop_low_value_metrics

RATED_3_KEPT = [
    "rollout/step",
    "rollout/raw_reward",
    "rollout/average_last_reward",
    "dynamic_sampling/raw_reward_all",
    "dynamic_sampling/completed_groups",
    "dynamic_sampling/dropped_groups",
    "agentic/outcome/reward_final/mean",
    "agentic/fresh/outcome/reward_final/mean",
    "agentic/fresh/solved_frac",
    "agentic/retro/samples",
    "perf/rollout_time",
    "perf/actor_train_time",
    "perf/tokens_per_gpu_per_sec",
    "agentic/exec_time_sec/mean",
    "agentic/exec_time_sec/p90",
    "agentic/timing/generate/mean",
    "agentic/timing/episode/max",
    "retro/staleness/fresh_max_behavior_lag",
    "retro/staleness/sequential_legs",
    "retro/mix/accepted",
    "retro/mix/lag_rejected",
    "retro/timing/retro_seconds",
    "async/behavior_lag/mean",
    "async/behavior_lag/max",
    "async/retro/behavior_lag/max",
    "behavior_lag/rejected_groups",
]

RATED_2_KEPT = [
    "agentic/context_length_exceeded_frac",
    "agentic/think_closure_frac",
    "agentic/format_errors/mean",
    "agentic/straggler_ratio",
    "agentic/elapsed_sec/p50",
    "agentic/elapsed_sec/p90",
    "agentic/elapsed_sec/max",
    "agentic/fresh/elapsed_sec/mean",
    "agentic/fresh/elapsed_sec/p90",
    "agentic/turns/mean",
    "agentic/turns/p90",
    "agentic/submissions/p90",
    "agentic/exec_count/mean",
    "agentic/exec_call_sec/max",
    "agentic/fresh/timing/generate/mean",
    "rollout/spec_accept_rate",
    "rollout/prefix_cache_hit_rate",
    "rollout/response_len/mean",
    "rollout/response_len/max",
    "retro/snapshot_capture/latency_seconds/p90",
    "retro/snapshot_capture/trajectory_fraction/mean",
    "async/versions_in_batch",
    "async/version_lag/max",
    "async/sample_age_sec/mean",
    "perf/longest_sample_tokens_per_sec",
]

RATED_1_DROPPED = [
    "agentic/elapsed_sec/p99",
    "agentic/fresh/elapsed_sec/p50",
    "agentic/retro/elapsed_sec/p99",
    "agentic/fresh/elapsed_sec/max",
    "agentic/exec_time_sec/p50",
    "agentic/exec_time_sec/p99",
    "agentic/fresh/exec_time_sec/max",
    "agentic/exec_time_per_call_sec/mean",
    "agentic/exec_time_per_call_sec/max",
    "agentic/exec_count/max",
    "agentic/exec_count/p90",
    "agentic/exec_timeouts/mean",
    "agentic/think_closures/mean",
    "agentic/chains/mean",
    "agentic/output_tokens/max",
    "agentic/context_length_wasted_tokens/max",
    "agentic/turns/max",
    "agentic/submissions/max",
    "agentic/timing/verifier/max",
    "agentic/timing/boot/max",
    "agentic/fresh/timing/generate/max",
    "agentic/fresh/timing/verifier/p90",
    "agentic/retro/timing/episode/max",
    "async/version_span/mean",
    "async/fresh/version_span/max",
    "async/sample_age_sec/max",
    "async/retro/sample_age_sec/max",
    "retro/snapshot_capture/fraction_error/mean",
    "retro/snapshot_capture/fraction_error/p90",
    "retro/snapshot_capture/trajectory_fraction/p50",
    "retro/snapshot_capture/turn_index/p90",
    "retro/snapshot_capture/latency_seconds/p99",
    "retro/snapshot_capture/latency_seconds/max",
    "retro/snapshot_restore/latency_seconds/max",
    "retro/snapshot_capture/estimated_files/mean",
    "rollout/dynamic_filter/drop_zero_std_0.0",
    "rollout/dynamic_filter/drop_zero_std_1.0",
    "rollout/response_len/min",
    "rollout/response_len/median",
    "perf/longest_effective_sample_tokens_per_sec",
]


def test_rated_1_dropped_and_rest_kept():
    metrics = {k: 1.0 for k in RATED_3_KEPT + RATED_2_KEPT + RATED_1_DROPPED}
    out = drop_low_value_metrics(metrics, step_key="rollout/step")
    assert sorted(out) == sorted(RATED_3_KEPT + RATED_2_KEPT)


def test_step_key_survives_even_if_pattern_matched():
    out = drop_low_value_metrics({"agentic/turns/max": 1.0}, step_key="agentic/turns/max")
    assert out == {"agentic/turns/max": 1.0}


def test_log_all_env_bypass(monkeypatch):
    monkeypatch.setenv("SLIME_WANDB_LOG_ALL", "1")
    metrics = {k: 1.0 for k in RATED_1_DROPPED}
    assert drop_low_value_metrics(metrics, step_key="rollout/step") == metrics


def test_train_metrics_untouched():
    metrics = {"train/loss": 0.1, "train/grad_norm": 1.2, "train/step": 5}
    assert drop_low_value_metrics(metrics, step_key="train/step") == metrics


def test_engine_allowlist_covers_requested_gauges():
    keep = metric_filter.ENGINE_METRICS_KEEP
    for k in (
        "sglang_num_running_reqs",
        "sglang_num_queue_reqs",
        "sglang_token_usage",
        "sglang_cache_hit_rate",
        "sglang_gen_throughput",
    ):
        assert k in keep
    for k in (
        "sglang_e2e_request_latency_seconds_bucket",
        "sglang_http_requests_total",
        "sglang_process_cpu_seconds_total",
        "sglang_num_pages",
        "sglang_engine_startup_time",
        "sglang_kv_used_tokens",
    ):
        assert k not in keep


def test_install_wraps_slime_choke_points(monkeypatch):
    """install() must filter what flows through slime's log() and the engine
    scraper's prometheus parse, and be idempotent."""
    logging_utils = types.ModuleType("slime.utils.logging_utils")
    seen = {}

    def raw_log(args, metrics, step_key):
        seen["metrics"] = metrics

    logging_utils.log = raw_log
    wandb_utils = types.ModuleType("slime.utils.wandb_utils")
    wandb_utils._parse_prometheus_text = lambda text: {
        "sglang_num_running_reqs": 12.0,
        "sglang_e2e_request_latency_seconds_bucket": 4.0,
    }
    slime_utils = types.ModuleType("slime.utils")
    slime_utils.logging_utils = logging_utils
    slime_utils.wandb_utils = wandb_utils
    monkeypatch.setitem(sys.modules, "slime.utils", slime_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.wandb_utils", wandb_utils)
    monkeypatch.setattr(metric_filter, "_installed", False)

    metric_filter.install()
    metric_filter.install()  # idempotent: second call must not double-wrap
    assert logging_utils.log is not raw_log

    logging_utils.log(
        SimpleNamespace(),
        {"rollout/step": 5, "rollout/raw_reward": 0.2, "agentic/turns/max": 60.0},
        step_key="rollout/step",
    )
    assert seen["metrics"] == {"rollout/step": 5, "rollout/raw_reward": 0.2}
    assert wandb_utils._parse_prometheus_text("") == {"sglang_num_running_reqs": 12.0}
