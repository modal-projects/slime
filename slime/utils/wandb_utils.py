import logging
import math
import os
import threading
import time
import urllib.request
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)

# Background scraper state (module-level: shared mode has no single logger actor
# to hang it off, so the process that owns the W&B run — the RolloutManager —
# starts it once).
_engine_metrics_stop: threading.Event | None = None
_engine_metrics_thread: threading.Thread | None = None


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if getattr(args, "use_critic", False):
        critic_args = _get_role_args_for_logging(args, role="critic")
        output.update(_prefix_config_keys(_args_to_config_dict(critic_args), "critic"))

    return output


def _args_to_config_dict(args):
    return deepcopy(args.__dict__)


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _get_role_args_for_logging(args, role):
    if getattr(args, "megatron_config_path", None) is None:
        return args

    from slime.utils.arguments import parse_megatron_role_args

    return parse_megatron_role_args(args, args.megatron_config_path, role=role)


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, role=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
    # Live SGLang serving gauges scraped off the router, plotted against their
    # own wall-clock axis (they are sampled on a timer, not per rollout step).
    wandb.define_metric("sgl_engine/uptime_sec")
    wandb.define_metric("sgl_engine/*", step_metric="sgl_engine/uptime_sec")


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition into {metric_name: mean across series}."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name = line[: line.index("{")]
            rest = line[line.index("}") + 1 :].split()
        else:
            parts = line.split()
            name, rest = parts[0], parts[1:]
        if not rest:
            continue
        try:
            value = float(rest[0])
        except ValueError:
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}


def start_engine_metrics_scraping(args, router_addr: str | None, interval_sec: float = 30.0) -> None:
    """Mirror the sglang router's ``/engine_metrics`` gauges into W&B.

    The router aggregates each backend engine's Prometheus metrics (running/queue
    requests, generation throughput, token/KV usage, cache hit rate, ...); this
    samples that endpoint on a timer and logs the mean across engines under
    ``sgl_engine/*``. Runs as a daemon thread in whichever process owns the W&B
    run (the RolloutManager, a shared-mode secondary writer). Best-effort:
    transient scrape failures are swallowed so it never disturbs the loop.
    """
    global _engine_metrics_stop, _engine_metrics_thread
    if not args.use_wandb or router_addr is None or _engine_metrics_thread is not None:
        return

    # Only the customized sgl-router (zhuzilin/sgl-router, whose version string
    # carries "slime") exposes the aggregated /engine_metrics endpoint.
    try:
        import sglang_router

        if "slime" not in sglang_router.__version__:
            logger.warning(
                "sglang_router %s does not expose /engine_metrics; skipping sgl_engine/* "
                "scraping (needs zhuzilin/sgl-router).",
                sglang_router.__version__,
            )
            return
    except Exception:
        return

    url = f"{router_addr}/engine_metrics"
    _engine_metrics_stop = threading.Event()
    _engine_metrics_thread = threading.Thread(
        target=_engine_metrics_loop,
        args=(url, _engine_metrics_stop, interval_sec),
        daemon=True,
    )
    _engine_metrics_thread.start()
    logger.info(f"Scraping SGLang engine metrics from {url} every {interval_sec:.0f}s -> sgl_engine/*")


def _engine_metrics_loop(url: str, stop: threading.Event, interval_sec: float) -> None:
    start = time.monotonic()
    while not stop.wait(interval_sec):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        metrics = _parse_prometheus_text(text)
        if not metrics or wandb.run is None:
            continue
        payload = {f"sgl_engine/{name}": value for name, value in metrics.items()}
        payload["sgl_engine/uptime_sec"] = time.monotonic() - start
        try:
            wandb.log(payload)
        except Exception:
            logger.exception("Failed to log SGLang engine metrics to W&B")


def stop_engine_metrics_scraping() -> None:
    global _engine_metrics_stop, _engine_metrics_thread
    if _engine_metrics_stop is not None:
        _engine_metrics_stop.set()
    _engine_metrics_thread = None
