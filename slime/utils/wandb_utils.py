import logging
import math
import os
import socket
import threading
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)

# Name of the Ray actor that owns the W&B run and performs ALL history writes.
#
# Why a single writer: on the current W&B backend, history logged by
# ``mode="shared"`` secondary writers (``x_primary=False``) is not ingested in
# real time — it lands hours late via a backfill path, or is dropped entirely
# when the writer process dies before flushing. The same delayed path swallows
# everything logged after a run is finished and re-initialized with
# ``resume="allow"``. So the run looks completely empty in the UI during (and
# long after) training. Funneling every ``wandb.log`` through the one primary
# writer that created the run keeps metrics on the real-time path. Secondary
# processes still attach in shared mode, but only for console logs and
# per-node system metrics.
LOGGER_ACTOR_NAME = "slime_wandb_logger"

_logger_actor = None


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _primary_init_kwargs(args):
    """Build the wandb.init kwargs shared by the offline path and the logger actor."""
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    return init_kwargs


class WandbLoggerActor:
    """Ray actor that owns the W&B run and is its only history writer.

    See the comment on ``LOGGER_ACTOR_NAME`` for why all metrics must flow
    through this single process.
    """

    def __init__(self, args):
        self.args = args
        self._scraper_thread = None
        self._stop = threading.Event()

        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_key is not None:
            wandb.login(key=args.wandb_key, host=args.wandb_host)

        init_kwargs = _primary_init_kwargs(args)
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True, x_label="primary-logger")
        wandb.init(**init_kwargs)
        _init_wandb_common()

    def get_run_id(self):
        return wandb.run.id

    def log(self, metrics: dict):
        if wandb.run is not None:
            wandb.log(metrics)

    def start_open_metrics(self, router_addr: str):
        """Poll the sglang router's metrics endpoint and log it as history.

        Replaces the previous finish + re-init with
        ``x_stats_open_metrics_endpoints``: resuming a finished shared-mode
        run sends all subsequent metric streams down the W&B backfill path
        (hours of delay), which made runs look empty.
        """
        if self._scraper_thread is not None:
            return
        url = f"{router_addr}/engine_metrics"
        self._scraper_thread = threading.Thread(target=self._scrape_loop, args=(url,), daemon=True)
        self._scraper_thread.start()
        logger.info(f"Scraping SGLang engine metrics from {url}.")

    def _scrape_loop(self, url):
        import urllib.request

        while not self._stop.wait(30):
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
            except Exception:
                continue
            metrics = _parse_prometheus_text(text)
            if metrics and wandb.run is not None:
                wandb.log({f"sgl_engine/{name}": value for name, value in metrics.items()})

    def finish(self):
        # Idempotent: called from RolloutManager.dispose() and again from the
        # driver's finish_tracking().
        self._stop.set()
        if wandb.run is not None:
            wandb.finish()


def _parse_prometheus_text(text):
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


def get_logger_actor():
    """Return the primary logger actor handle, or None (e.g. offline mode)."""
    global _logger_actor
    if _logger_actor is None:
        try:
            import ray

            if not ray.is_initialized():
                return None
            _logger_actor = ray.get_actor(LOGGER_ACTOR_NAME)
        except Exception:
            return None
    return _logger_actor


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

    if _is_offline_mode(args) or args.wandb_mode == "disabled":
        # Offline/disabled: every process writes locally (or not at all); no
        # actor needed. For "disabled", the WANDB_MODE env var set above must
        # stay in charge — an explicit Settings(mode=...) would override it.
        init_kwargs = _primary_init_kwargs(args)
        if _is_offline_mode(args):
            init_kwargs["settings"] = wandb.Settings(mode="offline")
        wandb.init(**init_kwargs)
        _init_wandb_common()
        args.wandb_run_id = wandb.run.id
        return

    if args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    import ray

    global _logger_actor
    _logger_actor = ray.remote(num_cpus=0)(WandbLoggerActor).options(name=LOGGER_ACTOR_NAME).remote(args)
    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = ray.get(_logger_actor.get_run_id.remote())

    # Attach the driver as a shared-mode secondary so its console output and
    # head-node system metrics still reach the run.
    init_wandb_secondary(args, role="driver")


def reinit_wandb_primary_with_open_metrics(args, router_addr):
    """Start uploading SGLang engine metrics now that the router is up.

    The primary wandb init happens before rollout servers start (to obtain
    ``wandb_run_id`` for secondary processes). This function is called
    *after* servers are up so the router address is available. The logger
    actor scrapes the router's Prometheus endpoint itself — the previous
    finish + re-init with ``x_stats_open_metrics_endpoints`` made the W&B
    backend route all subsequent metrics through its hours-delayed backfill
    path, so runs looked empty.
    """
    if not args.use_wandb or _is_offline_mode(args):
        return
    if getattr(args, "wandb_mode", None) == "disabled":
        return
    if router_addr is None:
        return
    if getattr(args, "wandb_run_id", None) is None:
        return

    import sglang_router

    if "slime" not in sglang_router.__version__:
        logger.warning(
            "Only customized sglang_router from https://github.com/zhuzilin/sgl-router supports uploading metrics."
        )
        return

    actor = get_logger_actor()
    if actor is None:
        return
    actor.start_open_metrics.remote(router_addr)


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
            # Distinct labels keep per-process system metrics and console
            # logs from clobbering each other on the W&B backend.
            x_label=f"{role or 'worker'}-{socket.gethostname()}-{os.getpid()}",
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
