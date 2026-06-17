import logging

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def update_tracking_open_metrics(args, router_addr):
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        logger_actor = wandb_utils.get_logger_actor()
        if logger_actor is not None:
            import ray

            ray.get(logger_actor.finish.remote(), timeout=120)
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb logger actor")
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        # All history must go through the single primary writer; metrics
        # logged from shared-mode secondary processes are ingested hours
        # late (or dropped) by the W&B backend. See wandb_utils. The call is
        # synchronous (it is cheap and infrequent) so that no metric can be
        # lost in a shutdown race and actor failures are surfaced here.
        logger_actor = wandb_utils.get_logger_actor()
        if logger_actor is not None:
            try:
                import ray

                ray.get(logger_actor.log.remote(metrics), timeout=60)
            except Exception:
                logging.getLogger(__name__).exception("Failed to log metrics via wandb logger actor")
        elif wandb.run is not None:
            wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
