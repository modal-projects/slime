from argparse import Namespace
from typing import Any

import torch


@torch.compile(dynamic=True)
def _cispo_policy_loss(
    ppo_kl: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    ratio_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    negative_approx_kl = torch.clamp(-ppo_kl, min=-20.0, max=20.0)
    ratio = negative_approx_kl.exp()
    clipped_ratio = ratio.clamp(max=ratio_max)

    pg_losses = -clipped_ratio.detach() * advantages * log_probs
    clipfrac = (ratio != clipped_ratio).float()

    return pg_losses, clipfrac


def compute_policy_loss(
    *,
    args: Namespace,
    ppo_kl: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CISPO policy-loss hook.

    This follows the TRL/ScaleRL convention: cap the IS ratio directly at
    ``policy_loss_ratio_max`` and apply stop-gradient to that capped weight.
    """
    ratio_max = args.policy_loss_ratio_max if args.policy_loss_ratio_max is not None else 5.0
    if ratio_max <= 0:
        raise ValueError(f"policy_loss_ratio_max must be positive for CISPO, got {ratio_max}.")

    return _cispo_policy_loss(ppo_kl, log_probs, advantages, ratio_max)
