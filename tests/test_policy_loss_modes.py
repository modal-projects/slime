from argparse import Namespace

import pytest
import torch

from slime.utils.misc import load_function
from slime.utils.ppo_utils import compute_policy_loss
from slime_plugins.losses.cispo import compute_policy_loss as compute_cispo_policy_loss


@pytest.mark.unit
def test_vanilla_policy_loss_matches_clipped_surrogate():
    ppo_kl = torch.tensor([-0.7, 0.3, 0.0], dtype=torch.float32)
    advantages = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)

    pg_losses, clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.4)

    ratio = torch.exp(-ppo_kl)
    unclipped = -ratio * advantages
    clipped = -ratio.clamp(0.8, 1.4) * advantages

    assert torch.allclose(pg_losses, torch.maximum(unclipped, clipped))
    assert torch.equal(clipfrac, torch.gt(clipped, unclipped).float())


@pytest.mark.unit
def test_cispo_policy_loss_hook_matches_scalerl_formula():
    ratio = torch.tensor([6.0, 0.1, 1.0], dtype=torch.float32)
    ppo_kl = -torch.log(ratio)
    log_probs = torch.tensor([-2.0, -1.0, -0.25], dtype=torch.float32)
    advantages = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    args = Namespace(policy_loss_ratio_max=5.0)

    pg_losses, clipfrac = compute_cispo_policy_loss(
        args=args,
        ppo_kl=ppo_kl,
        log_probs=log_probs,
        advantages=advantages,
    )

    expected_ratio = torch.clamp(torch.exp(torch.clamp(-ppo_kl, min=-20.0, max=20.0)), max=5.0)
    expected_losses = -expected_ratio.detach() * advantages * log_probs

    assert torch.allclose(pg_losses, expected_losses)
    assert torch.equal(clipfrac, torch.tensor([1.0, 0.0, 0.0]))


@pytest.mark.unit
def test_cispo_policy_loss_stops_gradient_through_ratio():
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
    log_probs = torch.tensor([-0.3, -3.0, -1.0], dtype=torch.float32, requires_grad=True)
    advantages = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    ppo_kl = old_log_probs - log_probs
    args = Namespace(policy_loss_ratio_max=1.2)

    pg_losses, _ = compute_cispo_policy_loss(
        args=args,
        ppo_kl=ppo_kl,
        log_probs=log_probs,
        advantages=advantages,
    )
    pg_losses.sum().backward()

    with torch.no_grad():
        ratio = torch.exp(torch.clamp(-(old_log_probs - log_probs), min=-20.0, max=20.0))
        clipped_ratio = ratio.clamp(max=1.2)
        expected_grad = -clipped_ratio * advantages

    assert torch.allclose(log_probs.grad, expected_grad)


@pytest.mark.unit
def test_cispo_policy_loss_hook_loads_by_path():
    hook = load_function("slime_plugins.losses.cispo.compute_policy_loss")
    assert hook is compute_cispo_policy_loss
