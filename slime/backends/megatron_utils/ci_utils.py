"""CI utilities for Megatron backend testing."""

import logging
from collections.abc import Sequence

from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def check_mtp_only_grad(model: Sequence[DDP], step_id: int) -> None:
    """Check that only MTP parameters have non-zero gradients.

    This is used for CI testing to verify that when all outputs are truncated,
    only the MTP layers receive gradients (since only mtp_loss contributes).

    Args:
        model: Sequence of DDP-wrapped model chunks.
        step_id: Current step index for logging.

    Raises:
        AssertionError: If any non-MTP parameter has a non-zero gradient.
    """
    non_mtp_nonzero_grads = []
    mtp_nonzero_grads = []

    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            # Get the main_grad from the distributed optimizer if available
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                continue

            grad_norm = grad.abs().max().item()
            is_mtp = ".mtp." in name

            if is_mtp:
                if grad_norm > 0:
                    mtp_nonzero_grads.append((name, grad_norm))
            else:
                if grad_norm > 0:
                    non_mtp_nonzero_grads.append((name, grad_norm))

    # Log the results
    logger.info(
        f"[CI MTP Grad Check] Step {step_id}: "
        f"MTP params with non-zero grad: {len(mtp_nonzero_grads)}, "
        f"non-MTP params with non-zero grad: {len(non_mtp_nonzero_grads)}"
    )

    if non_mtp_nonzero_grads:
        # Log the first few non-MTP params with non-zero gradients for debugging
        for name, grad_norm in non_mtp_nonzero_grads[:5]:
            logger.error(f"[CI MTP Grad Check] Non-MTP param with non-zero grad: {name}, max_grad={grad_norm}")

    assert len(non_mtp_nonzero_grads) == 0, (
        f"Expected all non-MTP parameters to have zero gradients, "
        f"but found {len(non_mtp_nonzero_grads)} with non-zero gradients. "
        f"First few: {non_mtp_nonzero_grads[:5]}"
    )

    # Also verify that MTP params do have gradients (otherwise the test is not valid)
    assert len(mtp_nonzero_grads) > 0, (
        "Expected MTP parameters to have non-zero gradients, but all were zero. "
        "This may indicate the MTP loss is not being computed."
    )


def check_peft_model_setup(model, peft_type: str) -> None:
    """Verify PEFT model has correct adapter structure and freeze pattern.

    Called once after model initialization when ``args.ci_test`` is True and a
    PEFT method is active. Checks that:
      - At least 100 adapter modules were injected.
      - Trainable parameter ratio is below 2%.

    Args:
        model: Model or list of DDP-wrapped model chunks.
        peft_type: The PEFT type string (e.g., "lora", "dora").

    Raises:
        AssertionError: If adapter count or trainable ratio is out of bounds.
    """
    from megatron.bridge.peft.lora import LinearAdapter, ParallelLinearAdapter

    models = model if isinstance(model, (list, tuple)) else [model]
    adapter_count = sum(
        1 for m in models for _, mod in m.named_modules()
        if isinstance(mod, (LinearAdapter, ParallelLinearAdapter))
    )
    assert adapter_count >= 100, f"Expected >= 100 adapters, got {adapter_count}"

    total = sum(p.numel() for m in models for p in m.parameters())
    trainable = sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)
    ratio = trainable / total
    assert ratio < 0.02, f"Expected trainable ratio < 2%, got {ratio:.4f} ({ratio*100:.2f}%)"

    logger.info(f"[CI PEFT Setup] {adapter_count} adapters, {ratio*100:.2f}% trainable")


def check_peft_grad_flow(model, step_id: int) -> None:
    """Verify gradients flow correctly through PEFT adapters.

    Called after the backward pass (same location as ``check_mtp_only_grad``)
    when ``args.ci_test`` is True and a PEFT method is active. Checks that:
      - At least one adapter (trainable) parameter received a non-zero gradient.
      - No frozen parameter has a non-zero gradient.

    Args:
        model: Model or list of DDP-wrapped model chunks.
        step_id: Current step index for logging.

    Raises:
        AssertionError: If gradient flow is incorrect.
    """
    models = model if isinstance(model, (list, tuple)) else [model]
    adapter_with_grad = 0
    frozen_with_grad = 0
    for m in models:
        for name, param in m.named_parameters():
            grad = getattr(param, "main_grad", param.grad)
            if grad is None:
                continue
            grad_max = grad.abs().max().item()
            if param.requires_grad:
                if grad_max > 0:
                    adapter_with_grad += 1
            else:
                if grad_max > 0:
                    frozen_with_grad += 1

    assert adapter_with_grad > 0, f"Step {step_id}: No adapter params received gradients"
    assert frozen_with_grad == 0, f"Step {step_id}: {frozen_with_grad} frozen params have non-zero grad"
    logger.info(f"[CI PEFT Grad] Step {step_id}: {adapter_with_grad} adapter params with grad")


def check_peft_weight_merge(merge_count: int, total_tasks: int) -> None:
    """Verify LoRA/DoRA deltas were merged during weight sync.

    Called from ``_MapWithLenAndMergeCount.__iter__`` after iteration completes
    when ``ci_test`` is enabled. Checks that at least one layer was merged.

    Args:
        merge_count: Number of layers that had LoRA deltas merged.
        total_tasks: Total number of conversion tasks iterated.

    Raises:
        AssertionError: If no layers were merged.
    """
    assert merge_count > 0, f"Expected merged layers > 0, got {merge_count}"
    logger.info(f"[CI PEFT Merge] Merged {merge_count}/{total_tasks} tasks")


def check_mtp_loss(mtp_loss: float, max_mtp_loss: float = 1.0) -> None:
    """Check that MTP loss is within expected bounds.

    Args:
        mtp_loss: The computed MTP loss value.
        max_mtp_loss: Maximum allowed MTP loss (default: 1.0).

    Raises:
        AssertionError: If MTP loss exceeds the maximum allowed value.
    """
    assert mtp_loss < max_mtp_loss, (
        f"MTP loss {mtp_loss} exceeds maximum allowed value {max_mtp_loss}. "
        "This may indicate an issue with MTP training."
    )
