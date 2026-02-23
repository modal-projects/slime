"""CI utilities for Megatron backend testing."""

import logging
from collections import defaultdict
from collections.abc import Sequence

import torch
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


def _iter_models(model):
    models = model if isinstance(model, (list, tuple)) else [model]
    return list(models)


def _model_key(idx: int, name: str) -> str:
    return f"[{idx}]{name}"


def _extract_adapter_prefix_and_role(param_name: str) -> tuple[str, str] | None:
    role_patterns = (
        (".linear_in.", "A"),
        (".linear_out.", "B"),
        (".lora_a.", "A"),
        (".lora_b.", "B"),
        (".weight_magnitude", "M"),
    )
    for marker, role in role_patterns:
        if marker in param_name:
            return param_name.split(marker, 1)[0], role
    return None


def _collect_peft_wrapper_modules(models):
    from megatron.bridge.peft.lora import LinearAdapter, ParallelLinearAdapter

    try:
        from megatron.bridge.peft.dora_layers import ParallelLinearDoRAAdapter
    except ImportError:
        ParallelLinearDoRAAdapter = None

    wrapper_names = []
    wrapper_leaf_names = []
    for model_idx, m in enumerate(models):
        for module_name, module in m.named_modules():
            if isinstance(module, LinearAdapter):
                wrapper_names.append(_model_key(model_idx, module_name))
                wrapper_leaf_names.append(module_name.rsplit(".", 1)[-1])
                continue

            adapter_child = None
            base_weight_child = None
            for _child_name, child in module.named_children():
                if ParallelLinearDoRAAdapter is not None and isinstance(child, ParallelLinearDoRAAdapter):
                    adapter_child = child
                elif isinstance(child, ParallelLinearAdapter):
                    adapter_child = child
                elif hasattr(child, "weight") and child.weight is not None:
                    base_weight_child = child

            if adapter_child is not None and base_weight_child is not None:
                wrapper_names.append(_model_key(model_idx, module_name))
                wrapper_leaf_names.append(module_name.rsplit(".", 1)[-1])

    return wrapper_names, wrapper_leaf_names


def check_peft_model_setup(model, peft_type: str, target_modules: Sequence[str] | None = None, exact: bool = False) -> None:
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

    models = _iter_models(model)
    adapter_count = sum(
        1 for m in models for _, mod in m.named_modules()
        if isinstance(mod, (LinearAdapter, ParallelLinearAdapter))
    )
    assert adapter_count >= 100, f"Expected >= 100 adapters, got {adapter_count}"

    total = sum(p.numel() for m in models for p in m.parameters())
    trainable = sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)
    trainable_tensors = sum(1 for m in models for _n, p in m.named_parameters() if p.requires_grad)
    ratio = trainable / total
    assert ratio < 0.02, f"Expected trainable ratio < 2%, got {ratio:.4f} ({ratio*100:.2f}%)"

    if peft_type in {"lora", "canonical_lora"}:
        expected_trainable_tensors = adapter_count * 2
        assert trainable_tensors == expected_trainable_tensors, (
            f"Expected {expected_trainable_tensors} trainable tensors "
            f"({adapter_count} adapters x 2 tensors), got {trainable_tensors}"
        )
    elif peft_type == "dora":
        expected_trainable_tensors = adapter_count * 3
        assert trainable_tensors == expected_trainable_tensors, (
            f"Expected {expected_trainable_tensors} trainable tensors "
            f"({adapter_count} adapters x 3 tensors), got {trainable_tensors}"
        )

    if exact:
        wrapper_names, wrapper_leaf_names = _collect_peft_wrapper_modules(models)
        assert len(wrapper_names) == adapter_count, (
            f"Expected {adapter_count} adapted wrapper modules, found {len(wrapper_names)}"
        )

        if target_modules is None:
            metadata_targets = []
            for m in models:
                raw = getattr(getattr(m, "module", m), "_ci_peft_target_modules", ())
                metadata_targets.extend(raw)
            target_modules = list(dict.fromkeys(metadata_targets))

        if target_modules:
            target_set = set(target_modules)
            unexpected_wrappers = [n for n, leaf in zip(wrapper_names, wrapper_leaf_names, strict=False) if leaf not in target_set]
            assert not unexpected_wrappers, (
                f"Found adapted wrappers outside lora_target_modules {sorted(target_set)}. "
                f"Examples: {unexpected_wrappers[:8]}"
            )

        expected_target_count = 0
        expected_target_names = []
        for model_idx, m in enumerate(models):
            root = getattr(m, "module", m)
            expected_target_count += getattr(root, "_ci_peft_expected_target_count", 0)
            expected_target_names.extend(_model_key(model_idx, n) for n in getattr(root, "_ci_peft_expected_target_names", ()))

        if expected_target_count > 0:
            assert len(wrapper_names) == expected_target_count, (
                f"Expected {expected_target_count} adapted target modules from pre-transform model, "
                f"found {len(wrapper_names)} after PEFT transform"
            )
        if expected_target_names:
            expected_name_set = set(expected_target_names)
            wrapper_name_set = set(wrapper_names)
            missing = sorted(expected_name_set - wrapper_name_set)
            extra = sorted(wrapper_name_set - expected_name_set)
            assert not missing and not extra, (
                "PEFT adapted module set mismatch. "
                f"Missing: {missing[:8]} Extra: {extra[:8]}"
            )

        expected_roles = {"A", "B"} if peft_type in {"lora", "canonical_lora"} else {"A", "B", "M"}
        trainable_groups = defaultdict(set)
        unknown_trainable = []
        for model_idx, m in enumerate(models):
            for name, param in m.named_parameters():
                if not param.requires_grad:
                    continue
                key = _model_key(model_idx, name)
                parsed = _extract_adapter_prefix_and_role(key)
                if parsed is None:
                    unknown_trainable.append(key)
                    continue
                prefix, role = parsed
                trainable_groups[prefix].add(role)

        assert not unknown_trainable, (
            "Found trainable tensors not recognized as PEFT adapter tensors. "
            f"Examples: {unknown_trainable[:8]}"
        )
        assert len(trainable_groups) == adapter_count, (
            f"Expected {adapter_count} adapter parameter groups, found {len(trainable_groups)}"
        )
        bad_groups = [(k, sorted(v)) for k, v in trainable_groups.items() if v != expected_roles]
        assert not bad_groups, (
            f"Adapter parameter-role mismatch for peft_type={peft_type}. "
            f"Expected roles={sorted(expected_roles)}, examples={bad_groups[:8]}"
        )

    logger.info(
        "[CI PEFT Setup] %d adapters, %.2f%% trainable, %d trainable tensors",
        adapter_count,
        ratio * 100,
        trainable_tensors,
    )


def check_peft_grad_flow(model, step_id: int, peft_type: str, exact: bool = False, rollout_id: int = 0) -> None:
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
    models = _iter_models(model)
    adapter_with_grad = 0
    adapter_total = 0
    frozen_with_grad = 0
    frozen_with_grad_names = []
    adapter_without_grad_names = []
    adapter_grad_by_name = {}
    for model_idx, m in enumerate(models):
        for name, param in m.named_parameters():
            full_name = _model_key(model_idx, name)
            # main_grad may exist but be None; fall back to param.grad in that case.
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                if param.requires_grad:
                    adapter_total += 1
                    adapter_without_grad_names.append(full_name)
                    adapter_grad_by_name[full_name] = 0.0
                continue
            grad_max = grad.abs().max().item()
            if param.requires_grad:
                adapter_total += 1
                adapter_grad_by_name[full_name] = grad_max
                if grad_max > 0:
                    adapter_with_grad += 1
                else:
                    adapter_without_grad_names.append(full_name)
            else:
                if grad_max > 0:
                    frozen_with_grad += 1
                    if len(frozen_with_grad_names) < 8:
                        frozen_with_grad_names.append(full_name)

    assert adapter_with_grad > 0, f"Step {step_id}: No trainable adapter tensors received gradients"
    assert frozen_with_grad == 0, (
        f"Step {step_id}: {frozen_with_grad} frozen params have non-zero grad. "
        f"Examples: {frozen_with_grad_names}"
    )

    # Stronger invariants:
    # - LoRA/canonical_lora: first step often has zero grad for A (B starts at zero),
    #   but should still exercise at least ~half of trainable tensors.
    # - DoRA: first step should exercise at least ~2/3 tensors (B + magnitude).
    # - Later steps should activate almost all trainable tensors.
    if adapter_total > 0:
        if peft_type in {"lora", "canonical_lora"}:
            min_ratio = 0.45 if step_id == 0 else 0.95
        elif peft_type == "dora":
            min_ratio = 0.60 if step_id == 0 else 0.95
        else:
            min_ratio = 0.10
        min_required = max(1, int(adapter_total * min_ratio))
        assert adapter_with_grad >= min_required, (
            f"Step {step_id}: too few trainable adapter tensors with non-zero grad: "
            f"{adapter_with_grad}/{adapter_total}, expected at least {min_required} "
            f"(peft_type={peft_type}). Missing examples: {adapter_without_grad_names[:8]}"
        )

    if exact:
        expected_roles = {"A", "B"} if peft_type in {"lora", "canonical_lora"} else {"A", "B", "M"}
        grouped_roles = defaultdict(set)
        grouped_grad = defaultdict(dict)
        unknown_adapter_params = []
        for name, grad_max in adapter_grad_by_name.items():
            parsed = _extract_adapter_prefix_and_role(name)
            if parsed is None:
                unknown_adapter_params.append(name)
                continue
            prefix, role = parsed
            grouped_roles[prefix].add(role)
            grouped_grad[prefix][role] = grad_max

        assert not unknown_adapter_params, (
            "Found trainable adapter params with unrecognized naming pattern. "
            f"Examples: {unknown_adapter_params[:8]}"
        )
        bad_roles = [(k, sorted(v)) for k, v in grouped_roles.items() if v != expected_roles]
        assert not bad_roles, (
            f"Step {step_id}: adapter role mismatch for peft_type={peft_type}. "
            f"Expected roles={sorted(expected_roles)} examples={bad_roles[:8]}"
        )

        violations = []
        is_cold_start_step = rollout_id == 0 and step_id == 0
        for prefix, role_grads in grouped_grad.items():
            if peft_type in {"lora", "canonical_lora"}:
                if is_cold_start_step:
                    if role_grads.get("B", 0.0) <= 0:
                        violations.append(f"{prefix}: expected non-zero grad on B at step0, got {role_grads.get('B', 0.0)}")
                    if role_grads.get("A", 0.0) > 0:
                        violations.append(f"{prefix}: expected zero grad on A at step0, got {role_grads.get('A', 0.0)}")
                else:
                    if role_grads.get("A", 0.0) <= 0:
                        violations.append(f"{prefix}: expected non-zero grad on A at step{step_id}")
                    if role_grads.get("B", 0.0) <= 0:
                        violations.append(f"{prefix}: expected non-zero grad on B at step{step_id}")
            elif peft_type == "dora":
                if is_cold_start_step:
                    if role_grads.get("B", 0.0) <= 0:
                        violations.append(f"{prefix}: expected non-zero grad on B at step0, got {role_grads.get('B', 0.0)}")
                    if role_grads.get("M", 0.0) <= 0:
                        violations.append(f"{prefix}: expected non-zero grad on M at step0, got {role_grads.get('M', 0.0)}")
                    if role_grads.get("A", 0.0) > 0:
                        violations.append(f"{prefix}: expected zero grad on A at step0, got {role_grads.get('A', 0.0)}")
                else:
                    if role_grads.get("A", 0.0) <= 0:
                        violations.append(f"{prefix}: expected non-zero grad on A at step{step_id}")
                    if role_grads.get("B", 0.0) <= 0:
                        violations.append(f"{prefix}: expected non-zero grad on B at step{step_id}")
                    if role_grads.get("M", 0.0) <= 0:
                        violations.append(f"{prefix}: expected non-zero grad on M at step{step_id}")

        assert not violations, (
            f"Step {step_id}: strict adapter grad semantic checks failed for peft_type={peft_type}. "
            f"Examples: {violations[:8]}"
        )

    logger.info(
        "[CI PEFT Grad] Step %d: %d/%d trainable adapter tensors with grad",
        step_id,
        adapter_with_grad,
        adapter_total,
    )


def check_peft_weight_merge(merge_count: int, total_tasks: int, expected_merge_count: int) -> None:
    """Verify LoRA/DoRA deltas were merged during weight sync.

    Called from ``_MapWithLenAndMergeCount.__iter__`` after iteration completes
    when ``ci_test`` is enabled. Checks that at least one layer was merged.

    Args:
        merge_count: Number of layers that had LoRA deltas merged.
        total_tasks: Total number of conversion tasks iterated.

    Raises:
        AssertionError: If no layers were merged.
    """
    assert expected_merge_count > 0, f"Expected merge targets > 0, got {expected_merge_count}"
    assert merge_count > 0, f"Expected merged layers > 0, got {merge_count}"
    assert merge_count == expected_merge_count, (
        f"Merged layer count mismatch: merged {merge_count}, expected {expected_merge_count} "
        f"(conversion tasks: {total_tasks})"
    )
    logger.info(
        "[CI PEFT Merge] Merged %d/%d tasks (expected merged layers: %d)",
        merge_count,
        total_tasks,
        expected_merge_count,
    )


def check_peft_exact_weight_sync(
    *,
    sync_round: int,
    expected_merge_count: int,
    expected_keys: set[str],
    matched_keys: set[str],
    duplicate_key_count: int,
    current_merged_by_task: dict[str, "object"],
    prev_merged_by_task: dict[str, "object"] | None,
) -> None:
    """Strict PEFT merge checks with exact key coverage and cross-sync delta sanity."""
    assert expected_merge_count > 0, "Strict PEFT merge CI expected at least one adapted layer, got 0"
    assert len(expected_keys) == expected_merge_count, (
        f"Expected key map size {expected_merge_count}, got {len(expected_keys)}"
    )
    assert duplicate_key_count == 0, (
        f"Expected each adapted key to be merged once, found {duplicate_key_count} duplicates"
    )
    assert matched_keys == expected_keys, (
        f"Merged key set mismatch. "
        f"Missing={len(expected_keys - matched_keys)} "
        f"Extra={len(matched_keys - expected_keys)}"
    )
    assert len(current_merged_by_task) == expected_merge_count, (
        f"Expected {expected_merge_count} merged task keys, got {len(current_merged_by_task)}"
    )
    curr_keys = set(current_merged_by_task)

    if prev_merged_by_task is None:
        logger.info(
            "[CI PEFT Exact Merge] Sync %d: captured baseline for %d adapted layers",
            sync_round,
            len(current_merged_by_task),
        )
        return

    prev_keys = set(prev_merged_by_task)
    assert curr_keys == prev_keys, (
        f"Cross-sync adapted-layer key mismatch. Missing={len(prev_keys - curr_keys)} Extra={len(curr_keys - prev_keys)}"
    )

    changed = 0
    max_abs_delta = 0.0
    for key, curr in current_merged_by_task.items():
        actual_delta = curr - prev_merged_by_task[key]

        delta = actual_delta.abs().max().item()
        if delta > 0:
            changed += 1
        if delta > max_abs_delta:
            max_abs_delta = delta

    assert changed > 0, "No adapted merged layer changed across sync rounds; expected PEFT updates between syncs"

    logger.info(
        "[CI PEFT Exact Merge] Sync %d: %d/%d adapted layers changed, max |delta|=%.6g",
        sync_round,
        changed,
        len(current_merged_by_task),
        max_abs_delta,
    )


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
