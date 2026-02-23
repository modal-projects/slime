import dataclasses

import torch
import torch.distributed as dist

from slime.utils import megatron_bridge_utils
from slime.utils.misc import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        self._ci_prev_merged_by_task = None
        self._ci_sync_round = 0

    def get_hf_weight_chunks(self, megatron_local_weights):
        peft_type = getattr(self.args, "peft_type", "none")
        ci_peft_exact = getattr(self.args, "ci_peft_exact", False)

        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        exact_state = {}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            # Pre-compute LoRA/DoRA deltas BEFORE entering the bridge's export
            # flow.  Discovers adapters by walking the model's AdapterWrapper
            # modules and keys the delta map by module name.  This must happen
            # before export_hf_weights because it performs NCCL all_gather calls
            # for TP gathering of A/B matrices, which cannot be interleaved with
            # the bridge's internal TP communication inside the per-task iteration.
            lora_delta_map = _build_adapter_delta_map(
                self._bridge, self.model, peft_type,
            )

            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(
                conversion_tasks, renamed_megatron_local_weights, lora_delta_map,
                ci_test=getattr(self.args, "ci_test", False),
                ci_peft_exact=ci_peft_exact,
                prev_merged_by_task=self._ci_prev_merged_by_task,
                sync_round=self._ci_sync_round,
                exact_state=exact_state,
            )

            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            named_weights = (
                (
                    hf_param_name,
                    postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    ),
                )
                for hf_param_name, weight, megatron_param_name in named_weights
            )

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)

        if ci_peft_exact:
            self._ci_prev_merged_by_task = exact_state.get("current_merged_by_task")
            self._ci_sync_round += 1


@torch.no_grad()
def _build_adapter_delta_map(bridge, model, peft_type):
    """Discover adapters by walking model modules and pre-compute per-rank LoRA/DoRA deltas.

    Walks the model's module tree to find adapter wrapper modules (modules with
    ``to_wrap`` and ``adapter`` children, which is Bridge's standard structure
    for LoRALinear/DoRALinear).  Returns a dict keyed by the base module name
    mapping to pre-computed delta tensors (LoRA) or ``(delta, dora_scale)``
    tuples (DoRA).

    For ParallelLinearAdapter with tensor parallelism, the LoRA matrices may be
    sharded across TP ranks.  We selectively gather only the dimensions needed
    so that the resulting delta matches the per-rank base weight shape:

      - ColumnParallel base: A has rank dim sharded on dim 0, B has output dim
        sharded on dim 0.  Gathering A on dim 0 fixes the matmul inner dims.
      - RowParallel base: inner dims already match.  Gathering B on dim 0
        produces delta ``(out, in/TP)`` matching the base.

    For DoRA adapters, additionally computes a per-row magnitude scale:
      ``dora_scale = weight_magnitude / ||base_weight + delta||_rows``

    The model may be offloaded to CPU via torch_memory_saver, so we use
    ``_get_safe_tensor()`` to access weights from CPU backups.

    As a side-effect, clears Bridge's cached adapter info so that
    ``export_hf_weights`` does not attempt a second merge on top of the deltas
    we already folded in.
    """
    if peft_type == "none":
        return {}

    # Detect adapter wrapper modules.  Try Bridge's base class first;
    # fall back to structural detection if the import path changed.
    try:
        from megatron.bridge.peft.adapter_wrapper import AdapterWrapper

        def _is_adapter_wrapper(mod):
            return isinstance(mod, AdapterWrapper)
    except ImportError:
        def _is_adapter_wrapper(mod):
            return (
                hasattr(mod, "to_wrap") and hasattr(mod, "adapter")
                and isinstance(getattr(mod, "to_wrap", None), torch.nn.Module)
                and hasattr(mod.to_wrap, "weight")
            )

    models = model if isinstance(model, (list, tuple)) else [model]

    delta_map = {}
    adapter_count = 0
    dora_count = 0

    for _vp_idx, m in enumerate(models):
        for name, mod in m.named_modules():
            if not _is_adapter_wrapper(mod):
                continue

            # Strip all DDP/wrapper "module." prefixes (there may be multiple
            # from DDP + Float16Module) to match Bridge's _unwrap_name.
            clean_name = name
            while clean_name.startswith("module."):
                clean_name = clean_name[len("module."):]

            adapter = mod.adapter
            base_weight = mod.to_wrap.weight
            base_shape = base_weight.shape

            # Handle ModuleDict (multi-adapter) vs single adapter.
            if isinstance(adapter, torch.nn.ModuleDict):
                adapter_items = list(adapter.items())
            else:
                adapter_items = [("default", adapter)]

            for _key, sub_adapter in adapter_items:
                # Get LoRA A/B weights (parallel vs non-parallel naming).
                if hasattr(sub_adapter, "linear_in"):
                    lora_a_ref = sub_adapter.linear_in.weight
                    lora_b_ref = sub_adapter.linear_out.weight
                else:
                    lora_a_ref = sub_adapter.lora_a.weight
                    lora_b_ref = sub_adapter.lora_b.weight

                alpha = getattr(sub_adapter, "alpha", None)
                dim = getattr(sub_adapter, "dim", None)
                if alpha is None or dim is None:
                    print(f"Warning: adapter at {clean_name} missing alpha/dim, skipping")
                    continue
                scale = alpha / dim

                is_expert = bool(getattr(sub_adapter, "is_expert", False))
                magnitude = getattr(sub_adapter, "weight_magnitude", None)

                lora_a = _get_safe_tensor(lora_a_ref).cuda()
                lora_b = _get_safe_tensor(lora_b_ref).cuda()

                # TP-gather A/B as needed so B @ A produces a per-rank-shaped delta.
                lora_a, lora_b = _gather_parallel_lora_ab(lora_a, lora_b, base_shape, is_expert)

                if magnitude is not None:
                    # DoRA: compute delta + per-row magnitude scale.
                    delta = _compute_lora_delta(lora_b, lora_a, scale)
                    base_w = _get_safe_tensor(base_weight).cuda()
                    combined = base_w + delta.to(base_w.dtype)
                    row_norms = torch.linalg.norm(combined.float(), dim=1)
                    mag = _get_safe_tensor(magnitude).cuda()
                    dora_scale = (mag / row_norms).to(delta.dtype)
                    delta_map[clean_name] = (delta, dora_scale)
                    dora_count += 1
                else:
                    delta_map[clean_name] = _compute_lora_delta(lora_b, lora_a, scale)

                adapter_count += 1

    if delta_map:
        parts = [f"{adapter_count} adapters"]
        if dora_count:
            parts.append(f"{dora_count} DoRA")
        print(f"LoRA weight sync: Pre-computed {len(delta_map)} merge deltas ({', '.join(parts)})")
    elif peft_type != "none":
        print(f"LoRA weight sync WARNING: peft_type={peft_type} but no merge deltas computed")

    # Prevent double-merge: clear any cached adapter info so Bridge's
    # export_hf_weights doesn't attempt a second merge.
    model_bridge = bridge._model_bridge
    if hasattr(model_bridge, "_cached_param_objects_adapter"):
        model_bridge._cached_param_objects_adapter = []

    return delta_map


def _compute_lora_delta(lora_b, lora_a, scale):
    """Compute delta = (B @ A) * scale using rank-1 outer product accumulation.

    Uses iterative outer products instead of a single matmul to avoid cuBLAS,
    whose workspace memory may be freed after torch_memory_saver.pause().
    Since LoRA rank is small (typically 8-64), this requires few iterations
    of element-wise CUDA ops and avoids any CPU<->GPU data transfer.
    """
    dtype = lora_b.dtype
    rank = lora_b.shape[1]
    b_f = lora_b.float()
    a_f = lora_a.float()
    delta = torch.zeros(lora_b.shape[0], lora_a.shape[1],
                        device=lora_b.device, dtype=torch.float32)
    for r in range(rank):
        delta += b_f[:, r:r+1] * a_f[r:r+1, :]
    delta *= scale
    return delta.to(dtype=dtype)


def _gather_parallel_lora_ab(lora_a, lora_b, base_shape, is_expert=False):
    """Gather LoRA A/B matrices so that B @ A produces a per-rank-shaped delta.

    Selectively gathers LoRA matrices along the dimensions needed to match
    the per-rank base weight shape.
    """
    # Fix inner dims: if B.shape[1] != A.shape[0], A's rank dim
    # is sharded (ColumnParallel base) → gather A on dim 0.
    if lora_b.shape[1] != lora_a.shape[0]:
        lora_a = _all_gather_tp(lora_a, dim=0, is_expert=is_expert)
    # Fix outer dims: if B.shape[0] != base_shape[0], B's output dim
    # is sharded (RowParallel base) → gather B on dim 0.
    if lora_b.shape[0] != base_shape[0]:
        lora_b = _all_gather_tp(lora_b, dim=0, is_expert=is_expert)
    # If A's input dim doesn't match base → gather A on dim 1.
    if lora_a.shape[1] != base_shape[1]:
        lora_a = _all_gather_tp(lora_a, dim=1, is_expert=is_expert)

    # Ensure tensors are valid before returning (catches freed-memory issues early)
    lora_a = lora_a.contiguous()
    lora_b = lora_b.contiguous()

    return lora_a, lora_b


def _get_safe_tensor(tensor):
    """Get a valid tensor, handling the case where GPU storage has been freed by torch_memory_saver.

    When the model is offloaded during colocate mode, torch_memory_saver frees
    the GPU storage but keeps a CPU backup. The tensor object still reports as
    CUDA but accessing its data causes an illegal memory error. This function
    retrieves the CPU backup when available.
    """
    try:
        from torch_memory_saver import torch_memory_saver
        cpu_backup = torch_memory_saver.get_cpu_backup(tensor)
        if cpu_backup is not None:
            return cpu_backup
    except ImportError:
        pass
    return tensor


def _all_gather_tp(tensor, dim=0, is_expert=False):
    """All-gather a tensor on the adapter's TP group along the given dimension."""
    from megatron.core import mpu

    if is_expert and hasattr(mpu, "get_expert_tensor_parallel_world_size"):
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    if tp_size <= 1:
        return tensor

    tensor = tensor.contiguous()
    partitions = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(partitions, tensor, group=tp_group)
    return torch.cat(partitions, dim=dim)


def _merge_base_weight_with_peft_entry(base_weight: torch.Tensor, entry):
    """Apply LoRA/DoRA merge entry to a base Megatron weight tensor."""
    merged = base_weight
    if isinstance(entry, tuple):
        # DoRA: entry is (delta, dora_column_scale)
        delta, dora_scale = entry
        merged = dora_scale.unsqueeze(-1) * (
            merged + delta.to(merged.device, dtype=merged.dtype)
        )
    else:
        # LoRA: entry is a bare delta tensor
        delta = entry
        merged = merged + delta.to(merged.device, dtype=merged.dtype)
    return merged


def _process_conversion_tasks(
    vanilla_conversion_tasks,
    new_weight_dict,
    lora_delta_map=None,
    ci_test=False,
    ci_peft_exact=False,
    prev_merged_by_task=None,
    sync_round=0,
    exact_state=None,
):
    """Process conversion tasks, optionally adding pre-computed LoRA deltas.

    Matches conversion tasks with pre-computed LoRA deltas by name: extracts
    the base module prefix from ``task.param_name`` (via the ``.to_wrap.weight``
    sentinel) and looks it up in the delta map.

    Args:
        vanilla_conversion_tasks: Original conversion tasks from bridge
        new_weight_dict: Dictionary of new weight tensors
        lora_delta_map: Map of global_base_prefix -> pre-computed delta tensors
        ci_test: Whether CI assertions are enabled
    """
    if lora_delta_map is None:
        lora_delta_map = {}

    merge_count = [0]
    expected_merge_count = len(lora_delta_map)
    expected_keys = set(lora_delta_map.keys())
    matched_keys = set()
    duplicate_key_count = [0]
    current_merged_by_task = {} if ci_peft_exact else None

    if exact_state is None:
        exact_state = {}

    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert weight_dict_key in new_weight_dict, f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        base_weight = new_weight_dict[weight_dict_key].cuda()
        new_param_weight = base_weight

        # Match by name: adapted layers have ".to_wrap.weight" in param_name.
        base_prefix = None
        if lora_delta_map and ".to_wrap.weight" in task.param_name:
            base_prefix = task.param_name.partition(".to_wrap.weight")[0]

        if base_prefix is not None and base_prefix in lora_delta_map:
            entry = lora_delta_map[base_prefix]
            new_param_weight = _merge_base_weight_with_peft_entry(new_param_weight, entry)
            merge_count[0] += 1
            if base_prefix in matched_keys:
                duplicate_key_count[0] += 1
            matched_keys.add(base_prefix)

            if ci_peft_exact:
                # Snapshot merged tensors on CPU so we can validate cross-sync deltas.
                current_merged_by_task[weight_dict_key] = new_param_weight.detach().cpu().clone()

        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLenAndMergeCount(
        _handle_one, vanilla_conversion_tasks, merge_count, log=bool(lora_delta_map),
        ci_test=ci_test, expected_merge_count=expected_merge_count,
        ci_peft_exact=ci_peft_exact,
        expected_keys=expected_keys,
        matched_keys=matched_keys,
        duplicate_key_count=duplicate_key_count,
        current_merged_by_task=current_merged_by_task,
        prev_merged_by_task=prev_merged_by_task,
        sync_round=sync_round,
        exact_state=exact_state,
    )


class _MapWithLenAndMergeCount:
    def __init__(
        self,
        fn,
        xs,
        merge_count,
        log=False,
        ci_test=False,
        expected_merge_count=0,
        ci_peft_exact=False,
        expected_keys=None,
        matched_keys=None,
        duplicate_key_count=None,
        current_merged_by_task=None,
        prev_merged_by_task=None,
        sync_round=0,
        exact_state=None,
    ):
        self.fn = fn
        self.xs = xs
        self.merge_count = merge_count
        self.log = log
        self.ci_test = ci_test
        self.expected_merge_count = expected_merge_count
        self.ci_peft_exact = ci_peft_exact
        self.expected_keys = expected_keys if expected_keys is not None else set()
        self.matched_keys = matched_keys if matched_keys is not None else set()
        self.duplicate_key_count = duplicate_key_count if duplicate_key_count is not None else [0]
        self.current_merged_by_task = current_merged_by_task if current_merged_by_task is not None else {}
        self.prev_merged_by_task = prev_merged_by_task
        self.sync_round = sync_round
        self.exact_state = exact_state if exact_state is not None else {}

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        self.merge_count[0] = 0
        for x in self.xs:
            yield self.fn(x)
        if self.log:
            print(f"LoRA weight sync: Merged {self.merge_count[0]} layers on-the-fly")
        if self.ci_test and self.log:
            from slime.backends.megatron_utils.ci_utils import check_peft_weight_merge

            check_peft_weight_merge(self.merge_count[0], len(self.xs), self.expected_merge_count)
        if self.ci_test and self.ci_peft_exact:
            from slime.backends.megatron_utils.ci_utils import check_peft_exact_weight_sync

            check_peft_exact_weight_sync(
                sync_round=self.sync_round,
                expected_merge_count=self.expected_merge_count,
                expected_keys=self.expected_keys,
                matched_keys=self.matched_keys,
                duplicate_key_count=self.duplicate_key_count[0],
                current_merged_by_task=self.current_merged_by_task,
                prev_merged_by_task=self.prev_merged_by_task,
            )
        if self.ci_peft_exact:
            self.exact_state["current_merged_by_task"] = self.current_merged_by_task
