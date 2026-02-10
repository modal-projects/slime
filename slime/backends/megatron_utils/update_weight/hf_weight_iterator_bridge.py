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
        self._lora_tensor_map = None  # Lazily built map of id(tensor) -> (adapter_type, module)
        self._ci_prev_merged_by_task = None
        self._ci_sync_round = 0

    def _build_lora_tensor_map(self):
        """Build a map from base weight tensor identity to LoRA adapter modules.

        Uses id(tensor) for matching instead of parameter name normalization,
        making it immune to internal naming conventions of adapter wrappers.

        The PEFT transform wraps each target linear module with a LoRALinear that
        has two children:
          - "to_wrap": the original parallel linear module (holds the base weight)
          - "adapter": the ParallelLinearAdapter (holds LoRA A/B matrices)

        The bridge's conversion tasks reference the to_wrap module's weight tensor,
        so we find LoRALinear wrappers (modules with both a ParallelLinearAdapter
        child and a weight-bearing child) and map id(base_weight) -> adapter.
        """
        if self._lora_tensor_map is not None:
            return

        peft_type = getattr(self.args, "peft_type", "none")
        if peft_type == "none":
            self._lora_tensor_map = {}
            return

        try:
            from megatron.bridge.peft.lora import LinearAdapter, ParallelLinearAdapter
        except ImportError:
            print("Warning: LinearAdapter/ParallelLinearAdapter not available, skipping LoRA merge")
            self._lora_tensor_map = {}
            return

        try:
            from megatron.bridge.peft.dora_layers import ParallelLinearDoRAAdapter
        except ImportError:
            ParallelLinearDoRAAdapter = None

        self._lora_tensor_map = {}

        # Handle model being a list (pipeline parallelism / virtual pipeline)
        models = self.model if isinstance(self.model, (list, tuple)) else [self.model]

        linear_adapter_count = 0
        parallel_adapter_count = 0
        dora_adapter_count = 0

        for model in models:
            for name, module in model.named_modules():
                if isinstance(module, LinearAdapter):
                    # LinearAdapter.weight IS the base weight tensor
                    self._lora_tensor_map[id(module.weight)] = ("linear", module, module.weight.shape)
                    linear_adapter_count += 1
                    continue

                # The PEFT transform wraps each target linear module with a LoRALinear
                # (or DoRALinear) that has two children:
                #   - "to_wrap": the original linear module (holds the base weight)
                #   - "adapter": the ParallelLinearAdapter/DoRA adapter (holds LoRA A/B matrices)
                # The bridge's conversion tasks reference to_wrap.weight, so we use
                # its tensor identity as the map key.
                adapter_child = None
                base_weight_child = None
                for child_name, child in module.named_children():
                    # Check DoRA before LoRA since ParallelLinearDoRAAdapter
                    # is a subclass of ParallelLinearAdapter
                    if ParallelLinearDoRAAdapter is not None and isinstance(child, ParallelLinearDoRAAdapter):
                        adapter_child = child
                    elif isinstance(child, ParallelLinearAdapter):
                        adapter_child = child
                    elif hasattr(child, "weight") and child.weight is not None:
                        base_weight_child = child

                if adapter_child is not None and base_weight_child is not None:
                    base_shape = base_weight_child.weight.shape
                    if ParallelLinearDoRAAdapter is not None and isinstance(adapter_child, ParallelLinearDoRAAdapter):
                        self._lora_tensor_map[id(base_weight_child.weight)] = (
                            "dora", adapter_child, base_shape, base_weight_child.weight
                        )
                        dora_adapter_count += 1
                    else:
                        self._lora_tensor_map[id(base_weight_child.weight)] = ("parallel", adapter_child, base_shape)
                        parallel_adapter_count += 1

        if self._lora_tensor_map:
            counts = f"{linear_adapter_count} LinearAdapter + {parallel_adapter_count} ParallelLinearAdapter + {dora_adapter_count} DoRA"
            print(f"LoRA weight sync: Found {counts} layers")
        elif peft_type != "none":
            print(f"LoRA weight sync WARNING: peft_type={peft_type} but no adapter tensor mappings found")

    def get_hf_weight_chunks(self, megatron_local_weights):
        # Build LoRA tensor map for on-the-fly merging (lazy, only once)
        self._build_lora_tensor_map()

        # Pre-compute LoRA deltas BEFORE entering the bridge's export flow.
        # This is necessary because the bridge uses NCCL internally for TP gathering,
        # and we cannot interleave our own all_gather calls inside the per-task iteration.
        lora_delta_map = _precompute_lora_deltas(self._lora_tensor_map)
        ci_peft_exact = getattr(self.args, "ci_peft_exact", False)

        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        exact_state = {}
        with megatron_bridge_utils.patch_megatron_model(self.model):
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
def _precompute_lora_deltas(lora_tensor_map):
    """Pre-compute LoRA/DoRA weight deltas for all adapters.

    For each adapter, computes: delta = B @ A * scale

    For ParallelLinearAdapter with tensor parallelism, the LoRA matrices may be
    sharded across TP ranks. We selectively gather only the dimensions needed so
    that the resulting delta matches the per-rank base weight shape directly:

      - ColumnParallel base (linear_qkv, linear_fc1): A has rank dim sharded on
        dim 0, B has output dim sharded on dim 0. Gathering A on dim 0 fixes the
        matmul inner dims, producing delta (out/TP, in) matching the base.
      - RowParallel base (linear_proj, linear_fc2): inner dims already match.
        Gathering B on dim 0 produces delta (out, in/TP) matching the base.

    For DoRA adapters, additionally computes a per-row magnitude scale:
      dora_scale = weight_magnitude / ||base_weight + delta||_rows
    The merged weight is then: dora_scale * (base_weight + delta).

    The base weight shape (stored in the tensor map) determines which gathers are
    needed. This must happen BEFORE the bridge's export_hf_weights flow to avoid
    NCCL collective conflicts with the bridge's internal TP communication.

    Note: The model may be offloaded to CPU via torch_memory_saver at this point,
    so we must use _get_safe_tensor() to access weights from CPU backups.

    Returns:
        dict mapping id(base_weight_tensor) -> pre-computed delta tensor (LoRA)
              or (delta, dora_column_scale) tuple (DoRA)
    """
    if not lora_tensor_map:
        return {}

    delta_map = {}

    for tensor_id, entry in lora_tensor_map.items():
        adapter_type = entry[0]
        if adapter_type == "linear":
            _, lora_module, base_shape = entry
            lora_a = _get_safe_tensor(lora_module.lora_a.weight)
            lora_b = _get_safe_tensor(lora_module.lora_b.weight)
            scale = lora_module.scale
            delta_map[tensor_id] = _compute_lora_delta(lora_b, lora_a, scale)
        elif adapter_type == "parallel":
            _, lora_module, base_shape = entry
            lora_a, lora_b = _get_parallel_lora_ab(lora_module, base_shape)
            scale = lora_module.alpha / lora_module.dim
            delta_map[tensor_id] = _compute_lora_delta(lora_b, lora_a, scale)
        elif adapter_type == "dora":
            _, lora_module, base_shape, base_weight_ref = entry
            lora_a, lora_b = _get_parallel_lora_ab(lora_module, base_shape)
            scale = lora_module.alpha / lora_module.dim

            delta = _compute_lora_delta(lora_b, lora_a, scale)

            # Compute DoRA magnitude scaling: m / ||W_0 + delta||_rows
            #
            # The DoRA forward pass computes per-rank partial norms
            # (using local weight shard) and applies the scale before
            # all_reduce. Our merge must match this: each rank applies
            # its own dora_scale to its weight shard, and the bridge
            # then TP-gathers the scaled shards into the full HF weight.
            # Using per-rank partial norms for both magnitude and weight_norm
            # ensures the scale is consistent with the forward pass.
            base_weight = _get_safe_tensor(base_weight_ref).cuda()
            combined = base_weight + delta.to(base_weight.dtype)
            row_norms = torch.linalg.norm(combined.float(), dim=1)

            magnitude = _get_safe_tensor(lora_module.weight_magnitude).cuda()
            dora_scale = (magnitude / row_norms).to(delta.dtype)

            delta_map[tensor_id] = (delta, dora_scale)
        else:
            continue

    print(f"LoRA weight sync: Pre-computed {len(delta_map)} merge deltas")
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


def _get_parallel_lora_ab(lora_module, base_shape):
    """Extract and gather LoRA A/B matrices for a ParallelLinearAdapter.

    Selectively gathers LoRA matrices so that B @ A produces a delta
    matching the per-rank base weight shape.
    """
    lora_a = _get_safe_tensor(lora_module.linear_in.weight).cuda()
    lora_b = _get_safe_tensor(lora_module.linear_out.weight).cuda()
    is_expert = bool(getattr(lora_module, "is_expert", False))

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

    Uses tensor identity (id(task.param_weight)) to match conversion tasks with
    pre-computed LoRA deltas, avoiding fragile parameter name normalization.

    Args:
        vanilla_conversion_tasks: Original conversion tasks from bridge
        new_weight_dict: Dictionary of new weight tensors
        lora_delta_map: Map of id(param_tensor) -> pre-computed delta tensors
        ci_test: Whether CI assertions are enabled
    """
    if lora_delta_map is None:
        lora_delta_map = {}

    merge_count = [0]
    expected_merge_count = len(lora_delta_map)
    expected_tensor_ids = set(lora_delta_map.keys())
    matched_tensor_ids = set()
    duplicate_tensor_id_count = [0]
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

        # Add pre-computed LoRA/DoRA delta if this is an adapted layer
        if id(task.param_weight) in lora_delta_map:
            entry = lora_delta_map[id(task.param_weight)]
            new_param_weight = _merge_base_weight_with_peft_entry(new_param_weight, entry)
            merge_count[0] += 1
            if id(task.param_weight) in matched_tensor_ids:
                duplicate_tensor_id_count[0] += 1
            matched_tensor_ids.add(id(task.param_weight))

            if ci_peft_exact:
                # Snapshot merged tensors on CPU so we can validate cross-sync deltas.
                current_merged_by_task[weight_dict_key] = new_param_weight.detach().cpu().clone()

        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLenAndMergeCount(
        _handle_one, vanilla_conversion_tasks, merge_count, log=bool(lora_delta_map),
        ci_test=ci_test, expected_merge_count=expected_merge_count,
        ci_peft_exact=ci_peft_exact,
        expected_tensor_ids=expected_tensor_ids,
        matched_tensor_ids=matched_tensor_ids,
        duplicate_tensor_id_count=duplicate_tensor_id_count,
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
        expected_tensor_ids=None,
        matched_tensor_ids=None,
        duplicate_tensor_id_count=None,
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
        self.expected_tensor_ids = expected_tensor_ids if expected_tensor_ids is not None else set()
        self.matched_tensor_ids = matched_tensor_ids if matched_tensor_ids is not None else set()
        self.duplicate_tensor_id_count = duplicate_tensor_id_count if duplicate_tensor_id_count is not None else [0]
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
                expected_tensor_ids=self.expected_tensor_ids,
                matched_tensor_ids=self.matched_tensor_ids,
                duplicate_tensor_id_count=self.duplicate_tensor_id_count[0],
                current_merged_by_task=self.current_merged_by_task,
                prev_merged_by_task=self.prev_merged_by_task,
            )
        if self.ci_peft_exact:
            self.exact_state["current_merged_by_task"] = self.current_merged_by_task
