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

        self._lora_tensor_map = {}

        # Handle model being a list (pipeline parallelism / virtual pipeline)
        models = self.model if isinstance(self.model, (list, tuple)) else [self.model]

        linear_adapter_count = 0
        parallel_adapter_count = 0

        for model in models:
            for name, module in model.named_modules():
                if isinstance(module, LinearAdapter):
                    # LinearAdapter.weight IS the base weight tensor
                    self._lora_tensor_map[id(module.weight)] = ("linear", module, module.weight.shape)
                    linear_adapter_count += 1
                    continue

                # The PEFT transform wraps each target linear module with a LoRALinear
                # that has two children:
                #   - "to_wrap": the original linear module (holds the base weight)
                #   - "adapter": the ParallelLinearAdapter (holds LoRA A/B matrices)
                # The bridge's conversion tasks reference to_wrap.weight, so we use
                # its tensor identity as the map key.
                adapter_child = None
                base_weight_child = None
                for child_name, child in module.named_children():
                    if isinstance(child, ParallelLinearAdapter):
                        adapter_child = child
                    elif hasattr(child, "weight") and child.weight is not None:
                        base_weight_child = child

                if adapter_child is not None and base_weight_child is not None:
                    base_shape = base_weight_child.weight.shape
                    self._lora_tensor_map[id(base_weight_child.weight)] = ("parallel", adapter_child, base_shape)
                    parallel_adapter_count += 1

        if self._lora_tensor_map:
            print(f"LoRA weight sync: Found {linear_adapter_count} LinearAdapter + {parallel_adapter_count} ParallelLinearAdapter layers")
        elif peft_type != "none":
            print(f"LoRA weight sync WARNING: peft_type={peft_type} but no adapter tensor mappings found")

    def get_hf_weight_chunks(self, megatron_local_weights):
        # Build LoRA tensor map for on-the-fly merging (lazy, only once)
        self._build_lora_tensor_map()

        # Pre-compute LoRA deltas BEFORE entering the bridge's export flow.
        # This is necessary because the bridge uses NCCL internally for TP gathering,
        # and we cannot interleave our own all_gather calls inside the per-task iteration.
        lora_delta_map = _precompute_lora_deltas(self._lora_tensor_map)

        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(
                conversion_tasks, renamed_megatron_local_weights, lora_delta_map
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


@torch.no_grad()
def _precompute_lora_deltas(lora_tensor_map):
    """Pre-compute LoRA weight deltas for all adapters.

    For each adapter, computes: delta = B @ A * scale

    For ParallelLinearAdapter with tensor parallelism, the LoRA matrices may be
    sharded across TP ranks. We selectively gather only the dimensions needed so
    that the resulting delta matches the per-rank base weight shape directly:

      - ColumnParallel base (linear_qkv, linear_fc1): A has rank dim sharded on
        dim 0, B has output dim sharded on dim 0. Gathering A on dim 0 fixes the
        matmul inner dims, producing delta (out/TP, in) matching the base.
      - RowParallel base (linear_proj, linear_fc2): inner dims already match.
        Gathering B on dim 0 produces delta (out, in/TP) matching the base.

    The base weight shape (stored in the tensor map) determines which gathers are
    needed. This must happen BEFORE the bridge's export_hf_weights flow to avoid
    NCCL collective conflicts with the bridge's internal TP communication.

    Note: The model may be offloaded to CPU via torch_memory_saver at this point,
    so we must use _get_safe_tensor() to access weights from CPU backups.

    Returns:
        dict mapping id(base_weight_tensor) -> pre-computed delta tensor
    """
    if not lora_tensor_map:
        return {}

    delta_map = {}

    for tensor_id, (adapter_type, lora_module, base_shape) in lora_tensor_map.items():
        if adapter_type == "linear":
            lora_a = _get_safe_tensor(lora_module.lora_a.weight).cuda()
            lora_b = _get_safe_tensor(lora_module.lora_b.weight).cuda()
            scale = lora_module.scale
        elif adapter_type == "parallel":
            lora_a = _get_safe_tensor(lora_module.linear_in.weight).cuda()
            lora_b = _get_safe_tensor(lora_module.linear_out.weight).cuda()
            scale = lora_module.alpha / lora_module.dim

            # Selectively gather LoRA matrices so that B @ A produces a delta
            # matching the per-rank base weight shape.
            # Fix inner dims first: if B.shape[1] != A.shape[0], A's rank dim
            # is sharded (ColumnParallel base) → gather A on dim 0.
            if lora_b.shape[1] != lora_a.shape[0]:
                lora_a = _all_gather_tp(lora_a, dim=0)
            # Fix outer dims: if B.shape[0] != base_shape[0], B's output dim
            # is sharded (RowParallel base) → gather B on dim 0.
            if lora_b.shape[0] != base_shape[0]:
                lora_b = _all_gather_tp(lora_b, dim=0)
            # If A's input dim doesn't match base → gather A on dim 1.
            if lora_a.shape[1] != base_shape[1]:
                lora_a = _all_gather_tp(lora_a, dim=1)
        else:
            continue

        delta_map[tensor_id] = (lora_b @ lora_a) * scale

    print(f"LoRA weight sync: Pre-computed {len(delta_map)} merge deltas")
    return delta_map


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


def _all_gather_tp(tensor, dim=0):
    """All-gather a tensor across the tensor-model-parallel group along the given dimension."""
    from megatron.core import mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return tensor

    tensor = tensor.contiguous()
    partitions = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(partitions, tensor, group=mpu.get_tensor_model_parallel_group())
    return torch.cat(partitions, dim=dim)


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict, lora_delta_map=None):
    """Process conversion tasks, optionally adding pre-computed LoRA deltas.

    Uses tensor identity (id(task.param_weight)) to match conversion tasks with
    pre-computed LoRA deltas, avoiding fragile parameter name normalization.

    Args:
        vanilla_conversion_tasks: Original conversion tasks from bridge
        new_weight_dict: Dictionary of new weight tensors
        lora_delta_map: Map of id(param_tensor) -> pre-computed delta tensors
    """
    if lora_delta_map is None:
        lora_delta_map = {}

    _merge_count = [0]

    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert weight_dict_key in new_weight_dict, f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()

        # Add pre-computed LoRA delta if this is an adapted layer
        if id(task.param_weight) in lora_delta_map:
            delta = lora_delta_map[id(task.param_weight)]
            new_param_weight = new_param_weight + delta.to(new_param_weight.device, dtype=new_param_weight.dtype)
            _merge_count[0] += 1
            if _merge_count[0] == 1:
                print(f"LoRA merge: First merge for {task.param_name}")

        return dataclasses.replace(task, param_weight=new_param_weight)

    class _MapWithLenAndLogging(_MapWithLen):
        """MapWithLen that logs merge count after iteration completes."""
        def __iter__(self):
            _merge_count[0] = 0
            for x in self.xs:
                yield self.fn(x)
            if lora_delta_map:
                print(f"LoRA weight sync: Merged {_merge_count[0]} layers on-the-fly")

    return _MapWithLenAndLogging(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
