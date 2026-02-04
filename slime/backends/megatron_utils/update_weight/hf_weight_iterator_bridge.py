import dataclasses

import torch

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
        self._lora_module_map = None  # Lazily built map of param_name -> LoRALinear module

    def _build_lora_module_map(self):
        """Build a map from parameter names to their LoRA adapter wrapper modules.

        This enables on-the-fly merging of LoRA weights during export without
        permanently modifying the model structure.

        Two adapter types are supported:
        1. LinearAdapter (for simple nn.Linear):
           - adapter.weight: base weight (frozen)
           - adapter.lora_a.weight: A matrix (trainable)
           - adapter.lora_b.weight: B matrix (trainable)
           - adapter.scale: alpha / rank

        2. ParallelLinearAdapter (for Megatron parallel linear):
           - Uses the base weight from the wrapped module
           - adapter.linear_in: A matrix (RowParallelLinear, trainable)
           - adapter.linear_out: B matrix (ColumnParallelLinear, trainable)
           - adapter.alpha, adapter.dim for scaling
        """
        if self._lora_module_map is not None:
            return

        if getattr(self.args, "peft_type", "none") == "none":
            self._lora_module_map = {}
            return

        try:
            from megatron.bridge.peft.lora import LinearAdapter, ParallelLinearAdapter
        except ImportError:
            print("Warning: LinearAdapter/ParallelLinearAdapter not available, skipping LoRA merge")
            self._lora_module_map = {}
            return

        self._lora_module_map = {}

        # Handle model being a list (pipeline parallelism / virtual pipeline)
        # or a single model wrapped in various ways
        models = self.model if isinstance(self.model, (list, tuple)) else [self.model]

        linear_adapter_count = 0
        parallel_adapter_count = 0

        for model in models:
            for name, module in model.named_modules():
                if isinstance(module, LinearAdapter):
                    # Normalize the key to match task.param_name format
                    base_param_name = self._normalize_lora_key(name)
                    self._lora_module_map[base_param_name] = ("linear", module)
                    linear_adapter_count += 1
                elif isinstance(module, ParallelLinearAdapter):
                    # Normalize the key to match task.param_name format
                    base_param_name = self._normalize_lora_key(name)
                    self._lora_module_map[base_param_name] = ("parallel", module)
                    parallel_adapter_count += 1

        if self._lora_module_map:
            print(f"LoRA weight sync: Found {linear_adapter_count} LinearAdapter + {parallel_adapter_count} ParallelLinearAdapter layers")

    def _normalize_lora_key(self, module_name: str) -> str:
        """Normalize module name to match task.param_name format.

        Transforms: module.module.decoder.layers.0.self_attention.linear_qkv.adapter
        To:         decoder.layers.0.self_attention.linear_qkv.weight
        """
        key = module_name
        # Strip common prefixes (module.module., module.)
        while key.startswith("module."):
            key = key[7:]  # len("module.") = 7
        # Strip .adapter suffix (ParallelLinearAdapter replaces the original layer)
        if key.endswith(".adapter"):
            key = key[:-8]  # len(".adapter") = 8
        # Add .weight suffix to match task.param_name format
        return f"{key}.weight"

    def get_hf_weight_chunks(self, megatron_local_weights):
        # Build LoRA module map for on-the-fly merging (lazy, only once)
        self._build_lora_module_map()

        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(
                conversion_tasks, renamed_megatron_local_weights, self._lora_module_map
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


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict, lora_module_map=None):
    """Process conversion tasks, optionally merging LoRA weights on-the-fly.

    For LoRA layers, computes: merged_weight = base_weight + B @ A * (alpha / rank)
    This enables weight sync to SGLang without permanently modifying the model.

    Args:
        vanilla_conversion_tasks: Original conversion tasks from bridge
        new_weight_dict: Dictionary of new weight tensors
        lora_module_map: Map of param_name -> (adapter_type, module) tuples
            adapter_type is "linear" for LinearAdapter or "parallel" for ParallelLinearAdapter
    """
    if lora_module_map is None:
        lora_module_map = {}

    # Build alternate lookup keys for LoRA modules
    # For adapters, the base weight path is "{module_name}.weight"
    # task.param_name should match this format
    lora_lookup = {}
    for key, (adapter_type, module) in lora_module_map.items():
        lora_lookup[key] = (adapter_type, module)
        # Also add without ".weight" suffix for flexibility
        if key.endswith(".weight"):
            lora_lookup[key[:-7]] = (adapter_type, module)

    _merge_count = [0]  # Track number of LoRA merges per sync
    _logged_first = [False]  # Use list to allow mutation in closure

    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert weight_dict_key in new_weight_dict, f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()

        # Debug: log first param name to help debug key matching
        if lora_module_map and not _logged_first[0]:
            sample_lora_keys = list(lora_module_map.keys())[:3]
            sample_lookup_keys = list(lora_lookup.keys())[:6]
            print(f"LoRA merge debug: task.param_name={task.param_name}")
            print(f"  sample lora_module_map keys: {sample_lora_keys}")
            print(f"  sample lora_lookup keys: {sample_lookup_keys}")
            _logged_first[0] = True

        # Check if this is a LoRA layer that needs on-the-fly merging
        if task.param_name in lora_lookup:
            adapter_type, lora_module = lora_lookup[task.param_name]
            new_param_weight = _merge_lora_weight(new_param_weight, lora_module, adapter_type)
            _merge_count[0] += 1
            # Log first merge to confirm key matching works
            if _merge_count[0] == 1:
                print(f"LoRA merge: First merge for {task.param_name} ({adapter_type})")

        return dataclasses.replace(task, param_weight=new_param_weight)

    class _MapWithLenAndLogging(_MapWithLen):
        """MapWithLen that logs merge count after iteration completes."""
        def __iter__(self):
            _merge_count[0] = 0
            for x in self.xs:
                yield self.fn(x)
            if lora_module_map:
                print(f"LoRA weight sync: Merged {_merge_count[0]} layers on-the-fly (expected ~{len(lora_module_map)})")

    return _MapWithLenAndLogging(_handle_one, vanilla_conversion_tasks)


@torch.no_grad()
def _merge_lora_weight(base_weight, lora_module, adapter_type):
    """Compute merged weight for a LoRA adapter without modifying the module.

    Supports two adapter types:

    1. LinearAdapter (adapter_type="linear"):
       - lora_module.lora_a.weight: A matrix (rank x in_features)
       - lora_module.lora_b.weight: B matrix (out_features x rank)
       - lora_module.scale: alpha / rank

    2. ParallelLinearAdapter (adapter_type="parallel"):
       - lora_module.linear_in.weight: A matrix (RowParallelLinear)
       - lora_module.linear_out.weight: B matrix (ColumnParallelLinear)
       - lora_module.alpha / lora_module.dim for scaling

    Args:
        base_weight: The base model weight tensor (from megatron_local_weights)
        lora_module: The adapter module containing LoRA weights
        adapter_type: "linear" for LinearAdapter, "parallel" for ParallelLinearAdapter

    Returns:
        Merged weight: base + B @ A * scale
    """
    base_device = base_weight.device
    base_dtype = base_weight.dtype

    if adapter_type == "linear":
        # LinearAdapter: uses lora_a and lora_b sub-modules
        lora_a = lora_module.lora_a.weight.to(base_device, dtype=base_dtype)  # (rank, in_features)
        lora_b = lora_module.lora_b.weight.to(base_device, dtype=base_dtype)  # (out_features, rank)
        scale = lora_module.scale  # alpha / rank
    elif adapter_type == "parallel":
        # ParallelLinearAdapter: uses linear_in and linear_out sub-modules
        lora_a = lora_module.linear_in.weight.to(base_device, dtype=base_dtype)  # (rank, in_features)
        lora_b = lora_module.linear_out.weight.to(base_device, dtype=base_dtype)  # (out_features, rank)
        scale = lora_module.alpha / lora_module.dim
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    # LoRA merge: W_merged = W_base + B @ A * scale
    merged_weight = base_weight + (lora_b @ lora_a) * scale

    return merged_weight


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
