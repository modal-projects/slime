"""Router params must be exported in the model dtype, not their fp32 training dtype.

With ``--moe-router-dtype fp32`` Megatron keeps the MoE router weight / expert_bias in fp32 even
when the model is bf16/fp16. ``update_weight_from_disk_delta`` XORs every freshly converted HF
tensor against the raw bytes of the base HF checkpoint (which stores the router in the model
dtype), so a leftover fp32 router is a 4-vs-2 byte-width mismatch that breaks the delta. Every
converter that emits a router must cast it back to the model dtype. This is pure tensor-metadata
work (no model load, no I/O), so the whole module runs in milliseconds.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0

CONVERTER_DIR = (
    Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "megatron_to_hf"
)

# Synthetic parent package so the converters' ``from .dtype_utils import ...`` relative import
# resolves to the real dtype_utils.py without running the package __init__ (which imports
# megatron/mbridge and is unavailable in the unit-test env).
_PKG = "_mhf_router_dtype_test_pkg"

# (converter module, converter function, router param names that must be cast).
# gpt-oss has no expert_bias; its second router buffer is mlp.router.bias.
CONVERTERS = [
    ("deepseekv3", "convert_deepseekv3_to_hf", ["mlp.router.weight", "mlp.router.expert_bias"]),
    ("glm4moe", "convert_glm4moe_to_hf", ["mlp.router.weight", "mlp.router.expert_bias"]),
    ("minimax_m2", "convert_minimax_m2_to_hf", ["mlp.router.weight", "mlp.router.expert_bias"]),
    ("qwen3moe", "convert_qwen3moe_to_hf", ["mlp.router.weight", "mlp.router.expert_bias"]),
    ("qwen3_5", "convert_qwen3_5_to_hf", ["mlp.router.weight", "mlp.router.expert_bias"]),
    ("qwen3_next", "convert_qwen3_next_to_hf", ["mlp.router.weight", "mlp.router.expert_bias"]),
    ("gpt_oss", "convert_gpt_oss_to_hf", ["mlp.router.weight", "mlp.router.bias"]),
]


def _load_converter(module_name, func_name):
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(CONVERTER_DIR)]
        sys.modules[_PKG] = pkg
    full_name = f"{_PKG}.{module_name}"
    sys.modules.pop(full_name, None)
    spec = importlib.util.spec_from_file_location(full_name, CONVERTER_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[full_name] = module  # let the relative import find the in-progress module
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def _args(**overrides):
    # Fields every converter dereferences before reaching the router branch (head_dim /
    # value_num_per_group). q_lora_rank is read in the package wrapper, not the converter itself.
    base = dict(
        kv_channels=None,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=4,
        num_layers=2,
        bf16=False,
        fp16=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _cases():
    for module_name, func_name, router_params in CONVERTERS:
        for router_param in router_params:
            for flag, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
                yield pytest.param(
                    module_name, func_name, router_param, flag, dtype,
                    id=f"{module_name}-{router_param.split('.')[-1]}-{flag}",
                )


@pytest.mark.unit
@pytest.mark.parametrize("module_name,func_name,router_param,flag,model_dtype", list(_cases()))
def test_router_param_cast_to_model_dtype(module_name, func_name, router_param, flag, model_dtype):
    convert = _load_converter(module_name, func_name)
    args = _args(**{flag: True})

    name = f"module.module.decoder.layers.0.{router_param}"
    fp32_param = torch.randn(8, 16, dtype=torch.float32)

    converted = convert(args, name, fp32_param)

    assert len(converted) == 1, f"{func_name} should map {router_param} to a single HF tensor"
    _, out = converted[0]
    # A fp32 router would be 2x the bytes of a bf16/fp16 base, breaking the disk-delta XOR.
    assert out.dtype == model_dtype


@pytest.mark.unit
@pytest.mark.parametrize("module_name,func_name,router_param,_flag,_dtype", list(_cases()))
def test_router_param_preserved_in_full_precision(module_name, func_name, router_param, _flag, _dtype):
    # No bf16/fp16 flag => no model dtype to match; leave the param untouched.
    convert = _load_converter(module_name, func_name)
    args = _args()

    name = f"module.module.decoder.layers.0.{router_param}"
    fp32_param = torch.randn(8, 16, dtype=torch.float32)

    _, out = convert(args, name, fp32_param)[0]

    assert out.dtype == torch.float32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
