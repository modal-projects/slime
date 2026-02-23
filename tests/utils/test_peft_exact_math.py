import dataclasses
import sys
import types

import pytest
import torch

from slime.backends.megatron_utils.ci_utils import check_peft_exact_weight_sync
from slime.backends.megatron_utils.update_weight import hf_weight_iterator_bridge as bridge_mod


def _reference_lora_delta(lora_b: torch.Tensor, lora_a: torch.Tensor, scale: float) -> torch.Tensor:
    return ((lora_b.float() @ lora_a.float()) * scale).to(dtype=lora_b.dtype)


class _CudaWrapper:
    """Wrap a CPU tensor with a .cuda() method so tests can run without GPUs."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def cuda(self):
        return self.tensor


def _maybe_cuda_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def _tensor_or_cuda_wrapper(tensor: torch.Tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return _CudaWrapper(tensor)


@dataclasses.dataclass
class _FakeTask:
    vp_stage: int
    param_name: str
    param_weight: object


def test_compute_lora_delta_matches_reference():
    torch.manual_seed(0)
    lora_b = _maybe_cuda_tensor(torch.randn(5, 3, dtype=torch.float32))
    lora_a = _maybe_cuda_tensor(torch.randn(3, 7, dtype=torch.float32))
    scale = 1.75

    expected = _reference_lora_delta(lora_b, lora_a, scale)
    actual = bridge_mod._compute_lora_delta(lora_b, lora_a, scale)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_merge_base_weight_with_dora_entry_matches_formula():
    torch.manual_seed(0)
    base = _maybe_cuda_tensor(torch.randn(4, 6, dtype=torch.float32))
    delta = _maybe_cuda_tensor(torch.randn(4, 6, dtype=torch.float32))
    dora_scale = _maybe_cuda_tensor(torch.rand(4, dtype=torch.float32) + 0.1)

    expected = dora_scale.unsqueeze(-1) * (base + delta)
    actual = bridge_mod._merge_base_weight_with_peft_entry(base, (delta, dora_scale))
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("lora_a_shape", "lora_b_shape", "base_shape", "is_expert", "expected_gather_dims", "expected_shapes"),
    [
        # Column-parallel style: gather A on dim 0.
        ((1, 8), (4, 2), (4, 8), False, [0], ((2, 8), (4, 2))),
        # Row-parallel style: gather B on dim 0.
        ((2, 4), (2, 2), (4, 4), False, [0], ((2, 4), (4, 2))),
        # Input mismatch: gather A on dim 1.
        ((2, 2), (4, 2), (4, 4), False, [1], ((2, 4), (4, 2))),
        # Expert-parallel adapter should route gathers via expert TP.
        ((2, 2), (4, 2), (4, 4), True, [1], ((2, 4), (4, 2))),
    ],
)
def test_gather_parallel_lora_ab_uses_expected_gather_dims(
    monkeypatch,
    lora_a_shape,
    lora_b_shape,
    base_shape,
    is_expert,
    expected_gather_dims,
    expected_shapes,
):
    calls = []

    def _fake_all_gather_tp(tensor, dim=0, is_expert=False):
        calls.append((dim, is_expert))
        return torch.cat([tensor, tensor], dim=dim)

    monkeypatch.setattr(bridge_mod, "_all_gather_tp", _fake_all_gather_tp)

    lora_a = _maybe_cuda_tensor(torch.randn(*lora_a_shape))
    lora_b = _maybe_cuda_tensor(torch.randn(*lora_b_shape))

    gathered_a, gathered_b = bridge_mod._gather_parallel_lora_ab(lora_a, lora_b, torch.Size(base_shape), is_expert)
    assert calls == [(dim, is_expert) for dim in expected_gather_dims]
    assert gathered_a.shape == torch.Size(expected_shapes[0])
    assert gathered_b.shape == torch.Size(expected_shapes[1])


def test_all_gather_tp_uses_expert_tp_group_when_requested(monkeypatch):
    calls = []

    class _FakeMPU:
        @staticmethod
        def get_tensor_model_parallel_world_size():
            return 2

        @staticmethod
        def get_expert_tensor_parallel_world_size():
            return 2

        @staticmethod
        def get_tensor_model_parallel_group():
            return "tp_group"

        @staticmethod
        def get_expert_tensor_parallel_group():
            return "expert_tp_group"

    def _fake_all_gather(partitions, tensor, group):
        calls.append(group)
        for i in range(len(partitions)):
            partitions[i].copy_(tensor)

    fake_core_mod = types.ModuleType("megatron.core")
    fake_core_mod.mpu = _FakeMPU()
    fake_megatron_mod = types.ModuleType("megatron")
    fake_megatron_mod.core = fake_core_mod

    monkeypatch.setitem(sys.modules, "megatron", fake_megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", fake_core_mod)
    monkeypatch.setattr(bridge_mod.dist, "all_gather", _fake_all_gather)

    x = torch.randn(2, 3, dtype=torch.float32)
    gathered = bridge_mod._all_gather_tp(x, dim=0, is_expert=True)
    assert gathered.shape == torch.Size((4, 3))
    assert calls == ["expert_tp_group"]


def test_process_conversion_tasks_exact_tracks_expected_keys_across_syncs():
    adapted_param = object()
    other_param = object()
    tasks = [
        _FakeTask(
            vp_stage=0,
            param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
            param_weight=adapted_param,
        ),
        _FakeTask(
            vp_stage=0,
            param_name="decoder.layers.0.self_attention.linear_qkv.weight",
            param_weight=other_param,
        ),
    ]

    base0 = torch.full((2, 3), 1.0, dtype=torch.float32)
    base1 = torch.full((2, 3), 5.0, dtype=torch.float32)
    new_weight_dict = {
        "vp_stages.0.decoder.layers.0.mlp.linear_fc1.to_wrap.weight": _tensor_or_cuda_wrapper(base0),
        "vp_stages.0.decoder.layers.0.self_attention.linear_qkv.weight": _tensor_or_cuda_wrapper(base1),
    }

    delta_round0 = _maybe_cuda_tensor(torch.full((2, 3), 0.25, dtype=torch.float32))
    delta_round1 = _maybe_cuda_tensor(torch.full((2, 3), 0.75, dtype=torch.float32))

    # Delta map keyed by base module name (name-based matching)
    exact_state0 = {}
    iterator0 = bridge_mod._process_conversion_tasks(
        tasks,
        new_weight_dict,
        lora_delta_map={"decoder.layers.0.mlp.linear_fc1": delta_round0},
        ci_test=True,
        ci_peft_exact=True,
        prev_merged_by_task=None,
        sync_round=0,
        exact_state=exact_state0,
    )
    output_round0 = list(iterator0)
    adapted_round0 = next(
        x for x in output_round0
        if x.param_name == "decoder.layers.0.mlp.linear_fc1.to_wrap.weight"
    ).param_weight
    assert torch.allclose(adapted_round0, _maybe_cuda_tensor(base0) + delta_round0)
    assert set(exact_state0["current_merged_by_task"]) == {
        "vp_stages.0.decoder.layers.0.mlp.linear_fc1.to_wrap.weight"
    }

    exact_state1 = {}
    iterator1 = bridge_mod._process_conversion_tasks(
        tasks,
        new_weight_dict,
        lora_delta_map={"decoder.layers.0.mlp.linear_fc1": delta_round1},
        ci_test=True,
        ci_peft_exact=True,
        prev_merged_by_task=exact_state0["current_merged_by_task"],
        sync_round=1,
        exact_state=exact_state1,
    )
    output_round1 = list(iterator1)
    adapted_round1 = next(
        x for x in output_round1
        if x.param_name == "decoder.layers.0.mlp.linear_fc1.to_wrap.weight"
    ).param_weight
    assert torch.allclose(adapted_round1, _maybe_cuda_tensor(base0) + delta_round1)
    assert set(exact_state1["current_merged_by_task"]) == {
        "vp_stages.0.decoder.layers.0.mlp.linear_fc1.to_wrap.weight"
    }


def test_check_peft_exact_weight_sync_rejects_unchanged_sync():
    merged = {"vp_stages.0.layer0.weight": _maybe_cuda_tensor(torch.ones(2, 2))}
    with pytest.raises(AssertionError, match="No adapted merged layer changed"):
        check_peft_exact_weight_sync(
            sync_round=1,
            expected_merge_count=1,
            expected_keys={"decoder.layers.0.mlp.linear_fc1"},
            matched_keys={"decoder.layers.0.mlp.linear_fc1"},
            duplicate_key_count=0,
            current_merged_by_task=merged,
            prev_merged_by_task={"vp_stages.0.layer0.weight": _maybe_cuda_tensor(torch.ones(2, 2))},
        )
