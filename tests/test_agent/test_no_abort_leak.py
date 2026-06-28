"""A finished-but-unusable episode must ship a *masked reward-0* sample, never a
bare ``ABORTED``.

Regression for the rollout-0 crash on ``w_qwen3_6_swe_rebench_v2_noncolocate_3n``:
a Modal sandbox that died before the first LLM call produced an episode with zero
usable chains. ``generate._build_samples`` returned a bare ``Status.ABORTED`` whose
``reward`` stayed ``None`` (the dataclass default). ``generate_rollout_async`` never
recycles ABORTED -- it appends every finished group straight to the training batch --
so the ``None`` reached ``_post_process_rewards`` and crashed the step with
``torch.tensor(raw_rewards)`` -> "must be real number, not NoneType" (observed: 2/256
samples in the rollout_0 dump). The fix routes that path through ``_ship_null``
(reward 0, fully masked, ``remove_sample=True``), matching the eval path and keeping
the group at ``n_samples_per_prompt`` (slime's GRPO reshape requires it).

Importing ``agentic_rl.generate`` drags in the whole slime/SGLang training stack; on a
CPU env we stub exactly the modules that aren't installed (never the ones that are), so
this runs under CI and locally and skips cleanly if the import still can't be satisfied.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _import_with_stubs(modname: str):
    """Import ``modname``, stubbing each *missing* dependency (permissively) and
    retrying. Only modules that fail to import get stubbed, so a full GPU/CI env uses
    the real ones untouched."""
    for _ in range(60):
        try:
            return importlib.import_module(modname)
        except ImportError as exc:
            missing = exc.name
            if not missing or missing in sys.modules:
                raise
            stub = types.ModuleType(missing)
            stub.__path__ = []  # treat as a package so submodule imports resolve
            stub.__getattr__ = lambda _name: MagicMock()  # noqa: B023 - permissive attrs
            sys.modules[missing] = stub
    return importlib.import_module(modname)


try:
    gen = _import_with_stubs("agentic_rl.generate")
    from slime.utils.types import Sample
except Exception as exc:  # pragma: no cover - unsatisfiable import env
    pytest.skip(f"agentic_rl.generate unimportable: {exc}", allow_module_level=True)


class _FakeTok:
    eos_token_id = 7

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=False):
        return "resp"


class _Model:
    """Duck-typed RecordingModel. Default = episode whose sandbox died before any LLM
    call: no chains, so no usable response."""

    def __init__(self, chains=None, exit_status="ImageUnusable"):
        self.chains = chains or []
        self.exit_status = exit_status
        self.cached_tokens = self.input_tokens = self.n_format_errors = 0
        self.gen_time = 0.0


_MD = {"instance_id": "pandas-dev__pandas-57665", "problem_statement": "fix it"}


def _sample() -> "Sample":
    s = Sample(index=5, prompt="p", label="pandas-dev__pandas-57665")
    s.group_index = 0
    s.metadata = dict(_MD)
    return s


@pytest.mark.parametrize("evaluation", [False, True])
def test_unusable_episode_ships_masked_reward0_not_aborted(evaluation):
    out = gen._build_samples(
        _sample(), _Model(), None, _FakeTok(), _MD, SimpleNamespace(),
        elapsed=1.0, evaluation=evaluation,
    )
    assert isinstance(out, Sample)
    assert out.status == Sample.Status.COMPLETED  # was Status.ABORTED
    assert out.reward == 0.0 and out.reward is not None  # was None -> torch.tensor crash
    assert out.remove_sample is True  # contributes no gradient
    assert out.response_length == 1
    assert out.metadata["agentic"]["exit_status"] == "ImageUnusable"


def test_usable_episode_still_trains_normally():
    """Guard the normal path: a real chain still trains with its real reward."""
    from agentic_rl.environment.base import RewardResult
    from agentic_rl.model import Chain

    chain = Chain(
        tokens=[1, 2, 3, 4],
        loss_mask=[0, 0, 1, 1],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        versions=["v0"],
        prompt_len=2,
        full_prompt="sys+inst",
    )
    out = gen._build_samples(
        _sample(), _Model(chains=[chain]), RewardResult(reward=1.0, is_solved=True, extra={}),
        _FakeTok(), _MD, SimpleNamespace(), elapsed=1.0, evaluation=False,
    )
    s0 = out[0] if isinstance(out, list) else out
    assert s0.status == Sample.Status.COMPLETED
    assert s0.reward == 1.0
    assert not getattr(s0, "remove_sample", False)
