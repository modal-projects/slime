import asyncio
import os
from argparse import Namespace
from types import SimpleNamespace

import pytest

from slime.utils.url_utils import (
    get_external_engine_base_urls_from_args,
    get_model_url_from_args,
    join_url,
    parse_external_engine_addr,
)


def _modal_delta_sync():
    pytest.importorskip("modal")
    from examples.delta_weight_sync import modal_delta_sync

    return modal_delta_sync


def test_parse_external_engine_addr_accepts_https_url():
    addr = parse_external_engine_addr("https://rollout.example.modal.run/base/")

    assert addr.base_url == "https://rollout.example.modal.run/base"
    assert addr.host == "rollout.example.modal.run"
    assert addr.port == 443
    assert addr.dist_init_addr == "rollout.example.modal.run:443"
    assert addr.is_url is True
    assert join_url(addr.base_url, "/health_generate") == "https://rollout.example.modal.run/base/health_generate"


def test_parse_external_engine_addr_accepts_host_port():
    addr = parse_external_engine_addr("127.0.0.1:8000")

    assert addr.base_url == "http://127.0.0.1:8000"
    assert addr.host == "127.0.0.1"
    assert addr.port == 8000
    assert addr.dist_init_addr == "127.0.0.1:8000"
    assert addr.is_url is False


def test_external_engine_base_urls_from_args_normalizes_admin_urls():
    args = Namespace(
        rollout_external_engine_addrs=[
            "https://rollout.example.modal.run/base/",
            "127.0.0.1:8000",
        ]
    )

    assert get_external_engine_base_urls_from_args(args) == [
        "https://rollout.example.modal.run/base",
        "http://127.0.0.1:8000",
    ]


def test_rollout_router_url_takes_generation_precedence():
    args = Namespace(
        rollout_router_url="https://rollout.example.modal.run",
        sglang_router_ip="10.0.0.1",
        sglang_router_port=3000,
    )

    assert get_model_url_from_args(args, "default") == "https://rollout.example.modal.run/generate"


def test_named_router_may_be_full_url():
    args = Namespace(
        sglang_router_ip="10.0.0.1",
        sglang_router_port=3000,
        sglang_model_routers={"actor": "https://actor.example.modal.run/base"},
    )

    assert get_model_url_from_args(args, "actor") == "https://actor.example.modal.run/base/generate"


def test_validate_delta_update_payload_accepts_version_dir(tmp_path):
    modal_delta_sync = _modal_delta_sync()
    version_dir = tmp_path / "weight_v000123"
    version_dir.mkdir()

    validated = modal_delta_sync.validate_delta_update_payload(
        {"load_format": "delta", "model_path": str(version_dir)},
        delta_mount_path=str(tmp_path),
    )

    assert validated == os.path.realpath(version_dir)


@pytest.mark.parametrize(
    "payload",
    [
        {"load_format": "full", "model_path": "/delta/weight_v000123"},
        {"load_format": "delta"},
        {"load_format": "delta", "model_path": "/delta/not_a_version"},
        {"load_format": "delta", "model_path": "/tmp/weight_v000123"},
    ],
)
def test_validate_delta_update_payload_rejects_bad_paths(payload, tmp_path):
    modal_delta_sync = _modal_delta_sync()
    with pytest.raises(ValueError):
        modal_delta_sync.validate_delta_update_payload(payload, delta_mount_path=str(tmp_path))


def test_verify_delta_dir_ready_requires_done_and_safetensors(tmp_path):
    modal_delta_sync = _modal_delta_sync()
    version_dir = tmp_path / "weight_v000123"
    version_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        modal_delta_sync.verify_delta_dir_ready(str(version_dir))

    (version_dir / "DONE").write_text("")
    with pytest.raises(FileNotFoundError):
        modal_delta_sync.verify_delta_dir_ready(str(version_dir))

    (version_dir / "rank0000_flush000000.safetensors").write_text("")
    modal_delta_sync.verify_delta_dir_ready(str(version_dir))


def test_external_abort_uses_explicit_engine_addrs(monkeypatch):
    sglang_rollout = pytest.importorskip("slime.rollout.sglang_rollout")
    posted = []

    class FakeGenerateState:
        def __init__(self, args):
            self.aborted = False
            self.pendings = set()

    async def fake_get(*args, **kwargs):
        raise AssertionError("external abort should not query router worker endpoints")

    async def fake_post(url, payload):
        posted.append((url, payload))
        return {"ok": True}

    monkeypatch.setattr(sglang_rollout, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(sglang_rollout, "get", fake_get)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    aborted_samples = asyncio.run(
        sglang_rollout.abort(
            Namespace(
                rollout_external=True,
                rollout_external_engine_addrs=["https://rollout.example.modal.run/base/"],
                partial_rollout=False,
            ),
            rollout_id=0,
        )
    )

    assert posted == [("https://rollout.example.modal.run/base/abort_request", {"abort_all": True})]
    assert aborted_samples == []


def test_external_rollout_placement_group_reserves_only_actor_gpus(monkeypatch):
    pytest.importorskip("ray")
    pytest.importorskip("numpy")
    pytest.importorskip("sglang")

    from slime.ray import placement_group as placement_group_mod

    calls = []

    def fake_create_placement_group(num_gpus):
        calls.append(num_gpus)
        return object(), list(range(num_gpus)), list(range(num_gpus))

    monkeypatch.setattr(placement_group_mod, "_create_placement_group", fake_create_placement_group)

    pgs = placement_group_mod.create_placement_groups(
        Namespace(
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=False,
            rollout_external=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=1,
            rollout_num_gpus=1,
            use_critic=False,
        )
    )

    assert calls == [1]
    assert pgs["rollout"][1] == []
    assert pgs["rollout"][2] == []


def test_external_rollout_engines_are_cpu_only_control_actors(monkeypatch):
    pytest.importorskip("ray")
    pytest.importorskip("numpy")
    pytest.importorskip("sglang")

    from slime.ray import rollout as rollout_mod

    actor_options = []

    class FakeRemoteActor:
        def options(self, **kwargs):
            actor_options.append(kwargs)
            return self

        def remote(self, *args, **kwargs):
            return SimpleNamespace(init=SimpleNamespace(remote=lambda **init_kwargs: init_kwargs))

    monkeypatch.setattr(rollout_mod.ray, "remote", lambda cls: FakeRemoteActor())

    group = rollout_mod.ServerGroup(
        args=Namespace(
            debug_train_only=False,
            rollout_external=True,
            rollout_external_engine_addrs=["https://rollout.example.modal.run"],
            num_gpus_per_node=1,
            sglang_dp_size=1,
        ),
        pg=(object(), [], []),
        all_engines=[None],
        num_gpus_per_engine=1,
        num_new_engines=0,
    )

    handles, port_cursors = group.start_engines()

    assert port_cursors == {}
    assert handles[0]["server_url"] == "https://rollout.example.modal.run"
    assert actor_options[0]["num_gpus"] == 0
    assert actor_options[0]["num_cpus"] == 0.2
    assert "scheduling_strategy" not in actor_options[0]
