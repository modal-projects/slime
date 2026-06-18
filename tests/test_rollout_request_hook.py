import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import ray  # noqa: F401
except ImportError:
    pass

try:
    from tests.plugin_contracts._shared import install_stubs
except ImportError:
    from plugin_contracts._shared import install_stubs

install_stubs(with_sglang_router=True, with_transformers=True)

NUM_GPUS = 0

from slime.rollout import sglang_rollout  # noqa: E402
from slime.rollout.sglang_rollout import generate  # noqa: E402
from slime.utils import http_utils  # noqa: E402
from slime.utils.types import Sample  # noqa: E402


def _args(**overrides):
    values = {
        "ci_test": False,
        "rollout_endpoint_url": None,
        "sglang_router_ip": "10.0.0.1",
        "sglang_router_port": 30000,
        "sglang_model_routers": None,
        "router_policy": None,
        "use_rollout_routing_replay": False,
        "partial_rollout": False,
        "mask_offpolicy_in_partial_rollout": False,
        "sglang_speculative_algorithm": None,
        "custom_rollout_request_hook_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


class _Tokenizer:
    def encode(self, prompt, add_special_tokens=False):
        assert add_special_tokens is False
        return [101, len(prompt)]


class _GenerateState:
    def __init__(self, args):
        self.args = args
        self.tokenizer = _Tokenizer()
        self.processor = None
        self.pendings = set()
        self.aborted = False


def _fake_generate_response():
    return {
        "text": " answer",
        "meta_info": {
            "output_token_logprobs": [[-0.25, 42]],
            "finish_reason": {"type": "stop"},
            "prompt_tokens": 2,
            "cached_tokens": 1,
        },
    }


def _run_generate(args, monkeypatch, sample_index=5):
    captured = {}

    async def fake_post(url, payload, headers=None, max_retries=60, retry_sleep=1.0):
        captured.update(url=url, payload=payload, headers=headers, max_retries=max_retries, retry_sleep=retry_sleep)
        return _fake_generate_response()

    monkeypatch.setattr(sglang_rollout, "GenerateState", _GenerateState)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    sample = asyncio.run(generate(args, Sample(index=sample_index, prompt="hi"), {"max_new_tokens": 8}))
    return sample, captured


def test_generate_without_hook_sends_plain_request(monkeypatch):
    """With no hook configured, generate calls post directly — no request dict, no extra fields."""
    args = _args(rollout_endpoint_url="https://rollout.example")
    sample, captured = _run_generate(args, monkeypatch)

    assert captured["url"] == "https://rollout.example/generate"
    assert "weight_version" not in captured["payload"]
    # default path uses post's defaults — the hook never ran to change them
    assert captured["max_retries"] == 60
    assert captured["retry_sleep"] == 1.0
    assert sample.status == Sample.Status.COMPLETED


def test_request_hook_can_mutate_request_in_place(monkeypatch):
    def hook(args, sample, request):
        # request carries how-to-send fields (incl. retry knobs); rollout context comes off the sample.
        assert set(request) == {"url", "payload", "headers", "max_retries", "retry_sleep"}
        request["headers"] = {**(request["headers"] or {}), "Authorization": "Bearer t"}
        request["payload"]["weight_version"] = {"min_version": sample.index}
        request["max_retries"], request["retry_sleep"] = 120, 0.5

    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: hook)
    args = _args(rollout_endpoint_url="https://rollout.example", custom_rollout_request_hook_path="example.hook")
    sample, captured = _run_generate(args, monkeypatch, sample_index=5)

    assert captured["headers"]["Authorization"] == "Bearer t"
    assert captured["payload"]["weight_version"] == {"min_version": 5}
    assert captured["max_retries"] == 120
    assert captured["retry_sleep"] == 0.5
    assert sample.status == Sample.Status.COMPLETED


def test_request_hook_can_return_dict_of_updates_and_be_async(monkeypatch):
    async def hook(args, sample, request):
        payload = dict(request["payload"])
        payload["weight_version"] = {"min_version": sample.index}
        return {"payload": payload}

    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: hook)
    args = _args(rollout_endpoint_url="https://rollout.example", custom_rollout_request_hook_path="example.hook")
    sample, captured = _run_generate(args, monkeypatch, sample_index=3)

    assert captured["payload"]["weight_version"] == {"min_version": 3}
    assert sample.status == Sample.Status.COMPLETED


def test_request_hook_must_return_none_or_dict(monkeypatch):
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: lambda a, s, r: "nope")
    args = _args(rollout_endpoint_url="https://rollout.example", custom_rollout_request_hook_path="example.hook")
    with pytest.raises(TypeError, match="None or a dict"):
        _run_generate(args, monkeypatch)


class _FakeStreamResponse:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamClient:
    def __init__(self, captured, lines):
        self._captured = captured
        self._lines = lines

    def stream(self, method, url, json=None, headers=None):
        self._captured.update(method=method, url=url, json=json, headers=headers)
        return _FakeStreamResponse(self._lines)


def test_streaming_generate_routes_to_endpoint_and_applies_hook(monkeypatch):
    """The streaming path funnels through the same hook and uses get_model_url, so it reaches the
    opaque endpoint (not the unset router) and a hook can shape the request."""
    from slime.rollout import sglang_streaming_rollout

    chunk = {
        "text": " answer",
        "meta_info": {
            "output_token_logprobs": [[-0.25, 42]],
            "finish_reason": {"type": "stop"},
            "prompt_tokens": 2,
            "cached_tokens": 0,
        },
    }
    captured = {}
    client = _FakeStreamClient(captured, [f"data: {json.dumps(chunk)}", "data: [DONE]"])

    def hook(args, sample, request):
        request["headers"] = {**(request["headers"] or {}), "Authorization": "Bearer t"}
        request["payload"]["weight_version"] = {"min_version": sample.index}

    # The hook runs inside apply_rollout_request_hook, which resolves load_function from sglang_rollout.
    monkeypatch.setattr(sglang_streaming_rollout, "GenerateState", _GenerateState)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: hook)
    monkeypatch.setattr(http_utils, "_http_client", client)

    args = _args(rollout_endpoint_url="https://rollout.example", custom_rollout_request_hook_path="example.hook")
    sample = asyncio.run(
        sglang_streaming_rollout.generate_streaming(args, Sample(index=7, prompt="hi"), {"max_new_tokens": 8})
    )

    assert captured["url"] == "https://rollout.example/generate"
    assert captured["headers"]["Authorization"] == "Bearer t"
    assert captured["json"]["weight_version"] == {"min_version": 7}
    assert sample.status == Sample.Status.COMPLETED


def test_streaming_generate_without_hook_opens_stream_unchanged(monkeypatch):
    """No hook configured: the stream is opened with the original payload/headers, no hook detour."""
    from slime.rollout import sglang_streaming_rollout

    chunk = {
        "text": " answer",
        "meta_info": {"output_token_logprobs": [[-0.25, 42]], "finish_reason": {"type": "stop"}},
    }
    captured = {}
    client = _FakeStreamClient(captured, [f"data: {json.dumps(chunk)}", "data: [DONE]"])

    monkeypatch.setattr(sglang_streaming_rollout, "GenerateState", _GenerateState)
    monkeypatch.setattr(http_utils, "_http_client", client)

    args = _args(rollout_endpoint_url="https://rollout.example")
    asyncio.run(sglang_streaming_rollout.generate_streaming(args, Sample(index=0, prompt="hi"), {"max_new_tokens": 8}))

    assert captured["url"] == "https://rollout.example/generate"
    assert "weight_version" not in captured["json"]


def test_post_retries_until_version_available_with_backoff(monkeypatch):
    """A gating hook relies on this: the fleet rejects a not-yet-loaded version, and post retries
    with retry_sleep backoff until it is served."""
    import httpx

    attempts = []
    sleeps = []

    class _Resp:
        def __init__(self, status):
            self.status_code = status
            self.text = "weight version not loaded"

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("not ready", request=None, response=self)

        async def aread(self):
            return b'{"ok": true}'

        async def aclose(self):
            pass

    class _Client:
        async def post(self, url, json=None, headers=None):
            attempts.append(json)
            return _Resp(409 if len(attempts) <= 2 else 200)

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(http_utils.asyncio, "sleep", fake_sleep)

    payload = {"weight_version": {"min_version": 11}}
    out = asyncio.run(http_utils._post(_Client(), "http://fleet/generate", payload, max_retries=5, retry_sleep=0.01))

    assert out == {"ok": True}
    assert len(attempts) == 3
    assert sleeps == [0.01, 0.01]  # backed off once per rejection, honoring retry_sleep
