"""In-rollout profiling: phase timers + per-session adapter turn stats.

Two collectors, both feeding ``sample.metadata["timing"]`` (and from there the
``rollout_{id}.pt`` debug dumps that ``profile.py`` aggregates offline):

``PhaseTimer``
    Wall-clock spans for the env-side phases of one sample's rollout
    (work_boot / prep / agent / diff / eval / eval_boot). Owned by the env's
    ``rollout()``; serialized via ``as_dict()`` into ``RewardResult.extra``.

``timing_middleware`` / ``pop_session_stats``
    An aiohttp middleware installed on the adapter app (``generate._State``
    owns both the adapter and the server start, so this needs NO slime-core
    change). It times every generation request and attributes it to the
    session via the bearer token -- which IS the slime session_id by the
    adapter's auth convention. ``gen_s`` measures the whole adapter hop
    (render -> tokenize -> SGLang /generate -> response build), so
    ``gen_s/n_turns`` minus the engine-side e2e latency (W&B ``sgl_engine``)
    is the adapter's own per-turn overhead.

Caveats (fine for profiling, do not treat as exact accounting):
  * the store is per-process (one rollout worker = one adapter = one store);
  * a request in flight when a sample is aborted may be missing from, or
    inflate, that sample's stats;
  * dict mutation is GIL-atomic and the reader (``generate()``) only pops
    after the agent finished, so no locking is needed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from aiohttp import web

# Endpoints that represent one model turn (OpenAI chat/responses, Anthropic
# messages). Health checks and model listings must not count as turns.
_GENERATION_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")


def _session_id(request: web.Request) -> str:
    """Bearer token (slime's session auth convention), else x-api-key.

    Mirrors slime.agent.adapters.common.request_session_id without importing
    it (that pulls the torch-heavy slime chain into this otherwise
    dependency-free module). The body-based fallbacks there don't apply: our
    runtimes always authenticate with session_id as the key.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer ") and auth[7:].strip():
        return auth[7:].strip()
    return (request.headers.get("X-Api-Key") or "").strip() or "default"


class PhaseTimer:
    """Accumulates named wall-clock spans for one sample's rollout."""

    def __init__(self) -> None:
        self._spans: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - t0)

    def record(self, name: str, seconds: float) -> None:
        self._spans[name] = round(self._spans.get(name, 0.0) + seconds, 3)

    def as_dict(self) -> dict[str, float]:
        return dict(self._spans)


# ---------------------------------------------------------------------------
# Adapter-side per-session turn stats
# ---------------------------------------------------------------------------
_SESSION_STATS: dict[str, dict[str, float]] = {}


@web.middleware
async def timing_middleware(request: web.Request, handler):
    if request.method != "POST" or not request.path.endswith(_GENERATION_PATHS):
        return await handler(request)
    t0 = time.monotonic()
    try:
        return await handler(request)
    finally:
        sid = _session_id(request)
        stats = _SESSION_STATS.setdefault(sid, {"n_turns": 0, "gen_s": 0.0})
        stats["n_turns"] += 1
        stats["gen_s"] = round(stats["gen_s"] + (time.monotonic() - t0), 3)


def install(app: web.Application) -> None:
    """Idempotently add the timing middleware (call BEFORE the app starts)."""
    if timing_middleware not in app.middlewares:
        app.middlewares.append(timing_middleware)


def pop_session_stats(session_id: str) -> dict[str, float] | None:
    """Drain one session's turn stats (None if the agent never dialed in)."""
    return _SESSION_STATS.pop(session_id, None)
