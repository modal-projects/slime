"""Regression test for the cross-event-loop session-teardown crash.

The HTTP adapter is served on its OWN event loop (run_app_in_thread), so every
``inflight[sid]`` handler task is created on that *serving* loop. Session
teardown, however, is awaited from the rollout loop (``generate.py`` ->
``finish_session`` -> ``shutdown_session``). Before the fix, that ran
``asyncio.wait``/``cancel``/``gather`` over serving-loop tasks from the rollout
loop, which raises "got Future attached to a different loop" and (under uvloop)
aborts the whole process whenever a session was finalized mid-request -- e.g.
on a budget-kill while a turn was still generating. The fix marshals the
shutdown onto the captured serving loop.
"""

import asyncio
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.adapters.common import BaseAdapter, shutdown_session_tasks


def _start_background_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Spin up a forever-running loop in a daemon thread (mimics the adapter)."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, name="serving-loop", daemon=True)
    thread.start()
    ready.wait(timeout=5)
    return loop, thread


def _stop_background_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _register_inflight_on(loop: asyncio.AbstractEventLoop, adapter: BaseAdapter, sid: str) -> asyncio.Task:
    """Create a long-running handler task ON ``loop`` and register it, exactly as
    ``_run_turn`` does via ``asyncio.current_task()`` inside an HTTP handler."""
    started = threading.Event()

    async def _inflight() -> None:
        started.set()
        await asyncio.sleep(3600)  # never completes on its own

    async def _setup() -> asyncio.Task:
        task = asyncio.create_task(_inflight())
        adapter.inflight.setdefault(sid, set()).add(task)
        return task

    task = asyncio.run_coroutine_threadsafe(_setup(), loop).result(timeout=5)
    assert started.wait(timeout=5), "inflight task never started"
    return task


@pytest.mark.unit
def test_capture_serving_loop_records_running_loop():
    adapter = BaseAdapter(tokenizer=object(), sglang_url="http://sglang")

    async def _go() -> None:
        await adapter._capture_serving_loop(adapter.app)
        assert adapter._serving_loop is asyncio.get_running_loop()

    asyncio.run(_go())


@pytest.mark.unit
def test_shutdown_session_bridges_to_serving_loop_and_cancels_inflight():
    """The fix: tearing down from a DIFFERENT loop must not raise, and must
    actually cancel the serving-loop handler task."""
    adapter = BaseAdapter(tokenizer=object(), sglang_url="http://sglang")
    serving_loop, thread = _start_background_loop()
    try:
        adapter._serving_loop = serving_loop  # what _capture_serving_loop sets
        sid = "sess-cross-loop"
        task = _register_inflight_on(serving_loop, adapter, sid)

        # Teardown from a fresh rollout loop (loop A != serving loop).
        async def _teardown() -> None:
            await adapter.shutdown_session(sid, wait_timeout=0.2)

        asyncio.run(_teardown())  # must NOT raise "attached to a different loop"

        assert task.done(), "inflight handler task should be cancelled/finished"
        assert task.cancelled()
        assert sid in adapter.closed
        assert sid not in adapter.inflight
    finally:
        _stop_background_loop(serving_loop, thread)


@pytest.mark.unit
def test_shutdown_session_same_loop_path_still_works():
    """Degenerate path: when the serving loop IS the current loop (or unset),
    shutdown runs inline and still cancels the inflight task."""
    adapter = BaseAdapter(tokenizer=object(), sglang_url="http://sglang")

    async def _go() -> None:
        adapter._serving_loop = asyncio.get_running_loop()
        sid = "sess-same-loop"

        async def _inflight() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(_inflight())
        adapter.inflight.setdefault(sid, set()).add(task)
        await asyncio.sleep(0)  # let it start

        await adapter.shutdown_session(sid, wait_timeout=0.2)
        assert task.cancelled()
        assert sid in adapter.closed
        assert sid not in adapter.inflight

    asyncio.run(_go())


@pytest.mark.unit
def test_unbridged_shutdown_over_foreign_loop_tasks_raises():
    """Documents WHY the bridge exists: running the raw task-shutdown over
    foreign-loop tasks from another loop is exactly what used to crash. If this
    ever stops raising, the bridge in shutdown_session is no longer load-bearing
    and this guard can be revisited."""
    adapter = BaseAdapter(tokenizer=object(), sglang_url="http://sglang")
    serving_loop, thread = _start_background_loop()
    try:
        sid = "sess-foreign"
        _register_inflight_on(serving_loop, adapter, sid)

        async def _raw_teardown() -> None:
            # Bypass the fix and call the low-level helper from the wrong loop.
            await shutdown_session_tasks(sid, adapter.closed, adapter.inflight, wait_timeout=0.2)

        with pytest.raises(RuntimeError, match="different loop"):
            asyncio.run(_raw_teardown())
    finally:
        _stop_background_loop(serving_loop, thread)
