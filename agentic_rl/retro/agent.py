"""mini-swe control loop with a coherent post-tool turn hook."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow

TurnCallback = Callable[["SnapshottingAgent", dict[str, Any], list[dict[str, Any]]], None]


class SnapshottingAgent(DefaultAgent):
    """DefaultAgent that exposes the state after tool output is in ``messages``."""

    def __init__(self, *args, turn_callback: TurnCallback | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_callback = turn_callback

    def execute_actions(self, message: dict) -> list[dict]:
        observations = super().execute_actions(message)
        if self.turn_callback is not None:
            self.turn_callback(self, message, observations)
        return observations

    def resume(self, *, task: str = "", **kwargs) -> dict:
        """Continue from preloaded messages/counters without resetting the agent."""

        if not self.messages:
            raise ValueError("SnapshottingAgent.resume requires preloaded messages")
        self.extra_template_vars |= {"task": task, **kwargs}
        while True:
            try:
                self.step()
            except InterruptAgentFlow as exc:
                self.add_messages(*exc.messages)
            except Exception as exc:
                self.handle_uncaught_exception(exc)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})
