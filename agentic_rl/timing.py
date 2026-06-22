"""Tiny phase timer for per-episode wall-clock breakdown (boot/prep/agent/grade)."""

from __future__ import annotations

import time
from contextlib import contextmanager


class PhaseTimer:
    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    def record(self, phase: str, seconds: float) -> None:
        self._totals[phase] = self._totals.get(phase, 0.0) + seconds

    @contextmanager
    def phase(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.record(name, time.monotonic() - t0)

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 1) for k, v in self._totals.items()}
