from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from cachepilot.core.models import InferenceRequest


@dataclass(frozen=True)
class TokenEvent:
    index: int
    text: str


@dataclass(frozen=True)
class DoneEvent:
    queue_wait_ms: float
    ttft_ms: float
    total_latency_ms: float
    output_tokens: int
    finish_reason: str


GenerationEvent = TokenEvent | DoneEvent


class ReportsStats(Protocol):
    worker_id: str

    def heartbeat_snapshot(self) -> dict[str, Any]: ...


StatsListener = Callable[[ReportsStats], Awaitable[None]]


class InferenceWorker(Protocol):
    """Interface every worker backend (simulated, vLLM, ...) must satisfy.

    `generate` yields tokens as they are produced and ends with exactly one
    DoneEvent carrying the timings the gateway should record. `backend` names
    the implementation so the API can say which numbers are simulated.
    """

    worker_id: str
    model: str
    backend: str
    on_stats_changed: StatsListener | None

    def generate(
        self, request: InferenceRequest, cache_overlap: float
    ) -> AsyncIterator[GenerationEvent]: ...

    async def health(self) -> bool: ...

    async def stats(self) -> dict[str, Any]: ...

    def heartbeat_snapshot(self) -> dict[str, Any]: ...
