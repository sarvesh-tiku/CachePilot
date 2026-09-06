from __future__ import annotations

from typing import Protocol

from cachepilot.core.models import InferenceRequest, RoutingDecision, WorkerState


class KVDirectory(Protocol):
    """Logical record of which prefixes each worker is believed to hold."""

    def cache_overlap(self, request: InferenceRequest, worker_id: str) -> float: ...


class NullKVDirectory:
    """Directory used until prefix fingerprinting exists: nothing is ever cached."""

    def cache_overlap(self, request: InferenceRequest, worker_id: str) -> float:
        return 0.0


class Scheduler(Protocol):
    policy: str

    def parameters(self) -> dict[str, float]:
        """Tunable knobs, surfaced through the API so decisions can be reproduced."""
        ...

    async def choose_worker(
        self,
        request: InferenceRequest,
        workers: list[WorkerState],
        kv_directory: KVDirectory,
    ) -> RoutingDecision: ...
