from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from cachepilot.core.errors import WorkerAlreadyRegisteredError, WorkerNotFoundError
from cachepilot.core.models import WorkerState, WorkerStatus, utcnow

StatusListener = Callable[[WorkerState, WorkerStatus], Awaitable[None]]


class WorkerRegistry:
    """In-memory source of truth for worker state within a single gateway process."""

    def __init__(self, on_status_change: StatusListener | None = None) -> None:
        self._workers: dict[str, WorkerState] = {}
        self._lock = asyncio.Lock()
        self._on_status_change = on_status_change

    async def register(
        self,
        worker_id: str,
        model: str,
        kv_capacity_bytes: int,
        backend: str = "simulated",
    ) -> WorkerState:
        async with self._lock:
            if worker_id in self._workers:
                raise WorkerAlreadyRegisteredError(worker_id)
            state = WorkerState(
                worker_id=worker_id,
                model=model,
                backend=backend,
                kv_capacity_bytes=kv_capacity_bytes,
            )
            self._workers[worker_id] = state
            return state

    async def heartbeat(
        self,
        worker_id: str,
        *,
        queue_depth: int,
        active_requests: int,
        tokens_per_second_estimate: float,
    ) -> WorkerState:
        async with self._lock:
            state = self._get(worker_id)
            state.queue_depth = queue_depth
            state.active_requests = active_requests
            state.tokens_per_second_estimate = tokens_per_second_estimate
            state.last_heartbeat = utcnow()
            previous = state.status
            # A heartbeat proves liveness, so an unhealthy worker recovers.
            # Draining is an operator decision and is never overridden here.
            if state.status == WorkerStatus.UNHEALTHY:
                state.status = WorkerStatus.HEALTHY
        await self._notify(state, previous)
        return state

    async def update_kv_usage(self, worker_id: str, kv_used_bytes: int) -> WorkerState:
        async with self._lock:
            state = self._get(worker_id)
            state.kv_used_bytes = kv_used_bytes
            return state

    async def mark_unhealthy(self, worker_id: str) -> WorkerState:
        return await self._set_status(worker_id, WorkerStatus.UNHEALTHY)

    async def drain(self, worker_id: str) -> WorkerState:
        return await self._set_status(worker_id, WorkerStatus.DRAINING)

    async def restore(self, worker_id: str) -> WorkerState:
        return await self._set_status(worker_id, WorkerStatus.HEALTHY)

    async def get(self, worker_id: str) -> WorkerState:
        async with self._lock:
            return self._get(worker_id)

    async def list_all(self) -> list[WorkerState]:
        async with self._lock:
            return list(self._workers.values())

    async def list_healthy(self) -> list[WorkerState]:
        async with self._lock:
            return [w for w in self._workers.values() if w.status == WorkerStatus.HEALTHY]

    async def _set_status(self, worker_id: str, status: WorkerStatus) -> WorkerState:
        async with self._lock:
            state = self._get(worker_id)
            previous = state.status
            state.status = status
        await self._notify(state, previous)
        return state

    async def _notify(self, state: WorkerState, previous: WorkerStatus) -> None:
        if self._on_status_change is not None and state.status != previous:
            await self._on_status_change(state, previous)

    def _get(self, worker_id: str) -> WorkerState:
        try:
            return self._workers[worker_id]
        except KeyError:
            raise WorkerNotFoundError(worker_id) from None
