from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from cachepilot.core.clock import Clock
from cachepilot.core.models import EventType, WorkerState, WorkerStatus
from cachepilot.gateway.service import InferenceGateway
from cachepilot.kv.directory import InMemoryKVDirectory
from cachepilot.kv.fingerprint import Fingerprinter
from cachepilot.persistence.repositories import RunStore
from cachepilot.scheduler.kv_aware import KVAwareWeights
from cachepilot.scheduler.policies import build_all_schedulers
from cachepilot.scheduler.scoring import TTFTEstimator
from cachepilot.telemetry.events import Telemetry
from cachepilot.telemetry.prometheus import Metrics
from cachepilot.workers.base import InferenceWorker, ReportsStats
from cachepilot.workers.batched import BatchedSimulatedWorker
from cachepilot.workers.registry import WorkerRegistry
from cachepilot.workers.simulated import SimulatedWorker, SimulatedWorkerConfig


@dataclass
class Stack:
    registry: WorkerRegistry
    kv_directory: InMemoryKVDirectory
    workers: list[InferenceWorker]
    gateway: InferenceGateway
    telemetry: Telemetry
    metrics: Metrics
    estimator: TTFTEstimator


async def build_stack(
    *,
    simulated_workers: int,
    clock: Clock,
    store: RunStore,
    default_policy: str,
    weights: KVAwareWeights,
    chunk_tokens: int,
    bytes_per_token: int,
    worker_config: SimulatedWorkerConfig | None = None,
    estimator: TTFTEstimator | None = None,
    real_workers: Sequence[tuple[InferenceWorker, int]] = (),
    model: str = "sim-model",
    logger: logging.Logger | None = None,
) -> Stack:
    """Wire registry, KV directory, workers, schedulers, telemetry, and gateway.

    Shared by the API server and the benchmark runner so both exercise the
    identical routing path. `real_workers` pairs each real worker with the KV
    capacity the gateway should assume for it; its directory entries are marked
    *estimated* because the gateway cannot see the server's actual KV state.
    """
    config = worker_config or SimulatedWorkerConfig()
    estimator = estimator or TTFTEstimator()
    metrics = Metrics()
    telemetry = Telemetry(store, metrics, logger)

    async def on_status_change(state: WorkerState, previous: WorkerStatus) -> None:
        if state.status == WorkerStatus.UNHEALTHY:
            event_type = EventType.WORKER_UNHEALTHY
        elif state.status == WorkerStatus.DRAINING:
            event_type = EventType.WORKER_DRAINING
        else:
            event_type = EventType.WORKER_RECOVERED
        await telemetry.emit(
            event_type,
            worker_id=state.worker_id,
            previous=previous.value,
            status=state.status.value,
        )

    registry = WorkerRegistry(on_status_change=on_status_change)
    kv_directory = InMemoryKVDirectory(chunk_tokens=chunk_tokens, bytes_per_token=bytes_per_token)

    async def publish(worker: ReportsStats) -> None:
        await registry.heartbeat(worker.worker_id, **worker.heartbeat_snapshot())

    workers: list[InferenceWorker] = []
    worker_class = BatchedSimulatedWorker if config.queue_model == "batched" else SimulatedWorker
    for i in range(simulated_workers):
        simulated = worker_class(
            worker_id=f"worker-{i}",
            model=model,
            config=config,
            seed=i,
            clock=clock,
            on_stats_changed=publish,
        )
        await registry.register(simulated.worker_id, simulated.model, config.kv_capacity_bytes)
        kv_directory.register_worker(simulated.worker_id, config.kv_capacity_bytes)
        workers.append(simulated)

    for worker, kv_capacity in real_workers:
        worker.on_stats_changed = publish
        await registry.register(worker.worker_id, worker.model, kv_capacity, backend=worker.backend)
        kv_directory.register_worker(worker.worker_id, kv_capacity, source="estimated")
        workers.append(worker)

    gateway = InferenceGateway(
        registry=registry,
        workers={w.worker_id: w for w in workers},
        schedulers=build_all_schedulers(weights=weights, estimator=estimator),
        fingerprinter=Fingerprinter(chunk_tokens),
        kv_directory=kv_directory,
        store=store,
        telemetry=telemetry,
        default_policy=default_policy,
    )
    return Stack(registry, kv_directory, workers, gateway, telemetry, metrics, estimator)
