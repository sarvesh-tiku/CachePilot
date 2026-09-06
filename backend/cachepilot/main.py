from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cachepilot.api.routes_benchmarks import router as benchmarks_router
from cachepilot.api.routes_chat import router as chat_router
from cachepilot.api.routes_decisions import router as decisions_router
from cachepilot.api.routes_events import router as events_router
from cachepilot.api.routes_kv import router as kv_router
from cachepilot.api.routes_metrics import prometheus_router
from cachepilot.api.routes_metrics import router as metrics_router
from cachepilot.api.routes_policies import router as policies_router
from cachepilot.api.routes_requests import router as requests_router
from cachepilot.api.routes_workers import router as workers_router
from cachepilot.core.clock import Clock, RealClock
from cachepilot.core.config import Settings, get_settings
from cachepilot.gateway.factory import build_stack
from cachepilot.persistence.db import create_schema, make_engine
from cachepilot.persistence.repositories import SqlRunStore
from cachepilot.scheduler.kv_aware import KVAwareWeights
from cachepilot.scheduler.scoring import TTFTEstimator
from cachepilot.telemetry.logging import configure_logging
from cachepilot.workers.base import InferenceWorker
from cachepilot.workers.registry import WorkerRegistry
from cachepilot.workers.simulated import SimulatedWorkerConfig
from cachepilot.workers.vllm import VLLMWorker


async def _simulated_heartbeat_loop(
    registry: WorkerRegistry, worker: InferenceWorker, interval_s: float
) -> None:
    while True:
        await registry.heartbeat(worker.worker_id, **worker.heartbeat_snapshot())
        await asyncio.sleep(interval_s)


async def _real_worker_poll_loop(
    registry: WorkerRegistry,
    worker: VLLMWorker,
    kv_capacity_bytes: int,
    interval_s: float,
    logger: logging.Logger,
) -> None:
    """Health-check and scrape a real server; observed stats override gateway estimates."""
    while True:
        try:
            if await worker.health():
                stats = await worker.stats()
                await registry.heartbeat(worker.worker_id, **worker.heartbeat_snapshot())
                fraction = stats.get("kv_usage_fraction")
                if fraction is not None:
                    await registry.update_kv_usage(
                        worker.worker_id, int(float(fraction) * kv_capacity_bytes)
                    )
            else:
                await registry.mark_unhealthy(worker.worker_id)
        except Exception:  # a flaky backend must not kill the poll loop
            logger.exception(
                "real_worker_poll_failed", extra={"fields": {"worker": worker.worker_id}}
            )
            await registry.mark_unhealthy(worker.worker_id)
        await asyncio.sleep(interval_s)


def kv_aware_weights(settings: Settings) -> KVAwareWeights:
    return KVAwareWeights(
        alpha=settings.kv_aware_alpha,
        beta=settings.kv_aware_beta,
        gamma=settings.kv_aware_gamma,
        delta=settings.kv_aware_delta,
    )


def ttft_estimator(settings: Settings) -> TTFTEstimator:
    return TTFTEstimator(
        queue_wait_ms_per_pending=settings.ttft_queue_wait_ms_per_pending,
        prefill_ms_per_token=settings.ttft_prefill_ms_per_token,
        fixed_overhead_ms=settings.ttft_fixed_overhead_ms,
    )


def simulated_worker_config(settings: Settings) -> SimulatedWorkerConfig:
    if settings.simulated_queue_model not in ("linear", "batched"):
        raise ValueError(
            f"CACHEPILOT_SIMULATED_QUEUE_MODEL must be 'linear' or 'batched', "
            f"got {settings.simulated_queue_model!r}"
        )
    return SimulatedWorkerConfig(
        queue_model=settings.simulated_queue_model,  # type: ignore[arg-type]
        max_batch_size=settings.simulated_max_batch_size,
    )


def build_real_workers(settings: Settings) -> list[tuple[VLLMWorker, int]]:
    if not settings.vllm_url:
        return []
    if not settings.vllm_model:
        raise ValueError("CACHEPILOT_VLLM_MODEL is required when CACHEPILOT_VLLM_URL is set")
    worker = VLLMWorker(
        settings.vllm_worker_id,
        settings.vllm_model,
        settings.vllm_url,
        api_key=settings.vllm_api_key,
        timeout_s=settings.vllm_timeout_s,
    )
    return [(worker, settings.vllm_kv_capacity_bytes)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    clock: Clock = app.state.clock
    logger = configure_logging(settings.log_level)

    engine = make_engine(settings.database_url)
    await create_schema(engine)
    store = SqlRunStore(engine)

    real_workers = build_real_workers(settings)
    stack = await build_stack(
        simulated_workers=settings.simulated_worker_count,
        clock=clock,
        store=store,
        default_policy=settings.scheduler_policy,
        weights=kv_aware_weights(settings),
        chunk_tokens=settings.kv_chunk_tokens,
        bytes_per_token=settings.effective_kv_bytes_per_token(),
        estimator=ttft_estimator(settings),
        worker_config=simulated_worker_config(settings),
        real_workers=real_workers,
        logger=logger,
    )

    app.state.registry = stack.registry
    app.state.kv_directory = stack.kv_directory
    app.state.store = store
    app.state.gateway = stack.gateway
    app.state.metrics = stack.metrics
    app.state.telemetry = stack.telemetry
    app.state.estimator = stack.estimator

    interval = settings.worker_heartbeat_interval_s
    real_ids = {w.worker_id for w, _ in real_workers}
    tasks = [
        asyncio.create_task(_simulated_heartbeat_loop(stack.registry, w, interval))
        for w in stack.workers
        if w.worker_id not in real_ids
    ]
    tasks += [
        asyncio.create_task(_real_worker_poll_loop(stack.registry, w, capacity, interval, logger))
        for w, capacity in real_workers
    ]
    logger.info(
        "gateway_started",
        extra={
            "fields": {
                "simulated_workers": settings.simulated_worker_count,
                "real_workers": sorted(real_ids),
                "policy": settings.scheduler_policy,
                "queue_model": settings.simulated_queue_model,
                "ttft_estimator": stack.estimator.parameters(),
                "environment": settings.environment,
            }
        },
    )

    yield

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for worker, _ in real_workers:
        await worker.aclose()
    await engine.dispose()


def create_app(settings: Settings | None = None, clock: Clock | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    app.state.clock = clock or RealClock()
    app.include_router(chat_router)
    app.include_router(workers_router)
    app.include_router(decisions_router)
    app.include_router(kv_router)
    app.include_router(policies_router)
    app.include_router(requests_router)
    app.include_router(metrics_router)
    app.include_router(prometheus_router)
    app.include_router(benchmarks_router)
    app.include_router(events_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
