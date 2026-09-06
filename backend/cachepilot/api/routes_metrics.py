from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST

from cachepilot.core.models import utcnow
from cachepilot.kv.directory import InMemoryKVDirectory
from cachepilot.persistence.repositories import RunStore
from cachepilot.telemetry.metrics import MetricsSummary, summarize
from cachepilot.telemetry.prometheus import Metrics
from cachepilot.workers.registry import WorkerRegistry

router = APIRouter(prefix="/api/metrics", tags=["metrics"])
prometheus_router = APIRouter(tags=["metrics"])


@router.get("/summary", response_model=MetricsSummary)
async def metrics_summary(
    request: Request,
    window_s: Annotated[int, Query(ge=1, le=3600)] = 60,
) -> MetricsSummary:
    store: RunStore = request.app.state.store
    registry: WorkerRegistry = request.app.state.registry
    directory: InMemoryKVDirectory = request.app.state.kv_directory

    results = await store.list_results(since=utcnow() - timedelta(seconds=window_s))
    return summarize(results, await registry.list_all(), directory.summaries(), window_s=window_s)


@prometheus_router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request) -> Response:
    metrics: Metrics = request.app.state.metrics
    registry: WorkerRegistry = request.app.state.registry
    metrics.sync_workers(await registry.list_all())
    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)
