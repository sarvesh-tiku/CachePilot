from __future__ import annotations

from pydantic import BaseModel

from cachepilot.core.models import RequestResult, WorkerState, WorkerStatus
from cachepilot.kv.directory import WorkerCacheSummary


class Percentiles(BaseModel):
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    rank = max(0, min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1))))
    return sorted_values[rank]


def percentiles(values: list[float]) -> Percentiles:
    ordered = sorted(values)
    return Percentiles(
        mean=sum(ordered) / len(ordered) if ordered else 0.0,
        p50=percentile(ordered, 0.50),
        p95=percentile(ordered, 0.95),
        p99=percentile(ordered, 0.99),
        max=ordered[-1] if ordered else 0.0,
    )


class WorkerLoad(BaseModel):
    worker_id: str
    backend: str
    status: WorkerStatus
    queue_depth: int
    active_requests: int
    kv_utilization: float
    tokens_per_second_estimate: float
    cache_entries: int
    completed_in_window: int


class MetricsSummary(BaseModel):
    window_s: int
    completed_requests: int
    requests_per_s: float
    output_tokens_per_s: float
    ttft_ms: Percentiles
    total_latency_ms: Percentiles
    queue_wait_ms: Percentiles
    cache_hit_rate: float
    mean_cache_overlap: float
    healthy_workers: int
    total_workers: int
    active_requests: int
    queued_requests: int
    workers: list[WorkerLoad]


def summarize(
    results: list[RequestResult],
    workers: list[WorkerState],
    caches: list[WorkerCacheSummary],
    *,
    window_s: int,
) -> MetricsSummary:
    completed = len(results)
    entries_by_worker = {c.worker_id: c.entries for c in caches}
    completed_by_worker: dict[str, int] = {}
    for result in results:
        completed_by_worker[result.worker_id] = completed_by_worker.get(result.worker_id, 0) + 1

    return MetricsSummary(
        window_s=window_s,
        completed_requests=completed,
        requests_per_s=completed / window_s,
        output_tokens_per_s=sum(r.output_tokens for r in results) / window_s,
        ttft_ms=percentiles([r.ttft_ms for r in results]),
        total_latency_ms=percentiles([r.total_latency_ms for r in results]),
        queue_wait_ms=percentiles([r.queue_wait_ms for r in results]),
        cache_hit_rate=sum(1 for r in results if r.cache_hit) / completed if completed else 0.0,
        mean_cache_overlap=sum(r.cache_overlap for r in results) / completed if completed else 0.0,
        healthy_workers=sum(1 for w in workers if w.status == WorkerStatus.HEALTHY),
        total_workers=len(workers),
        active_requests=sum(w.active_requests for w in workers),
        queued_requests=sum(w.queue_depth for w in workers),
        workers=[
            WorkerLoad(
                worker_id=w.worker_id,
                backend=w.backend,
                status=w.status,
                queue_depth=w.queue_depth,
                active_requests=w.active_requests,
                kv_utilization=w.kv_used_bytes / w.kv_capacity_bytes if w.kv_capacity_bytes else 0,
                tokens_per_second_estimate=w.tokens_per_second_estimate,
                cache_entries=entries_by_worker.get(w.worker_id, 0),
                completed_in_window=completed_by_worker.get(w.worker_id, 0),
            )
            for w in workers
        ],
    )
