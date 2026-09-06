from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from cachepilot.core.models import RequestResult, WorkerState, WorkerStatus

LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


class Metrics:
    """Prometheus collectors on a private registry so many apps can coexist in one process."""

    def __init__(self) -> None:
        r = CollectorRegistry()
        self.registry = r
        self.requests_total = Counter(
            "cachepilot_requests_total",
            "Completed requests",
            ["policy", "worker", "cache_hit"],
            registry=r,
        )
        self.request_duration_ms = Histogram(
            "cachepilot_request_duration_ms",
            "End-to-end request latency",
            ["policy"],
            buckets=LATENCY_BUCKETS_MS,
            registry=r,
        )
        self.ttft_ms = Histogram(
            "cachepilot_ttft_ms",
            "Time to first token",
            ["policy"],
            buckets=LATENCY_BUCKETS_MS,
            registry=r,
        )
        self.queue_wait_ms = Histogram(
            "cachepilot_queue_wait_ms",
            "Queue wait before prefill",
            ["policy"],
            buckets=LATENCY_BUCKETS_MS,
            registry=r,
        )
        self.cache_hits_total = Counter(
            "cachepilot_cache_hits_total",
            "Requests that reused a resident prefix",
            ["worker"],
            registry=r,
        )
        self.cache_misses_total = Counter(
            "cachepilot_cache_misses_total",
            "Requests with no resident prefix",
            ["worker"],
            registry=r,
        )
        self.cache_evictions_total = Counter(
            "cachepilot_cache_evictions_total",
            "Prefix chunks evicted by LRU",
            ["worker"],
            registry=r,
        )
        self.worker_queue_depth = Gauge(
            "cachepilot_worker_queue_depth", "Requests queued on a worker", ["worker"], registry=r
        )
        self.worker_active_requests = Gauge(
            "cachepilot_worker_active_requests",
            "Requests being generated on a worker",
            ["worker"],
            registry=r,
        )
        self.worker_kv_utilization = Gauge(
            "cachepilot_worker_kv_utilization",
            "Fraction of KV capacity in use",
            ["worker"],
            registry=r,
        )
        self.worker_tokens_per_second = Gauge(
            "cachepilot_worker_tokens_per_second",
            "Estimated decode throughput",
            ["worker"],
            registry=r,
        )
        self.worker_healthy = Gauge(
            "cachepilot_worker_healthy", "1 if the worker is routable", ["worker"], registry=r
        )

    def observe_request(self, policy: str, result: RequestResult) -> None:
        self.requests_total.labels(policy, result.worker_id, str(result.cache_hit).lower()).inc()
        self.request_duration_ms.labels(policy).observe(result.total_latency_ms)
        self.ttft_ms.labels(policy).observe(result.ttft_ms)
        self.queue_wait_ms.labels(policy).observe(result.queue_wait_ms)

    def sync_workers(self, workers: list[WorkerState]) -> None:
        for w in workers:
            self.worker_queue_depth.labels(w.worker_id).set(w.queue_depth)
            self.worker_active_requests.labels(w.worker_id).set(w.active_requests)
            self.worker_kv_utilization.labels(w.worker_id).set(
                w.kv_used_bytes / w.kv_capacity_bytes if w.kv_capacity_bytes else 0.0
            )
            self.worker_tokens_per_second.labels(w.worker_id).set(w.tokens_per_second_estimate)
            self.worker_healthy.labels(w.worker_id).set(
                1 if w.status == WorkerStatus.HEALTHY else 0
            )

    def render(self) -> bytes:
        return generate_latest(self.registry)
