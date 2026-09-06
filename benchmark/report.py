from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from benchmark.workloads import WorkloadSpec
from cachepilot.core.models import utcnow
from cachepilot.kv.directory import WorkerCacheSummary
from cachepilot.telemetry.metrics import Percentiles, percentiles


class RequestRow(BaseModel):
    index: int
    arrival_ms: float
    completed_ms: float
    prefix_id: str
    worker_id: str
    prompt_tokens: int
    output_tokens: int
    effective_prefill_tokens: int
    cache_overlap: float
    cache_hit: bool
    queue_wait_ms: float
    ttft_ms: float
    total_latency_ms: float
    pending_at_decision: int = Field(
        default=0, description="queued + active requests on the chosen worker when routed"
    )


class BenchmarkMetrics(BaseModel):
    requests: int
    completed: int
    makespan_ms: float
    requests_per_s: float
    output_tokens_per_s: float
    ttft_ms: Percentiles
    total_latency_ms: Percentiles
    queue_wait_ms: Percentiles
    cache_hit_rate: float
    mean_cache_overlap: float
    ttft_ms_by_cache: dict[str, Percentiles] = Field(
        default_factory=dict, description="TTFT split by estimated cache hit/miss"
    )
    prefill_tokens_total: int
    prefill_tokens_saved: int
    prefill_saved_fraction: float
    worker_request_counts: dict[str, int]
    worker_imbalance: float = Field(description="max requests on one worker / mean per worker")
    kv: list[WorkerCacheSummary]


class BenchmarkRun(BaseModel):
    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=utcnow)
    workload: WorkloadSpec
    policy: str
    policy_parameters: dict[str, float]
    seed: int
    requested: int
    workers: int
    metrics: BenchmarkMetrics
    mode: Literal["simulated", "live"] = "simulated"
    gateway_url: str | None = None
    worker_backends: dict[str, str] = Field(default_factory=dict)
    worker_config: dict[str, Any] = Field(
        default_factory=dict, description="SimulatedWorkerConfig used for simulated runs"
    )


def aggregate(
    rows: list[RequestRow],
    *,
    requested: int,
    worker_ids: list[str],
    kv: list[WorkerCacheSummary],
) -> BenchmarkMetrics:
    completed = len(rows)
    if completed == 0:
        raise ValueError("no completed requests to aggregate")

    first_arrival = min(r.arrival_ms for r in rows)
    last_completion = max(r.completed_ms for r in rows)
    makespan_ms = max(last_completion - first_arrival, 1e-9)
    output_tokens = sum(r.output_tokens for r in rows)

    counts = {worker_id: 0 for worker_id in worker_ids}
    for row in rows:
        counts[row.worker_id] = counts.get(row.worker_id, 0) + 1
    mean_per_worker = completed / max(len(counts), 1)

    prompt_total = sum(r.prompt_tokens for r in rows)
    prefill_total = sum(r.effective_prefill_tokens for r in rows)
    by_cache = {
        label: percentiles([r.ttft_ms for r in rows if r.cache_hit is flag])
        for label, flag in (("hit", True), ("miss", False))
        if any(r.cache_hit is flag for r in rows)
    }

    return BenchmarkMetrics(
        requests=requested,
        completed=completed,
        makespan_ms=makespan_ms,
        requests_per_s=completed / (makespan_ms / 1000.0),
        output_tokens_per_s=output_tokens / (makespan_ms / 1000.0),
        ttft_ms=percentiles([r.ttft_ms for r in rows]),
        total_latency_ms=percentiles([r.total_latency_ms for r in rows]),
        queue_wait_ms=percentiles([r.queue_wait_ms for r in rows]),
        cache_hit_rate=sum(1 for r in rows if r.cache_hit) / completed,
        mean_cache_overlap=sum(r.cache_overlap for r in rows) / completed,
        ttft_ms_by_cache=by_cache,
        prefill_tokens_total=prefill_total,
        prefill_tokens_saved=prompt_total - prefill_total,
        prefill_saved_fraction=(prompt_total - prefill_total) / prompt_total if prompt_total else 0,
        worker_request_counts=counts,
        worker_imbalance=max(counts.values()) / mean_per_worker if mean_per_worker else 0.0,
        kv=kv,
    )


def format_report(run: BenchmarkRun) -> str:
    m = run.metrics
    lines = [
        f"Run ID:               {run.run_id}",
        f"Mode:                 {run.mode}"
        + (
            f" against {run.gateway_url} ({', '.join(sorted(set(run.worker_backends.values())))})"
            if run.mode == "live"
            else ""
        ),
        f"Workload:             {run.workload.name}",
        f"Policy:               {run.policy}",
        f"Seed:                 {run.seed}",
        "",
        f"Requests:             {m.requests}",
        f"Completed:            {m.completed}",
        f"Makespan:             {m.makespan_ms / 1000:.2f} s",
        f"Throughput:           {m.requests_per_s:.2f} req/s, {m.output_tokens_per_s:.0f} tok/s",
        f"TTFT p50:             {m.ttft_ms.p50:.1f} ms",
        f"TTFT p95:             {m.ttft_ms.p95:.1f} ms",
        f"TTFT p99:             {m.ttft_ms.p99:.1f} ms",
        f"Latency p50:          {m.total_latency_ms.p50:.1f} ms",
        f"Latency p99:          {m.total_latency_ms.p99:.1f} ms",
        f"Queue wait mean:      {m.queue_wait_ms.mean:.1f} ms",
        f"Cache hit rate:       {m.cache_hit_rate:.1%}",
        f"Mean prefix overlap:  {m.mean_cache_overlap:.3f}",
        *[
            f"TTFT p50 ({label}):".ljust(22) + f"{p.p50:.1f} ms  (p99 {p.p99:.1f} ms)"
            for label, p in m.ttft_ms_by_cache.items()
        ],
        f"Prefill tokens saved: {m.prefill_tokens_saved:,} ({m.prefill_saved_fraction:.1%})",
        f"Worker imbalance:     {m.worker_imbalance:.2f}  {m.worker_request_counts}",
        f"KV evictions:         {sum(k.evictions for k in m.kv)}",
    ]
    if run.policy_parameters:
        lines.append(f"Policy parameters:    {run.policy_parameters}")
    if run.worker_config.get("queue_model") == "batched":
        lines.append(
            f"Worker model:         batched (max batch {run.worker_config['max_batch_size']}, "
            f"KV budget {run.worker_config['kv_budget_tokens']} tokens)"
        )
    return "\n".join(lines)


def write_outputs(
    run: BenchmarkRun, rows: list[RequestRow], out_dir: Path, *, stem: str | None = None
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        prefix = "live_" if run.mode == "live" else ""
        stem = f"{prefix}{run.workload.name}_{run.policy}_seed{run.seed}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    json_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RequestRow.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
    return json_path, csv_path
