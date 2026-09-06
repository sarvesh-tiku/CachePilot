from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from benchmark.report import BenchmarkRun, write_outputs
from benchmark.runner import DEFAULT_OUT_DIR, run_benchmark
from benchmark.workloads import WORKLOAD_DIR, WorkloadSpec, load_workload
from cachepilot.core.config import Settings
from cachepilot.scheduler.kv_aware import KVAwareWeights
from cachepilot.scheduler.policies import POLICY_NAMES
from cachepilot.scheduler.scoring import TTFTEstimator

router = APIRouter(prefix="/api/benchmark-runs", tags=["benchmarks"])

_run_lock = asyncio.Lock()


class BenchmarkRunRequest(BaseModel):
    workload: str
    policies: list[str] = Field(default_factory=lambda: list(POLICY_NAMES))
    requests: int = Field(default=300, ge=1, le=5000)
    seed: int = 42
    workers: int = Field(default=3, ge=1, le=16)


def _results_dir(request: Request) -> Path:
    settings: Settings = request.app.state.settings
    return (
        Path(settings.benchmark_results_dir) if settings.benchmark_results_dir else DEFAULT_OUT_DIR
    )


def _load_all(results_dir: Path) -> list[BenchmarkRun]:
    runs = [
        BenchmarkRun.model_validate(json.loads(p.read_text())) for p in results_dir.glob("*.json")
    ]
    return sorted(runs, key=lambda r: r.created_at, reverse=True)


@router.get("/workloads", response_model=list[WorkloadSpec])
async def list_workloads() -> list[WorkloadSpec]:
    return [load_workload(str(p)) for p in sorted(WORKLOAD_DIR.glob("*.json"))]


@router.get("", response_model=list[BenchmarkRun])
async def list_runs(request: Request) -> list[BenchmarkRun]:
    return _load_all(_results_dir(request))


@router.get("/{run_id}", response_model=BenchmarkRun)
async def get_run(run_id: uuid.UUID, request: Request) -> BenchmarkRun:
    for run in _load_all(_results_dir(request)):
        if run.run_id == run_id:
            return run
    raise HTTPException(status_code=404, detail=f"benchmark run not found: {run_id}")


@router.post("", response_model=list[BenchmarkRun], status_code=201)
async def start_runs(body: BenchmarkRunRequest, request: Request) -> list[BenchmarkRun]:
    unknown = [p for p in body.policies if p not in POLICY_NAMES]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown policies: {unknown}")
    try:
        spec = load_workload(body.workload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    settings: Settings = request.app.state.settings
    weights = KVAwareWeights(
        alpha=settings.kv_aware_alpha,
        beta=settings.kv_aware_beta,
        gamma=settings.kv_aware_gamma,
        delta=settings.kv_aware_delta,
    )
    estimator: TTFTEstimator = request.app.state.estimator
    out_dir = _results_dir(request)

    def run_all() -> list[BenchmarkRun]:
        # Each benchmark drives its own EventClock, which must own its event loop.
        runs = []
        for policy in body.policies:
            run, rows = asyncio.run(
                run_benchmark(
                    spec=spec,
                    policy=policy,
                    requests=body.requests,
                    seed=body.seed,
                    workers=body.workers,
                    weights=weights,
                    estimator=estimator,
                )
            )
            write_outputs(run, rows, out_dir)
            runs.append(run)
        return runs

    async with _run_lock:
        return await asyncio.to_thread(run_all)
