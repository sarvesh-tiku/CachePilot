from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict, replace
from pathlib import Path

from benchmark.report import BenchmarkRun, RequestRow, aggregate, format_report, write_outputs
from benchmark.workloads import BenchmarkRequest, WorkloadSpec, generate, load_workload
from cachepilot.core.clock import EventClock
from cachepilot.core.models import InferenceRequest
from cachepilot.core.tokens import estimate_prompt_tokens
from cachepilot.gateway.factory import build_stack
from cachepilot.gateway.service import InferenceGateway
from cachepilot.persistence.repositories import InMemoryRunStore
from cachepilot.scheduler.kv_aware import KVAwareWeights
from cachepilot.scheduler.policies import POLICY_NAMES
from cachepilot.scheduler.scoring import TTFTEstimator
from cachepilot.workers.base import DoneEvent
from cachepilot.workers.simulated import SimulatedWorkerConfig

DEFAULT_OUT_DIR = Path(__file__).parent / "results"
DEFAULT_CHUNK_TOKENS = 128
DEFAULT_BYTES_PER_TOKEN = 524_288


async def run_benchmark(
    *,
    spec: WorkloadSpec,
    policy: str,
    requests: int,
    seed: int,
    workers: int = 3,
    weights: KVAwareWeights | None = None,
    estimator: TTFTEstimator | None = None,
    worker_config: SimulatedWorkerConfig | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    bytes_per_token: int = DEFAULT_BYTES_PER_TOKEN,
) -> tuple[BenchmarkRun, list[RequestRow]]:
    """Replay a seeded workload through the real gateway on a discrete-event clock.

    The workload's per-worker KV capacity, when it sets one, overrides the
    worker config's; everything else about the worker (queue model, batch
    size, cost coefficients) comes from `worker_config`.
    """
    clock = EventClock()
    store = InMemoryRunStore()
    worker_config = worker_config or SimulatedWorkerConfig()
    if spec.kv_capacity_bytes_per_worker is not None:
        worker_config = replace(worker_config, kv_capacity_bytes=spec.kv_capacity_bytes_per_worker)

    stack = await build_stack(
        simulated_workers=workers,
        clock=clock,
        store=store,
        default_policy=policy,
        weights=weights or KVAwareWeights(),
        estimator=estimator,
        chunk_tokens=chunk_tokens,
        bytes_per_token=bytes_per_token,
        worker_config=worker_config,
    )

    stream = generate(spec, seed=seed, count=requests)
    rows: list[RequestRow] = []
    await clock.run(_drive(stack.gateway, clock, req, policy, rows) for req in stream)
    rows.sort(key=lambda r: r.index)

    metrics = aggregate(
        rows,
        requested=requests,
        worker_ids=[w.worker_id for w in stack.workers],
        kv=stack.kv_directory.summaries(),
    )
    run = BenchmarkRun(
        workload=spec,
        policy=policy,
        policy_parameters=stack.gateway.describe_policies()[policy],
        seed=seed,
        requested=requests,
        workers=workers,
        metrics=metrics,
        worker_config=asdict(worker_config),
    )
    return run, rows


async def _drive(
    gateway: InferenceGateway,
    clock: EventClock,
    req: BenchmarkRequest,
    policy: str,
    rows: list[RequestRow],
) -> None:
    await clock.sleep(req.arrival_ms / 1000.0)
    inference = InferenceRequest(
        model="sim-model",
        messages=req.messages,
        prompt_tokens_estimate=estimate_prompt_tokens(req.messages),
        max_tokens=req.max_tokens,
    )
    submission = await gateway.submit(inference, policy=policy)
    done: DoneEvent | None = None
    async for event in submission.events:
        if isinstance(event, DoneEvent):
            done = event
    if done is None:
        raise RuntimeError(f"request {req.index} ended without a DoneEvent")

    prompt_tokens = inference.prompt_tokens_estimate
    decision = submission.decision
    chosen = next(c for c in decision.candidates if c.worker_id == decision.selected_worker_id)
    rows.append(
        RequestRow(
            index=req.index,
            arrival_ms=req.arrival_ms,
            completed_ms=clock.now_ms,
            prefix_id=req.prefix_id,
            worker_id=submission.decision.selected_worker_id,
            prompt_tokens=prompt_tokens,
            output_tokens=done.output_tokens,
            effective_prefill_tokens=round(prompt_tokens * (1.0 - submission.cache_overlap)),
            cache_overlap=submission.cache_overlap,
            cache_hit=submission.cache_overlap > 0.0,
            queue_wait_ms=done.queue_wait_ms,
            ttft_ms=done.ttft_ms,
            total_latency_ms=done.total_latency_ms,
            pending_at_decision=chosen.queue_depth + chosen.active_requests,
        )
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.runner",
        description="Replay a seeded workload under one routing policy and report metrics.",
    )
    parser.add_argument("--workload", required=True, help="workload name or path to a spec JSON")
    parser.add_argument("--policy", required=True, choices=POLICY_NAMES)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--rate", type=float, default=None, help="override Poisson arrivals (req/s)"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha", type=float, default=None, help="kv_aware cache overlap weight")
    parser.add_argument("--beta", type=float, default=None, help="kv_aware normalized load weight")
    parser.add_argument("--gamma", type=float, default=None, help="kv_aware KV pressure weight")
    parser.add_argument("--delta", type=float, default=None, help="kv_aware predicted TTFT weight")
    add_worker_args(parser)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def add_worker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--queue-model",
        choices=("linear", "batched"),
        default="linear",
        help="linear: fixed penalty per pending request; batched: continuous batching, saturates",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=SimulatedWorkerConfig().max_batch_size
    )
    parser.add_argument(
        "--kv-budget-tokens", type=int, default=SimulatedWorkerConfig().kv_budget_tokens
    )


def worker_config_from_args(args: argparse.Namespace) -> SimulatedWorkerConfig:
    return SimulatedWorkerConfig(
        queue_model=args.queue_model,
        max_batch_size=args.max_batch_size,
        kv_budget_tokens=args.kv_budget_tokens,
    )


def with_rate(spec: WorkloadSpec, rate_rps: float | None) -> WorkloadSpec:
    """Override a Poisson workload's arrival rate; bursty arrivals keep their own timing."""
    if rate_rps is None or spec.arrival.kind != "poisson":
        return spec
    return spec.model_copy(
        update={"arrival": spec.arrival.model_copy(update={"rate_rps": rate_rps})}
    )


def output_stem(run: BenchmarkRun, *, rate_tag: bool = False) -> str | None:
    """Batched runs get their own stem so they never overwrite linear-model results."""
    cfg = run.worker_config
    if cfg.get("queue_model") != "batched":
        return None
    rate = f"_r{run.workload.arrival.rate_rps:g}" if rate_tag else ""
    return f"batched{cfg['max_batch_size']}_{run.workload.name}{rate}_{run.policy}_seed{run.seed}"


def _weights_from_args(args: argparse.Namespace) -> KVAwareWeights:
    defaults = KVAwareWeights()
    return KVAwareWeights(
        alpha=defaults.alpha if args.alpha is None else args.alpha,
        beta=defaults.beta if args.beta is None else args.beta,
        gamma=defaults.gamma if args.gamma is None else args.gamma,
        delta=defaults.delta if args.delta is None else args.delta,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    spec = with_rate(load_workload(args.workload), args.rate)
    run, rows = asyncio.run(
        run_benchmark(
            spec=spec,
            policy=args.policy,
            requests=args.requests,
            seed=args.seed,
            workers=args.workers,
            weights=_weights_from_args(args),
            worker_config=worker_config_from_args(args),
        )
    )
    json_path, csv_path = write_outputs(
        run, rows, args.out_dir, stem=output_stem(run, rate_tag=args.rate is not None)
    )
    if not args.quiet:
        print(format_report(run))
        print()
    print(f"Wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
