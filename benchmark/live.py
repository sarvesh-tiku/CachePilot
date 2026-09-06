from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from benchmark.report import BenchmarkRun, RequestRow, aggregate, format_report, write_outputs
from benchmark.workloads import BenchmarkRequest, WorkloadSpec, generate, load_workload
from cachepilot.kv.directory import WorkerCacheSummary
from cachepilot.scheduler.policies import POLICY_NAMES

DEFAULT_OUT_DIR = Path(__file__).parent / "results" / "live"


@dataclass
class Observation:
    request: BenchmarkRequest
    request_id: str
    worker_id: str
    sent_s: float
    first_token_s: float | None
    done_s: float
    tokens: int


async def run_live(
    *,
    client: httpx.AsyncClient,
    spec: WorkloadSpec,
    policy: str,
    requests: int,
    seed: int,
    rate_rps: float | None = None,
    prefix_tokens: int | None = None,
    output_tokens: int | None = None,
    shared_fraction: float | None = None,
    concurrency: int = 16,
    warmup: int = 1,
) -> tuple[BenchmarkRun, list[RequestRow]]:
    """Replay a seeded workload against a *running* gateway over HTTP with the real clock.

    TTFT and total latency are what a client would see: measured from request
    send to the first content delta / end of the SSE stream. Cache overlap and
    hit/miss come from the gateway's own record of each request.
    """
    overrides: dict[str, object] = {}
    if prefix_tokens is not None:
        overrides["prefix_tokens"] = prefix_tokens
    if output_tokens is not None:
        overrides["output_tokens"] = output_tokens
    if shared_fraction is not None:
        overrides["shared_fraction"] = shared_fraction
    if rate_rps is not None:
        overrides["arrival"] = spec.arrival.model_copy(update={"rate_rps": rate_rps})
    spec = spec.model_copy(update=overrides)

    stream = generate(spec, seed=seed, count=requests)
    for warm in generate(spec, seed=seed + 1_000_003, count=warmup):
        await _send(client, warm, policy)

    started = time.monotonic()
    semaphore = asyncio.Semaphore(concurrency)

    async def one(req: BenchmarkRequest) -> Observation:
        delay = req.arrival_ms / 1000.0 - (time.monotonic() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            return await _send(client, req, policy)

    observations = await asyncio.gather(*(one(req) for req in stream))

    rows = []
    for obs in observations:
        view = (await client.get(f"/api/requests/{obs.request_id}")).json()
        result = view["result"]
        chosen = next(c for c in view["decision"]["candidates"] if c["worker_id"] == obs.worker_id)
        pending = int(chosen["queue_depth"]) + int(chosen.get("active_requests", 0))
        prompt_tokens = int(view["request"]["prompt_tokens_estimate"])
        overlap = float(result["cache_overlap"])
        first = obs.first_token_s if obs.first_token_s is not None else obs.done_s
        rows.append(
            RequestRow(
                index=obs.request.index,
                arrival_ms=(obs.sent_s - started) * 1000.0,
                completed_ms=(obs.done_s - started) * 1000.0,
                prefix_id=obs.request.prefix_id,
                worker_id=obs.worker_id,
                prompt_tokens=prompt_tokens,
                output_tokens=int(result["output_tokens"]),
                effective_prefill_tokens=round(prompt_tokens * (1.0 - overlap)),
                cache_overlap=overlap,
                cache_hit=bool(result["cache_hit"]),
                queue_wait_ms=float(result["queue_wait_ms"]),
                ttft_ms=(first - obs.sent_s) * 1000.0,
                total_latency_ms=(obs.done_s - obs.sent_s) * 1000.0,
                pending_at_decision=pending,
            )
        )
    rows.sort(key=lambda r: r.index)

    workers = (await client.get("/api/workers")).json()
    caches = [WorkerCacheSummary.model_validate(c) for c in (await client.get("/api/kv")).json()]
    policies = {p["name"]: p["parameters"] for p in (await client.get("/api/policies")).json()}
    metrics = aggregate(
        rows, requested=requests, worker_ids=[w["worker_id"] for w in workers], kv=caches
    )
    run = BenchmarkRun(
        workload=spec,
        policy=policy,
        policy_parameters=policies.get(policy, {}),
        seed=seed,
        requested=requests,
        workers=len(workers),
        metrics=metrics,
        mode="live",
        gateway_url=str(client.base_url).rstrip("/"),
        worker_backends={w["worker_id"]: w["backend"] for w in workers},
    )
    return run, rows


async def _send(client: httpx.AsyncClient, req: BenchmarkRequest, policy: str) -> Observation:
    payload = {
        "model": "benchmark",
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
        "stream": True,
    }
    sent = time.monotonic()
    first: float | None = None
    tokens = 0
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload, headers={"X-CachePilot-Policy": policy}
    ) as resp:
        resp.raise_for_status()
        request_id = resp.headers["X-CachePilot-Request-Id"]
        worker_id = resp.headers["X-CachePilot-Worker"]
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                if (choice.get("delta") or {}).get("content"):
                    if first is None:
                        first = time.monotonic()
                    tokens += 1
    return Observation(req, request_id, worker_id, sent, first, time.monotonic(), tokens)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.live",
        description="Replay a seeded workload against a running gateway; report observed latency.",
    )
    parser.add_argument("--gateway", default="http://localhost:8000")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--policy", required=True, choices=POLICY_NAMES)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=None, help="override arrivals (req/s)")
    parser.add_argument("--prefix-tokens", type=int, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--shared-fraction", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


async def _main(args: argparse.Namespace) -> tuple[BenchmarkRun, list[RequestRow]]:
    async with httpx.AsyncClient(base_url=args.gateway, timeout=args.timeout) as client:
        return await run_live(
            client=client,
            spec=load_workload(args.workload),
            policy=args.policy,
            requests=args.requests,
            seed=args.seed,
            rate_rps=args.rate,
            prefix_tokens=args.prefix_tokens,
            output_tokens=args.output_tokens,
            shared_fraction=args.shared_fraction,
            concurrency=args.concurrency,
            warmup=args.warmup,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    run, rows = asyncio.run(_main(args))
    json_path, csv_path = write_outputs(run, rows, args.out_dir)
    if not args.quiet:
        print(format_report(run))
        print()
    print(f"Wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
