from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from benchmark.report import RequestRow
from cachepilot.scheduler.scoring import FitReport, TTFTEstimator, TTFTSample

ENV_NAMES = {
    "queue_wait_ms_per_pending": "CACHEPILOT_TTFT_QUEUE_WAIT_MS_PER_PENDING",
    "prefill_ms_per_token": "CACHEPILOT_TTFT_PREFILL_MS_PER_TOKEN",
    "fixed_overhead_ms": "CACHEPILOT_TTFT_FIXED_OVERHEAD_MS",
}


def samples_from_rows(rows: Sequence[RequestRow]) -> list[TTFTSample]:
    """Per-request rows from `benchmark.runner` / `benchmark.live` CSVs."""
    return [
        TTFTSample(
            pending=row.pending_at_decision,
            prompt_tokens=row.prompt_tokens,
            cache_overlap=row.cache_overlap,
            ttft_ms=row.ttft_ms,
        )
        for row in rows
    ]


def samples_from_views(views: Sequence[dict[str, Any]]) -> list[TTFTSample]:
    """Request views from `GET /api/requests`: the persisted decision plus the observed result.

    Uses the queue depth the scheduler *saw* for the worker it chose, so the
    sample reflects the information a live estimate would have had.
    """
    samples = []
    for view in views:
        decision, result = view.get("decision"), view.get("result")
        if not decision or not result:
            continue
        chosen = next(
            (c for c in decision["candidates"] if c["worker_id"] == decision["selected_worker_id"]),
            None,
        )
        if chosen is None:
            continue
        samples.append(
            TTFTSample(
                pending=int(chosen["queue_depth"]) + int(chosen.get("active_requests", 0)),
                prompt_tokens=int(view["request"]["prompt_tokens_estimate"]),
                cache_overlap=float(result["cache_overlap"]),
                ttft_ms=float(result["ttft_ms"]),
            )
        )
    return samples


def read_rows(path: Path) -> list[RequestRow]:
    with path.open(newline="") as handle:
        return [RequestRow.model_validate(record) for record in csv.DictReader(handle)]


async def fetch_views(
    gateway: str, *, worker_id: str | None, policy: str | None, limit: int
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if worker_id:
        params["worker_id"] = worker_id
    if policy:
        params["policy"] = policy
    async with httpx.AsyncClient(base_url=gateway, timeout=60.0) as client:
        response = await client.get("/api/requests", params=params)
        response.raise_for_status()
        views: list[dict[str, Any]] = response.json()
        return views


def holdout_reports(
    baseline: TTFTEstimator, sources: dict[str, list[TTFTSample]]
) -> dict[str, FitReport]:
    """Leave-one-source-out: fit on every other source, score on the held-out one."""
    reports = {}
    for name, held in sources.items():
        training = [s for other, group in sources.items() if other != name for s in group]
        if len(training) < 2 or not held:
            continue
        reports[name] = baseline.fit(training).evaluate(held)
    return reports


def format_fit(
    baseline: TTFTEstimator,
    fitted: TTFTEstimator,
    samples: Sequence[TTFTSample],
    holdout: dict[str, FitReport],
) -> str:
    before, after = baseline.evaluate(samples), fitted.evaluate(samples)
    pending_values = {s.pending for s in samples}
    token_values = {round(s.uncached_tokens) for s in samples}
    lines = [
        f"Samples:                 {len(samples)}",
        f"Distinct pending values: {len(pending_values)}"
        + ("  (queue coefficient not identifiable; kept)" if len(pending_values) < 2 else ""),
        f"Distinct uncached sizes: {len(token_values)}"
        + ("  (prefill coefficient not identifiable; kept)" if len(token_values) < 2 else ""),
        "",
        f"{'Coefficient':<28}{'current':>12}{'fitted':>12}",
    ]
    for key, value in baseline.parameters().items():
        lines.append(f"{key:<28}{value:>12.4f}{fitted.parameters()[key]:>12.4f}")
    lines += [
        "",
        f"{'In-sample':<28}{'current':>12}{'fitted':>12}",
        f"{'MAE (ms)':<28}{before.mae_ms:>12.1f}{after.mae_ms:>12.1f}",
        f"{'bias (ms)':<28}{before.bias_ms:>12.1f}{after.bias_ms:>12.1f}",
        f"{'R²':<28}{before.r2:>12.3f}{after.r2:>12.3f}",
    ]
    if holdout:
        lines += ["", "Leave-one-source-out (fit on the others, score on this one):"]
        for name, report in holdout.items():
            lines.append(
                f"  {name:<40} n={report.samples:<5} MAE {report.mae_ms:.1f} ms  "
                f"bias {report.bias_ms:+.1f} ms  R² {report.r2:.3f}"
            )
    lines += ["", "To route with the fitted estimator:"]
    for key, value in fitted.parameters().items():
        lines.append(f"  export {ENV_NAMES[key]}={value:.6g}")
    return "\n".join(lines)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.fit",
        description=(
            "Fit the TTFT estimator (queue wait + uncached prefill + overhead) to observed "
            "timings from benchmark CSVs and/or a running gateway's request history."
        ),
    )
    parser.add_argument("--csv", nargs="*", type=Path, default=[], help="per-request CSVs")
    parser.add_argument("--gateway", default=None, help="e.g. http://localhost:8000")
    parser.add_argument("--worker", default=None, help="only requests routed to this worker")
    parser.add_argument("--policy", default=None, help="only requests routed by this policy")
    parser.add_argument("--limit", type=int, default=1000, help="request history to fetch")
    defaults = TTFTEstimator()
    parser.add_argument("--queue-wait", type=float, default=defaults.queue_wait_ms_per_pending)
    parser.add_argument("--prefill", type=float, default=defaults.prefill_ms_per_token)
    parser.add_argument("--overhead", type=float, default=defaults.fixed_overhead_ms)
    parser.add_argument("--write", type=Path, default=None, help="save fitted coefficients as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not args.csv and not args.gateway:
        print("give at least one --csv file or a --gateway URL", file=sys.stderr)
        return 2

    sources: dict[str, list[TTFTSample]] = {}
    for path in args.csv:
        sources[path.name] = samples_from_rows(read_rows(path))
    if args.gateway:
        views = asyncio.run(
            fetch_views(args.gateway, worker_id=args.worker, policy=args.policy, limit=args.limit)
        )
        sources[args.gateway] = samples_from_views(views)

    samples = [s for group in sources.values() for s in group]
    if len(samples) < 2:
        print(f"only {len(samples)} usable samples; need at least 2", file=sys.stderr)
        return 1

    baseline = TTFTEstimator(
        queue_wait_ms_per_pending=args.queue_wait,
        prefill_ms_per_token=args.prefill,
        fixed_overhead_ms=args.overhead,
    )
    fitted = baseline.fit(samples)
    holdout = holdout_reports(baseline, sources) if len(sources) > 1 else {}
    print(format_fit(baseline, fitted, samples, holdout))

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(
                {
                    "estimator": fitted.parameters(),
                    "baseline": baseline.parameters(),
                    "samples": len(samples),
                    "sources": {name: len(group) for name, group in sources.items()},
                    "in_sample": fitted.evaluate(samples).__dict__,
                    "baseline_in_sample": baseline.evaluate(samples).__dict__,
                    "holdout": {name: report.__dict__ for name, report in holdout.items()},
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nWrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
