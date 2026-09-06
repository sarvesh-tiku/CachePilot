from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

from benchmark.analysis import _fmt, metric_values
from benchmark.report import BenchmarkRun, write_outputs
from benchmark.runner import (
    add_worker_args,
    output_stem,
    run_benchmark,
    with_rate,
    worker_config_from_args,
)
from benchmark.sweep import load_sweep_runs
from benchmark.workloads import WorkloadSpec, load_workload
from cachepilot.scheduler.policies import POLICY_NAMES
from cachepilot.workers.simulated import SimulatedWorkerConfig

DEFAULT_OUT_DIR = Path(__file__).parent / "results" / "batched"
DEFAULT_RATES = (6.0, 9.0, 12.0, 15.0, 18.0, 24.0)
SCAN_COLUMNS = [
    ("TTFT p50", "ttft_p50"),
    ("TTFT p99", "ttft_p99"),
    ("Queue mean", "queue_mean"),
    ("Hit rate", "hit_rate"),
    ("Imbalance", "imbalance"),
    ("Throughput", "throughput"),
]


async def run_loadscan(
    *,
    spec: WorkloadSpec,
    rates: tuple[float, ...],
    policies: tuple[str, ...] = POLICY_NAMES,
    requests: int,
    seed: int,
    workers: int = 3,
    worker_config: SimulatedWorkerConfig | None = None,
    out_dir: Path | None = None,
) -> list[BenchmarkRun]:
    """Replay one Poisson workload at several arrival rates under every policy."""
    runs = []
    for rate in rates:
        rated = with_rate(spec, rate)
        for policy in policies:
            run, rows = await run_benchmark(
                spec=rated,
                policy=policy,
                requests=requests,
                seed=seed,
                workers=workers,
                worker_config=worker_config,
            )
            if out_dir is not None:
                write_outputs(run, rows, out_dir, stem=output_stem(run, rate_tag=True))
            runs.append(run)
    return runs


def scan_table(runs: list[BenchmarkRun]) -> str:
    """Markdown per workload: one row per (rate, policy); deltas vs round_robin at the same rate."""
    by_workload: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for run in runs:
        by_workload[run.workload.name].append(run)

    sections = []
    for workload, group in sorted(by_workload.items()):
        first = group[0]
        cfg = first.worker_config
        model = (
            f"batched workers, max batch {cfg['max_batch_size']}"
            if cfg.get("queue_model") == "batched"
            else "linear workers"
        )
        header = ["Offered load", "Policy", *[label for label, _ in SCAN_COLUMNS]]
        lines = [
            f"#### {workload} ({first.requested} requests, seed {first.seed}, "
            f"{first.workers} {model})",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        by_rate: dict[float, list[BenchmarkRun]] = defaultdict(list)
        for run in group:
            by_rate[run.workload.arrival.rate_rps].append(run)
        for rate, at_rate in sorted(by_rate.items()):
            at_rate.sort(key=lambda r: POLICY_NAMES.index(r.policy))
            base = next((r for r in at_rate if r.policy == "round_robin"), None)
            for run in at_rate:
                values = _scan_values(run)
                cells = [f"{rate:g} req/s", run.policy]
                for _, key in SCAN_COLUMNS:
                    cell = _scan_fmt(key, values[key])
                    if base is not None and run is not base:
                        b = _scan_values(base)[key]
                        if key == "hit_rate":
                            cell += f" ({(values[key] - b) * 100:+.1f} pp)"
                        elif b:
                            cell += f" ({(values[key] - b) / b:+.1%})"
                    cells.append(cell)
                lines.append("| " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def _scan_values(run: BenchmarkRun) -> dict[str, float]:
    values = metric_values(run)
    values["throughput"] = run.metrics.requests_per_s
    return values


def _scan_fmt(key: str, value: float) -> str:
    if key == "throughput":
        return f"{value:.1f} req/s"
    return _fmt(key, value)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.loadscan",
        description=(
            "Replay a Poisson workload at increasing arrival rates under every policy "
            "to find where routing decisions change the saturation point."
        ),
    )
    parser.add_argument("--workload", action="append", required=False)
    parser.add_argument("--rates", default=",".join(f"{r:g}" for r in DEFAULT_RATES))
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    add_worker_args(parser)
    parser.set_defaults(queue_model="batched")
    parser.add_argument("--table", nargs="*", type=Path, help="tabulate these run JSONs and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.table is not None:
        print(scan_table(load_sweep_runs(args.table)), end="")
        return 0
    if not args.workload:
        print("give --workload NAME (repeatable) or --table", file=sys.stderr)
        return 2
    rates = tuple(float(r) for r in args.rates.split(","))
    runs: list[BenchmarkRun] = []
    for name in args.workload:
        spec = load_workload(name)
        if spec.arrival.kind != "poisson":
            print(f"{spec.name}: only Poisson workloads can be rate-scanned", file=sys.stderr)
            return 2
        runs += asyncio.run(
            run_loadscan(
                spec=spec,
                rates=rates,
                requests=args.requests,
                seed=args.seed,
                workers=args.workers,
                worker_config=worker_config_from_args(args),
                out_dir=args.out_dir,
            )
        )
    print(scan_table(runs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
