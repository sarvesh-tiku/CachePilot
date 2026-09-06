from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from benchmark.analysis import COLUMNS, _delta, _fmt, metric_values
from benchmark.report import BenchmarkRun, write_outputs
from benchmark.runner import add_worker_args, run_benchmark, with_rate, worker_config_from_args
from benchmark.workloads import WorkloadSpec, load_workload
from cachepilot.scheduler.kv_aware import KVAwareWeights
from cachepilot.scheduler.scoring import TTFTEstimator
from cachepilot.workers.simulated import SimulatedWorkerConfig

DEFAULT_OUT_DIR = Path(__file__).parent / "results" / "sweep"
DEFAULT_WEIGHTS = KVAwareWeights()
WEIGHT_NAMES = ("alpha", "beta", "gamma", "delta")
REFERENCE_POLICIES = ("round_robin", "least_loaded")

# One-at-a-time grid around the defaults (2.0 / 0.8 / 0.4 / 0.002). Each value replaces
# exactly one weight; the other three stay at their defaults.
DEFAULT_GRID: dict[str, list[float]] = {
    "alpha": [0.0, 0.5, 1.0, 4.0, 8.0],
    "beta": [0.0, 0.2, 2.0, 4.0],
    "gamma": [0.0, 2.0],
    "delta": [0.0, 0.01, 0.05],
}

SWEEP_COLUMNS = [
    (label, key)
    for label, key in COLUMNS
    if key in {"ttft_p50", "ttft_p99", "queue_mean", "hit_rate", "imbalance", "evictions"}
]


@dataclass(frozen=True)
class SweepPoint:
    label: str
    weights: KVAwareWeights

    @property
    def slug(self) -> str:
        return self.label.replace("=", "").replace(".", "p").replace(" ", "_")


def one_at_a_time(
    base: KVAwareWeights, grid: dict[str, list[float]] | None = None
) -> list[SweepPoint]:
    """`default` plus one point per (weight, value) in the grid, skipping the default value."""
    grid = DEFAULT_GRID if grid is None else grid
    points = [SweepPoint("default", base)]
    for name in WEIGHT_NAMES:
        for value in grid.get(name, []):
            if value == getattr(base, name):
                continue
            points.append(SweepPoint(f"{name}={value:g}", replace(base, **{name: value})))
    return points


def parse_point(text: str, base: KVAwareWeights = DEFAULT_WEIGHTS) -> SweepPoint:
    """`alpha=2,beta=0,delta=0` -> a point with those weights changed together."""
    changes: dict[str, float] = {}
    for item in text.split(","):
        name, _, raw = item.partition("=")
        if name not in WEIGHT_NAMES or not raw:
            raise argparse.ArgumentTypeError(f"expected <weight>=<value>[,...]: {text!r}")
        changes[name] = float(raw)
    weights = replace(base, **changes)
    return SweepPoint(label_for(weights.__dict__, base), weights)


def label_for(params: dict[str, float], base: KVAwareWeights = DEFAULT_WEIGHTS) -> str:
    """Recover a sweep point's label from the weights a run recorded."""
    changed = [
        f"{name}={params[name]:g}"
        for name in WEIGHT_NAMES
        if name in params and params[name] != getattr(base, name)
    ]
    return " ".join(changed) if changed else "default"


async def run_sweep(
    *,
    spec: WorkloadSpec,
    points: list[SweepPoint],
    requests: int,
    seed: int,
    workers: int = 3,
    estimator: TTFTEstimator | None = None,
    worker_config: SimulatedWorkerConfig | None = None,
    out_dir: Path | None = None,
    reference_policies: tuple[str, ...] = REFERENCE_POLICIES,
) -> list[BenchmarkRun]:
    """Replay one workload under every weight setting (and the reference policies)."""
    runs = []
    cfg = worker_config or SimulatedWorkerConfig()
    prefix = f"batched{cfg.max_batch_size}_" if cfg.queue_model == "batched" else ""
    if cfg.queue_model == "batched":
        prefix += f"r{spec.arrival.rate_rps:g}_"
    for policy in reference_policies:
        run, rows = await run_benchmark(
            spec=spec,
            policy=policy,
            requests=requests,
            seed=seed,
            workers=workers,
            estimator=estimator,
            worker_config=worker_config,
        )
        if out_dir is not None:
            write_outputs(run, rows, out_dir, stem=f"{prefix}{spec.name}_{policy}_seed{seed}")
        runs.append(run)
    for point in points:
        run, rows = await run_benchmark(
            spec=spec,
            policy="kv_aware",
            requests=requests,
            seed=seed,
            workers=workers,
            weights=point.weights,
            estimator=estimator,
            worker_config=worker_config,
        )
        if out_dir is not None:
            stem = f"{prefix}{spec.name}_kv_aware_{point.slug}_seed{seed}"
            write_outputs(run, rows, out_dir, stem=stem)
        runs.append(run)
    return runs


def sweep_table(runs: list[BenchmarkRun], base: KVAwareWeights = DEFAULT_WEIGHTS) -> str:
    """Markdown per workload: one row per weight setting, deltas against the default weights."""
    by_workload: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for run in runs:
        key = run.workload.name
        if run.worker_config.get("queue_model") == "batched":
            key += f" @ {run.workload.arrival.rate_rps:g} req/s"
        by_workload[key].append(run)

    sections = []

    def section_order(item: tuple[str, list[BenchmarkRun]]) -> tuple[str, float]:
        key, group = item
        return (key.split(" @ ")[0], group[0].workload.arrival.rate_rps)

    for _, group in sorted(by_workload.items(), key=section_order):
        workload = group[0].workload.name
        rows = []
        for run in group:
            is_kv = run.policy == "kv_aware"
            label = label_for(run.policy_parameters, base) if is_kv else run.policy
            rows.append((label, run))
        baseline = next((r for label, r in rows if label == "default"), None)
        base_values = metric_values(baseline) if baseline is not None else None

        def order(item: tuple[str, BenchmarkRun]) -> tuple[int, int, float, str]:
            label, run = item
            if run.policy != "kv_aware":
                return (0, 0, 0.0, label)
            if label == "default":
                return (1, 0, 0.0, label)
            if " " in label:  # combined point: after the one-at-a-time rows
                return (3, 0, 0.0, label)
            name, _, value = label.partition("=")
            return (2, WEIGHT_NAMES.index(name) if name in WEIGHT_NAMES else 9, float(value), label)

        header = ["Setting", *[label for label, _ in SWEEP_COLUMNS]]
        first = group[0]
        cfg = first.worker_config
        model = (
            f", batched workers (max batch {cfg['max_batch_size']}) "
            f"at {first.workload.arrival.rate_rps:g} req/s"
            if cfg.get("queue_model") == "batched"
            else ""
        )
        lines = [
            f"#### {workload} "
            f"({first.requested} requests, seed {first.seed}, {first.workers} workers{model})",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        for label, run in sorted(rows, key=order):
            values = metric_values(run)
            name = label if run.policy == "kv_aware" else f"{label} (reference)"
            cells = [name if label != "default" else "**default** (α=2 β=0.8 γ=0.4 δ=0.002)"]
            for _, key in SWEEP_COLUMNS:
                cell = _fmt(key, values[key])
                if base_values is not None and run is not baseline and key != "evictions":
                    cell += f" ({_delta(key, values[key], base_values[key])})"
                cells.append(cell)
            lines.append("| " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def load_sweep_runs(paths: list[Path]) -> list[BenchmarkRun]:
    """Load run JSONs, ignoring the sweep's own index file if it is globbed in."""
    runs = []
    for path in paths:
        payload = json.loads(path.read_text())
        if "metrics" in payload:
            runs.append(BenchmarkRun.model_validate(payload))
    return runs


def _parse_grid(values: list[str]) -> dict[str, list[float]]:
    grid: dict[str, list[float]] = {}
    for item in values:
        name, _, raw = item.partition("=")
        if name not in WEIGHT_NAMES or not raw:
            raise argparse.ArgumentTypeError(
                f"expected <alpha|beta|gamma|delta>=v1,v2,...: {item!r}"
            )
        grid[name] = [float(v) for v in raw.split(",")]
    return grid


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.sweep",
        description=(
            "Vary kv_aware weights one at a time on identical seeded request streams and "
            "tabulate the effect; or --table existing sweep JSONs."
        ),
    )
    parser.add_argument("--workload", action="append", default=None, help="repeatable")
    parser.add_argument("--all-workloads", action="store_true")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="WEIGHT=v1,v2,...",
        help="override one weight's values (repeatable); default is the built-in grid",
    )
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="W=v[,W=v...]",
        help="an extra setting that changes several weights at once (repeatable)",
    )
    parser.add_argument("--rate", type=float, default=None, help="override Poisson rate (req/s)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    add_worker_args(parser)
    parser.add_argument("--no-reference", action="store_true", help="skip round_robin/least_loaded")
    parser.add_argument("--table", nargs="*", type=Path, help="tabulate these run JSONs and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.table is not None:
        print(sweep_table(load_sweep_runs(args.table)), end="")
        return 0

    from benchmark.workloads import WORKLOAD_DIR

    names = args.workload or []
    if args.all_workloads:
        names = sorted(p.stem for p in WORKLOAD_DIR.glob("*.json"))
    if not names:
        print("give --workload NAME (repeatable), --all-workloads, or --table", file=sys.stderr)
        return 2

    grid = {**DEFAULT_GRID, **_parse_grid(args.grid)} if args.grid else DEFAULT_GRID
    points = one_at_a_time(KVAwareWeights(), grid) + [parse_point(p) for p in args.point]
    references = () if args.no_reference else REFERENCE_POLICIES

    runs: list[BenchmarkRun] = []
    for name in names:
        spec = with_rate(load_workload(name), args.rate)
        if not args.quiet:
            print(
                f"{spec.name}: {len(points)} weight settings + {len(references)} references",
                file=sys.stderr,
            )
        runs += asyncio.run(
            run_sweep(
                spec=spec,
                points=points,
                requests=args.requests,
                seed=args.seed,
                workers=args.workers,
                worker_config=worker_config_from_args(args),
                out_dir=args.out_dir,
                reference_policies=references,
            )
        )
    print(sweep_table(runs), end="")
    index = args.out_dir / "index.json"
    payload = {"grid": grid, "runs": [str(r.run_id) for r in runs]}
    index.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
