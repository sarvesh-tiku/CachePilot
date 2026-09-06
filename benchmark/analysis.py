from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from benchmark.report import BenchmarkRun

BASELINE = "round_robin"

COLUMNS: list[tuple[str, str]] = [
    ("TTFT p50", "ttft_p50"),
    ("TTFT p99", "ttft_p99"),
    ("Lat p99", "lat_p99"),
    ("Queue mean", "queue_mean"),
    ("Hit rate", "hit_rate"),
    ("Prefill saved", "prefill_saved"),
    ("Imbalance", "imbalance"),
    ("Evictions", "evictions"),
]


def load_runs(paths: list[Path]) -> list[BenchmarkRun]:
    return [BenchmarkRun.model_validate(json.loads(p.read_text())) for p in paths]


def metric_values(run: BenchmarkRun) -> dict[str, float]:
    m = run.metrics
    return {
        "ttft_p50": m.ttft_ms.p50,
        "ttft_p99": m.ttft_ms.p99,
        "lat_p99": m.total_latency_ms.p99,
        "queue_mean": m.queue_wait_ms.mean,
        "hit_rate": m.cache_hit_rate,
        "prefill_saved": m.prefill_saved_fraction,
        "imbalance": m.worker_imbalance,
        "evictions": float(sum(k.evictions for k in m.kv)),
    }


def _fmt(key: str, value: float) -> str:
    if key in {"hit_rate", "prefill_saved"}:
        return f"{value:.1%}"
    if key == "imbalance":
        return f"{value:.2f}"
    if key == "evictions":
        return f"{int(value)}"
    return f"{value:.0f} ms"


def _delta(key: str, value: float, base: float) -> str:
    if key in {"hit_rate", "prefill_saved"}:
        return f"{(value - base) * 100:+.1f} pp"
    if base == 0:
        return "n/a"
    return f"{(value - base) / base:+.1%}"


def compare(runs: list[BenchmarkRun]) -> str:
    by_workload: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for run in runs:
        by_workload[run.workload.name].append(run)

    sections: list[str] = []
    for workload, group in sorted(by_workload.items()):
        group.sort(key=lambda r: r.policy)
        baseline = next((r for r in group if r.policy == BASELINE), None)
        header = ["Policy", *[label for label, _ in COLUMNS]]
        cfg = group[0].worker_config
        model = (
            f", batched workers (max batch {cfg['max_batch_size']})"
            if cfg.get("queue_model") == "batched"
            else ""
        )
        first = group[0]
        lines = [
            f"### {workload} "
            f"({first.requested} requests, seed {first.seed}, {first.workers} workers{model})",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        for run in group:
            values = metric_values(run)
            cells = [run.policy]
            for _, key in COLUMNS:
                cell = _fmt(key, values[key])
                if baseline is not None and run is not baseline and key != "evictions":
                    cell += f" ({_delta(key, values[key], metric_values(baseline)[key])})"
                cells.append(cell)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"_{group[0].workload.description}_")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.analysis",
        description="Compare benchmark run JSONs as a markdown table (deltas vs round_robin).",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    print(compare(load_runs(args.paths)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
