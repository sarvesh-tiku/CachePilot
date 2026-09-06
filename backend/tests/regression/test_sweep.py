import json
from pathlib import Path

import pytest

from benchmark.runner import run_benchmark
from benchmark.sweep import DEFAULT_GRID, label_for, main, one_at_a_time, run_sweep, sweep_table
from benchmark.workloads import ArrivalSpec, WorkloadSpec
from cachepilot.scheduler.kv_aware import KVAwareWeights

TINY = WorkloadSpec(
    name="tiny_sweep",
    prefix_tokens=512,
    question_tokens=8,
    output_tokens=4,
    shared_prefix_count=2,
    shared_fraction=0.8,
    arrival=ArrivalSpec(rate_rps=50.0),
)
GRID = {"alpha": [0.0, 2.0, 8.0], "delta": [0.05]}


def test_one_at_a_time_grid_skips_default_values_and_varies_one_weight() -> None:
    points = one_at_a_time(KVAwareWeights(), GRID)
    assert [p.label for p in points] == ["default", "alpha=0", "alpha=8", "delta=0.05"]
    for point in points[1:]:
        changed = [n for n in ("alpha", "beta", "gamma", "delta")
                   if getattr(point.weights, n) != getattr(KVAwareWeights(), n)]  # fmt: skip
        assert len(changed) == 1
        assert label_for(point.weights.__dict__) == point.label
    assert label_for(KVAwareWeights().__dict__) == "default"
    default_points = one_at_a_time(KVAwareWeights())
    assert len(default_points) == 1 + sum(len(v) for v in DEFAULT_GRID.values())
    assert len({p.slug for p in default_points}) == len(default_points)


async def test_sweep_default_point_matches_a_plain_kv_aware_run(tmp_path: Path) -> None:
    runs = await run_sweep(
        spec=TINY, points=one_at_a_time(KVAwareWeights(), GRID), requests=40, seed=1,
        out_dir=tmp_path,
    )  # fmt: skip
    assert [r.policy for r in runs] == ["round_robin", "least_loaded"] + ["kv_aware"] * 4
    plain, _ = await run_benchmark(spec=TINY, policy="kv_aware", requests=40, seed=1)
    default = next(
        r for r in runs if r.policy == "kv_aware" and label_for(r.policy_parameters) == "default"
    )
    assert default.metrics == plain.metrics

    alpha0 = next(r for r in runs if label_for(r.policy_parameters) == "alpha=0")
    assert alpha0.metrics.mean_cache_overlap <= default.metrics.mean_cache_overlap

    names = sorted(p.name for p in tmp_path.glob("*.json"))
    assert names == [
        "tiny_sweep_kv_aware_alpha0_seed1.json",
        "tiny_sweep_kv_aware_alpha8_seed1.json",
        "tiny_sweep_kv_aware_default_seed1.json",
        "tiny_sweep_kv_aware_delta0p05_seed1.json",
        "tiny_sweep_least_loaded_seed1.json",
        "tiny_sweep_round_robin_seed1.json",
    ]

    table = sweep_table(runs)
    assert "#### tiny_sweep" in table
    assert "| **default**" in table
    assert "| alpha=0 |" in table and "| delta=0.05 |" in table
    assert "| round_robin (reference) |" in table
    # Every non-default row carries a delta against the default row.
    assert table.count("%)") >= 5 * 4


def test_cli_runs_and_tabulates_from_disk(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec_path = tmp_path / "tiny.json"
    spec_path.write_text(TINY.model_dump_json())
    out = tmp_path / "sweep"
    code = main(
        [
            "--workload", str(spec_path), "--requests", "20", "--seed", "2",
            "--grid", "alpha=0,2", "--grid", "beta=0",
            "--grid", "gamma=0.4", "--grid", "delta=0.002",
            "--out-dir", str(out), "--quiet",
        ]
    )  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    assert "| alpha=0 |" in printed and "| beta=0 |" in printed
    assert "| gamma=" not in printed  # gamma grid held at its default: no row
    index = json.loads((out / "index.json").read_text())
    assert index["grid"]["alpha"] == [0.0, 2.0]
    assert len(index["runs"]) == 2 + 3

    code = main(["--table", *map(str, sorted(out.glob("*.json")))])
    assert code == 0
    assert "| alpha=0 |" in capsys.readouterr().out
