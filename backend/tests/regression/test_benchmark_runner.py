import json
from pathlib import Path

import pytest

from benchmark.analysis import compare, load_runs
from benchmark.report import RequestRow, write_outputs
from benchmark.runner import main, run_benchmark
from benchmark.workloads import ArrivalSpec, WorkloadSpec

TINY = WorkloadSpec(
    name="tiny_shared",
    prefix_tokens=512,
    question_tokens=8,
    output_tokens=4,
    shared_prefix_count=2,
    shared_fraction=0.8,
    arrival=ArrivalSpec(rate_rps=50.0),
)
N = 40


@pytest.mark.parametrize("policy", ["round_robin", "least_loaded", "kv_aware"])
async def test_each_policy_completes_the_workload(policy: str) -> None:
    run, rows = await run_benchmark(spec=TINY, policy=policy, requests=N, seed=1)

    assert run.metrics.completed == N
    assert len(rows) == N
    assert [r.index for r in rows] == list(range(N))
    assert sum(run.metrics.worker_request_counts.values()) == N
    assert 0.0 <= run.metrics.cache_hit_rate <= 1.0
    assert 0.0 <= run.metrics.mean_cache_overlap <= 1.0
    assert 0 < run.metrics.ttft_ms.p50 <= run.metrics.ttft_ms.p99 < 5000
    assert run.metrics.total_latency_ms.p50 > run.metrics.ttft_ms.p50
    for row in rows:
        # Simulated wall time must agree with the cost model's own latency.
        assert row.completed_ms - row.arrival_ms == pytest.approx(row.total_latency_ms, abs=1e-6)


async def test_same_seed_reproduces_identical_results() -> None:
    run_a, rows_a = await run_benchmark(spec=TINY, policy="kv_aware", requests=N, seed=7)
    run_b, rows_b = await run_benchmark(spec=TINY, policy="kv_aware", requests=N, seed=7)
    assert rows_a == rows_b
    assert run_a.metrics == run_b.metrics


async def test_different_seed_changes_results() -> None:
    _, rows_a = await run_benchmark(spec=TINY, policy="kv_aware", requests=N, seed=7)
    _, rows_b = await run_benchmark(spec=TINY, policy="kv_aware", requests=N, seed=8)
    assert rows_a != rows_b


async def test_round_robin_alternates_and_kv_aware_reuses_more_prefix() -> None:
    rr, _ = await run_benchmark(spec=TINY, policy="round_robin", requests=N, seed=1)
    kv, _ = await run_benchmark(spec=TINY, policy="kv_aware", requests=N, seed=1)
    assert max(rr.metrics.worker_request_counts.values()) <= N // 3 + 1
    # Locality is a structural property of kv_aware; latency wins are not asserted.
    assert kv.metrics.mean_cache_overlap >= rr.metrics.mean_cache_overlap


async def test_outputs_round_trip_and_compare(tmp_path: Path) -> None:
    runs = []
    for policy in ("round_robin", "kv_aware"):
        run, rows = await run_benchmark(spec=TINY, policy=policy, requests=N, seed=1)
        json_path, csv_path = write_outputs(run, rows, tmp_path)
        runs.append(run)
        assert json.loads(json_path.read_text())["policy"] == policy
        lines = csv_path.read_text().splitlines()
        assert lines[0].split(",") == list(RequestRow.model_fields)
        assert len(lines) == N + 1

    loaded = load_runs(sorted(tmp_path.glob("*.json")))
    table = compare(loaded)
    assert "### tiny_shared" in table
    assert "| round_robin |" in table and "| kv_aware |" in table


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec_path = tmp_path / "tiny.json"
    spec_path.write_text(TINY.model_dump_json())
    code = main(
        [
            "--workload", str(spec_path), "--policy", "least_loaded",
            "--requests", "10", "--seed", "3", "--out-dir", str(tmp_path / "out"),
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0
    assert "Policy:               least_loaded" in out
    assert "Completed:            10" in out
    assert (tmp_path / "out" / "tiny_shared_least_loaded_seed3.json").exists()
