import json
from pathlib import Path

import pytest

from benchmark.fit import main, read_rows, samples_from_rows, samples_from_views
from benchmark.report import write_outputs
from benchmark.runner import run_benchmark
from benchmark.workloads import ArrivalSpec, WorkloadSpec
from cachepilot.scheduler.scoring import TTFTEstimator
from cachepilot.workers.simulated import SimulatedWorkerConfig

MIXED = WorkloadSpec(
    name="tiny_mixed",
    prefix_tokens=1024,
    question_tokens=8,
    output_tokens=4,
    shared_prefix_count=3,
    shared_fraction=0.7,
    arrival=ArrivalSpec(rate_rps=60.0),
)


async def test_fit_from_simulated_rows_recovers_the_simulator_cost_model(tmp_path: Path) -> None:
    """The simulator *is* the estimator's model plus noise, so fitting its own output must
    land on its own coefficients — the end-to-end check that samples are assembled right."""
    _, rows = await run_benchmark(spec=MIXED, policy="round_robin", requests=300, seed=5)
    samples = samples_from_rows(rows)
    assert {s.pending for s in samples} != {0}, "workload never queued; pending not identifiable"
    assert len({round(s.uncached_tokens) for s in samples}) > 1

    fitted = TTFTEstimator(
        queue_wait_ms_per_pending=0, prefill_ms_per_token=0, fixed_overhead_ms=0
    ).fit(samples)
    cfg = SimulatedWorkerConfig()
    assert fitted.queue_wait_ms_per_pending == pytest.approx(cfg.queue_wait_ms_per_pending, abs=1.0)
    assert fitted.prefill_ms_per_token == pytest.approx(cfg.prefill_cost_per_token_ms, abs=0.01)
    assert fitted.fixed_overhead_ms == pytest.approx(cfg.fixed_overhead_ms, abs=3.0)
    assert fitted.evaluate(samples).mae_ms < 2 * cfg.latency_noise_ms_stddev


async def test_cli_fits_from_csvs_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = []
    for seed in (1, 2):
        run, rows = await run_benchmark(spec=MIXED, policy="kv_aware", requests=120, seed=seed)
        _, csv_path = write_outputs(run, rows, tmp_path)
        paths.append(csv_path)
    assert read_rows(paths[0])[0].pending_at_decision >= 0

    out = tmp_path / "fit.json"
    code = main(["--csv", *map(str, paths), "--prefill", "1.0", "--write", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert "Leave-one-source-out" in text
    assert "export CACHEPILOT_TTFT_PREFILL_MS_PER_TOKEN=" in text

    saved = json.loads(out.read_text())
    assert saved["baseline"]["prefill_ms_per_token"] == 1.0
    assert saved["estimator"]["prefill_ms_per_token"] == pytest.approx(0.05, abs=0.01)
    assert saved["in_sample"]["mae_ms"] < saved["baseline_in_sample"]["mae_ms"]
    assert set(saved["holdout"]) == {p.name for p in paths}


def test_cli_requires_a_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "--csv" in capsys.readouterr().err


def test_samples_from_views_use_the_chosen_candidate_and_skip_incomplete() -> None:
    views = [
        {
            "request": {"prompt_tokens_estimate": 1000},
            "decision": {
                "selected_worker_id": "w1",
                "candidates": [
                    {"worker_id": "w0", "queue_depth": 9, "active_requests": 9},
                    {"worker_id": "w1", "queue_depth": 2, "active_requests": 1},
                ],
            },
            "result": {"cache_overlap": 0.5, "ttft_ms": 123.0},
        },
        {"request": {"prompt_tokens_estimate": 10}, "decision": None, "result": None},
        {
            "request": {"prompt_tokens_estimate": 10},
            "decision": {"selected_worker_id": "w0", "candidates": []},
            "result": {"cache_overlap": 0.0, "ttft_ms": 1.0},
        },
    ]
    samples = samples_from_views(views)
    assert len(samples) == 1
    assert (samples[0].pending, samples[0].prompt_tokens, samples[0].cache_overlap) == (
        3,
        1000,
        0.5,
    )
    assert samples[0].uncached_tokens == 500
