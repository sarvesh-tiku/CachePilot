import random

import pytest

from cachepilot.scheduler.scoring import TTFTEstimator, TTFTSample


def synthetic(
    estimator: TTFTEstimator, *, n: int, seed: int, noise_ms: float = 0.0
) -> list[TTFTSample]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        pending = rng.randint(0, 6)
        prompt = rng.choice([256, 512, 1024, 2048, 4096])
        overlap = rng.choice([0.0, 0.25, 0.5, 0.9, 1.0])
        ttft = estimator.predict_from(pending, prompt * (1 - overlap)) + rng.gauss(0, noise_ms)
        samples.append(TTFTSample(pending, prompt, overlap, max(ttft, 0.0)))
    return samples


def test_fit_recovers_exact_coefficients_from_noiseless_data() -> None:
    truth = TTFTEstimator(
        queue_wait_ms_per_pending=30, prefill_ms_per_token=0.3, fixed_overhead_ms=180
    )
    fitted = TTFTEstimator().fit(synthetic(truth, n=60, seed=1))
    assert fitted.queue_wait_ms_per_pending == pytest.approx(30, rel=1e-6)
    assert fitted.prefill_ms_per_token == pytest.approx(0.3, rel=1e-6)
    assert fitted.fixed_overhead_ms == pytest.approx(180, rel=1e-6)
    assert fitted.evaluate(synthetic(truth, n=20, seed=2)).mae_ms == pytest.approx(0, abs=1e-6)


def test_fit_is_close_under_noise_and_beats_the_default() -> None:
    truth = TTFTEstimator(
        queue_wait_ms_per_pending=25, prefill_ms_per_token=0.4, fixed_overhead_ms=200
    )
    training = synthetic(truth, n=400, seed=3, noise_ms=20)
    held_out = synthetic(truth, n=100, seed=4, noise_ms=20)
    default = TTFTEstimator()
    fitted = default.fit(training)
    assert fitted.prefill_ms_per_token == pytest.approx(0.4, rel=0.1)
    assert fitted.fixed_overhead_ms == pytest.approx(200, rel=0.1)
    assert fitted.evaluate(held_out).mae_ms < default.evaluate(held_out).mae_ms / 5
    assert fitted.evaluate(held_out).r2 > 0.95


def test_unidentifiable_feature_keeps_current_coefficient() -> None:
    # Single idle worker: pending is always 0, so its cost cannot be learned from this data.
    truth = TTFTEstimator(
        queue_wait_ms_per_pending=99, prefill_ms_per_token=0.3, fixed_overhead_ms=150
    )
    samples = [
        TTFTSample(0, prompt, overlap, truth.predict_from(0, prompt * (1 - overlap)))
        for prompt in (512, 1024, 2048)
        for overlap in (0.0, 0.5, 1.0)
    ]
    fitted = TTFTEstimator(queue_wait_ms_per_pending=12).fit(samples)
    assert fitted.queue_wait_ms_per_pending == 12
    assert fitted.prefill_ms_per_token == pytest.approx(0.3, rel=1e-6)
    assert fitted.fixed_overhead_ms == pytest.approx(150, rel=1e-6)


def test_held_feature_contribution_is_removed_before_fitting_the_rest() -> None:
    # Every sample has pending=2: the queue coefficient is held (at 10), and the intercept
    # must absorb only what is left after subtracting 2 * 10 from each observation.
    samples = [TTFTSample(2, p, 0.0, 20 + 0.5 * p + 100) for p in (100, 200, 300, 400)]
    fitted = TTFTEstimator(queue_wait_ms_per_pending=10).fit(samples)
    assert fitted.queue_wait_ms_per_pending == 10
    assert fitted.prefill_ms_per_token == pytest.approx(0.5)
    assert fitted.fixed_overhead_ms == pytest.approx(100)


def test_coefficients_never_go_negative() -> None:
    # TTFT that *falls* with more uncached tokens is noise; the fit must clamp rather than
    # report a negative prefill cost.
    samples = [TTFTSample(0, p, 0.0, 300 - 0.1 * p) for p in (100, 500, 1000, 2000)]
    fitted = TTFTEstimator().fit(samples)
    assert fitted.prefill_ms_per_token == 0.0
    assert fitted.fixed_overhead_ms == pytest.approx(
        sum(300 - 0.1 * p for p in (100, 500, 1000, 2000)) / 4
    )


def test_fit_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError):
        TTFTEstimator().fit([TTFTSample(0, 100, 0.0, 50.0)])


def test_evaluate_reports_bias_direction() -> None:
    samples = [TTFTSample(0, 1000, 0.0, 10.0)] * 3  # observed 10 ms, default predicts 65 ms
    report = TTFTEstimator().evaluate(samples)
    assert report.samples == 3
    assert report.bias_ms == pytest.approx(55.0)
    assert report.mae_ms == pytest.approx(55.0)


def test_parameters_round_trip() -> None:
    est = TTFTEstimator(queue_wait_ms_per_pending=1, prefill_ms_per_token=2, fixed_overhead_ms=3)
    assert est.parameters() == {
        "queue_wait_ms_per_pending": 1,
        "prefill_ms_per_token": 2,
        "fixed_overhead_ms": 3,
    }
