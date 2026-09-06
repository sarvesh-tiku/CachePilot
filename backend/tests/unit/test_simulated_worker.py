import pytest

from cachepilot.core.clock import VirtualClock
from cachepilot.core.models import InferenceRequest, Message
from cachepilot.workers.base import DoneEvent, TokenEvent
from cachepilot.workers.simulated import SimulatedWorker, SimulatedWorkerConfig

CONFIG = SimulatedWorkerConfig(
    prefill_cost_per_token_ms=0.1,
    decode_cost_per_token_ms=10.0,
    fixed_overhead_ms=20.0,
    kv_capacity_bytes=1024 * 1024 * 1024,
    queue_wait_ms_per_pending=15.0,
    latency_noise_ms_stddev=2.0,
)


def make_worker(seed: int = 42) -> SimulatedWorker:
    return SimulatedWorker("worker-0", "sim-model", config=CONFIG, seed=seed, clock=VirtualClock())


def make_request(prompt_tokens: int = 1000, max_tokens: int = 8) -> InferenceRequest:
    return InferenceRequest(
        model="sim-model",
        messages=[Message(role="user", content="x" * prompt_tokens * 4)],
        prompt_tokens_estimate=prompt_tokens,
        max_tokens=max_tokens,
    )


def test_same_seed_and_inputs_give_identical_results() -> None:
    worker_a = make_worker(seed=7)
    worker_b = make_worker(seed=7)

    for _ in range(3):
        assert worker_a.simulate_request(1000, 50) == worker_b.simulate_request(1000, 50)


def test_different_seeds_diverge() -> None:
    result_a = make_worker(seed=1).simulate_request(1000, 50)
    result_b = make_worker(seed=2).simulate_request(1000, 50)
    assert result_a.ttft_ms != result_b.ttft_ms


def test_cache_overlap_reduces_prefill_work() -> None:
    miss = make_worker(seed=7).simulate_request(1000, 50, cache_overlap=0.0)
    hit = make_worker(seed=7).simulate_request(1000, 50, cache_overlap=0.9)

    assert miss.effective_prefill_tokens == 1000
    assert hit.effective_prefill_tokens == 100
    assert hit.prefill_ms < miss.prefill_ms
    assert hit.ttft_ms < miss.ttft_ms


def test_queue_depth_increases_ttft() -> None:
    idle_worker = make_worker(seed=7)
    busy_worker = make_worker(seed=7)
    busy_worker.queue_depth = 10

    idle = idle_worker.simulate_request(500, 20)
    busy = busy_worker.simulate_request(500, 20)

    assert busy.queue_wait_ms == 150.0
    assert busy.ttft_ms > idle.ttft_ms


def test_decode_cost_counts_steps_after_the_first_token() -> None:
    one = make_worker(seed=7).simulate_request(100, 1)
    ten = make_worker(seed=7).simulate_request(100, 10)
    hundred = make_worker(seed=7).simulate_request(100, 100)
    assert one.decode_ms == 0.0
    assert ten.decode_ms == pytest.approx(9 * 10.0)
    assert hundred.decode_ms == pytest.approx(99 * 10.0)


def test_service_rate_multiplier_speeds_up_worker() -> None:
    fast_cfg = SimulatedWorkerConfig(**{**CONFIG.__dict__, "service_rate_multiplier": 2.0})
    slow = make_worker(seed=7).simulate_request(1000, 50)
    fast = SimulatedWorker("w", "m", config=fast_cfg, seed=7).simulate_request(1000, 50)
    assert fast.prefill_ms == pytest.approx(slow.prefill_ms / 2)
    assert fast.decode_ms == pytest.approx(slow.decode_ms / 2)


@pytest.mark.parametrize("overlap", [-0.1, 1.5])
def test_invalid_cache_overlap_raises(overlap: float) -> None:
    with pytest.raises(ValueError):
        make_worker().simulate_request(100, 10, cache_overlap=overlap)


async def test_generate_streams_tokens_then_done_with_planned_timings() -> None:
    worker = make_worker(seed=7)
    expected = make_worker(seed=7).simulate_request(1000, 8)
    request = make_request(prompt_tokens=1000, max_tokens=8)

    events = [event async for event in worker.generate(request)]

    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert [t.index for t in tokens] == list(range(8))
    assert all(t.text for t in tokens)

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.output_tokens == 8
    assert done.ttft_ms == expected.ttft_ms
    assert done.total_latency_ms == expected.total_latency_ms
    assert done.queue_wait_ms == expected.queue_wait_ms


async def test_generate_output_text_is_deterministic_per_request() -> None:
    request = make_request()
    text_a = "".join(
        e.text
        for e in [x async for x in make_worker().generate(request)]
        if isinstance(e, TokenEvent)
    )
    text_b = "".join(
        e.text
        for e in [x async for x in make_worker().generate(request)]
        if isinstance(e, TokenEvent)
    )
    assert text_a == text_b


async def test_generate_tracks_queue_and_active_counts() -> None:
    seen: list[tuple[int, int]] = []

    async def listener(w: SimulatedWorker) -> None:
        seen.append((w.queue_depth, w.active_requests))

    worker = SimulatedWorker(
        "w", "m", config=CONFIG, clock=VirtualClock(), on_stats_changed=listener
    )
    async for _ in worker.generate(make_request(max_tokens=2)):
        pass

    assert seen == [(1, 0), (0, 1), (0, 0)]
    assert worker.queue_depth == 0
    assert worker.active_requests == 0


async def test_generate_restores_counts_when_consumer_disconnects() -> None:
    worker = make_worker()
    stream = worker.generate(make_request(max_tokens=5))
    await anext(stream)
    assert worker.active_requests == 1
    await stream.aclose()
    assert worker.active_requests == 0
    assert worker.queue_depth == 0
