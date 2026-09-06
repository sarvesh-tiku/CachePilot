import asyncio

import pytest

from cachepilot.core.clock import EventClock, VirtualClock
from cachepilot.core.models import InferenceRequest, Message
from cachepilot.workers.base import DoneEvent, TokenEvent
from cachepilot.workers.batched import BatchedSimulatedWorker
from cachepilot.workers.simulated import SimulatedWorkerConfig

CONFIG = SimulatedWorkerConfig(
    queue_model="batched",
    prefill_cost_per_token_ms=0.1,
    fixed_overhead_ms=10.0,
    decode_step_base_ms=8.0,
    decode_step_per_seq_ms=1.0,
    max_batch_size=4,
    kv_budget_tokens=10_000,
    latency_noise_ms_stddev=0.0,
)


def make_request(prompt_tokens: int = 1000, max_tokens: int = 4) -> InferenceRequest:
    return InferenceRequest(
        model="sim-model",
        messages=[Message(role="user", content="x" * prompt_tokens * 4)],
        prompt_tokens_estimate=prompt_tokens,
        max_tokens=max_tokens,
    )


async def collect(worker: BatchedSimulatedWorker, request: InferenceRequest, overlap: float = 0.0):
    events = [e async for e in worker.generate(request, overlap)]
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert sum(isinstance(e, TokenEvent) for e in events) == done.output_tokens
    return done


async def test_single_request_timing_is_prefill_then_one_step_per_token() -> None:
    clock = EventClock()
    worker = BatchedSimulatedWorker("w", "m", CONFIG, clock=clock)
    results: list[DoneEvent] = []

    async def one() -> None:
        results.append(await collect(worker, make_request(1000, 4)))

    await clock.run([one()])
    done = results[0]
    # First iteration: overhead + prefill(1000 tokens); first token appears at its end.
    assert done.queue_wait_ms == 0.0
    assert done.ttft_ms == pytest.approx(10.0 + 100.0)
    # Three more tokens at one decode step (batch of 1) each.
    assert done.total_latency_ms == pytest.approx(110.0 + 3 * 9.0)
    assert clock.now_ms == pytest.approx(done.total_latency_ms)


async def test_cache_overlap_shortens_prefill_only() -> None:
    clock = EventClock()
    worker = BatchedSimulatedWorker("w", "m", CONFIG, clock=clock)
    results: list[DoneEvent] = []

    async def one() -> None:
        results.append(await collect(worker, make_request(1000, 2), overlap=0.9))

    await clock.run([one()])
    assert results[0].ttft_ms == pytest.approx(10.0 + 10.0)


async def test_batch_limit_queues_the_fifth_request_and_decode_step_grows_with_batch() -> None:
    clock = EventClock()
    worker = BatchedSimulatedWorker("w", "m", CONFIG, clock=clock)
    results: dict[int, DoneEvent] = {}

    async def one(i: int) -> None:
        results[i] = await collect(worker, make_request(100, 3))

    await clock.run([one(i) for i in range(5)])
    first_four = [results[i] for i in range(4)]
    # All admitted together: one prefill iteration covers 4 x 100 uncached tokens.
    assert all(d.queue_wait_ms == 0.0 for d in first_four)
    assert all(d.ttft_ms == pytest.approx(10.0 + 40.0) for d in first_four)
    # The fifth waited for a slot: it was admitted only after the batch drained.
    assert results[4].queue_wait_ms > 0.0
    assert results[4].ttft_ms > first_four[0].ttft_ms
    # Decode step with 4 running is 8 + 4 = 12 ms; a lone sequence's step is 9 ms.
    assert first_four[0].total_latency_ms == pytest.approx(50.0 + 2 * 12.0)


async def test_kv_budget_blocks_head_of_line_until_memory_frees() -> None:
    config = SimulatedWorkerConfig(**{**CONFIG.__dict__, "kv_budget_tokens": 1500})
    clock = EventClock()
    worker = BatchedSimulatedWorker("w", "m", config, clock=clock)
    results: dict[int, DoneEvent] = {}

    async def one(i: int) -> None:
        results[i] = await collect(worker, make_request(1000, 2))

    await clock.run([one(0), one(1)])
    assert results[0].queue_wait_ms == 0.0
    assert results[1].queue_wait_ms == pytest.approx(results[0].total_latency_ms)


async def test_beyond_the_batch_cap_each_extra_batch_waits_a_full_drain() -> None:
    """The linear model charges 12 ms per pending request. Here requests 5-8 of a burst of 12
    wait for the whole first batch to drain: prefill + 7 decode steps at batch size 4."""

    async def burst(n: int) -> float:
        clock = EventClock()
        worker = BatchedSimulatedWorker("w", "m", CONFIG, clock=clock)
        results: list[DoneEvent] = []

        async def one() -> None:
            results.append(await collect(worker, make_request(500, 8)))

        await clock.run([one() for _ in range(n)])
        return sorted(d.ttft_ms for d in results)[n // 2]

    small, large = await burst(4), await burst(12)
    assert small == pytest.approx(10.0 + 4 * 50.0)
    assert large - small == pytest.approx(small + 7 * (8.0 + 4.0))
    assert large - small > 8 * CONFIG.queue_wait_ms_per_pending


async def test_same_seed_reproduces_and_noise_changes_with_seed() -> None:
    noisy = SimulatedWorkerConfig(**{**CONFIG.__dict__, "latency_noise_ms_stddev": 3.0})

    async def run(seed: int) -> list[float]:
        clock = EventClock()
        worker = BatchedSimulatedWorker("w", "m", noisy, seed=seed, clock=clock)
        results: list[DoneEvent] = []

        async def one() -> None:
            results.append(await collect(worker, make_request(200, 3)))

        await clock.run([one() for _ in range(3)])
        return sorted(d.total_latency_ms for d in results)

    assert await run(1) == await run(1)
    assert await run(1) != await run(2)


async def test_consumer_cancellation_drops_the_sequence() -> None:
    worker = BatchedSimulatedWorker("w", "m", CONFIG, clock=VirtualClock())
    gen = worker.generate(make_request(100, 50))
    first = await gen.__anext__()
    assert isinstance(first, TokenEvent)
    assert worker.active_requests == 1
    await gen.aclose()
    assert worker.active_requests == 0
    await asyncio.sleep(0.01)  # engine loop notices the empty batch and exits


async def test_stats_and_heartbeat_describe_the_engine() -> None:
    worker = BatchedSimulatedWorker("w", "m", CONFIG, clock=VirtualClock())
    stats = await worker.stats()
    assert stats["queue_model"] == "batched"
    assert stats["max_batch_size"] == 4
    assert worker.heartbeat_snapshot() == {
        "queue_depth": 0,
        "active_requests": 0,
        "tokens_per_second_estimate": pytest.approx(1000 / 9.0),
    }


async def test_rejects_bad_overlap() -> None:
    worker = BatchedSimulatedWorker("w", "m", CONFIG, clock=VirtualClock())
    with pytest.raises(ValueError):
        await worker.generate(make_request(), 1.5).__anext__()
