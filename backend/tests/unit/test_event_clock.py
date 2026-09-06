import asyncio

import pytest

from cachepilot.core.clock import EventClock


async def test_sleepers_wake_in_simulated_time_order_not_start_order() -> None:
    clock = EventClock()
    order: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        order.append(f"{name}@{clock.now_ms:.0f}")

    await clock.run([sleeper("slow", 0.5), sleeper("fast", 0.1), sleeper("mid", 0.25)])

    assert order == ["fast@100", "mid@250", "slow@500"]
    assert clock.now_ms == 500.0


async def test_nested_sleeps_accumulate_from_wake_time() -> None:
    clock = EventClock()
    stamps: list[float] = []

    async def worker() -> None:
        for _ in range(3):
            await clock.sleep(0.2)
            stamps.append(clock.now_ms)

    await clock.run([worker(), worker()])
    assert stamps == [200.0, 200.0, 400.0, 400.0, 600.0, 600.0]


async def test_concurrent_interleaving_reflects_simulated_time() -> None:
    """A task starting later but sleeping less finishes first, as in real time."""
    clock = EventClock()
    log: list[str] = []

    async def a() -> None:
        await clock.sleep(0.1)
        log.append("a-start")
        await clock.sleep(1.0)
        log.append("a-end")

    async def b() -> None:
        await clock.sleep(0.5)
        log.append("b-start")
        await clock.sleep(0.1)
        log.append("b-end")

    await clock.run([a(), b()])
    assert log == ["a-start", "b-start", "b-end", "a-end"]


async def test_equal_wake_times_are_fifo() -> None:
    clock = EventClock()
    order: list[int] = []

    async def t(i: int) -> None:
        await clock.sleep(0.1)
        order.append(i)

    await clock.run([t(i) for i in range(5)])
    assert order == [0, 1, 2, 3, 4]


async def test_exceptions_propagate_and_remaining_tasks_are_cancelled() -> None:
    clock = EventClock()
    finished = False

    async def failing() -> None:
        await clock.sleep(0.1)
        raise ValueError("boom")

    async def long_running() -> None:
        nonlocal finished
        await clock.sleep(10)
        finished = True

    with pytest.raises(ValueError, match="boom"):
        await clock.run([failing(), long_running()])
    assert finished is False


async def test_run_with_no_tasks_returns_immediately() -> None:
    clock = EventClock()
    await clock.run([])
    assert clock.now_ms == 0.0


async def test_task_blocking_on_foreign_primitive_is_detected() -> None:
    clock = EventClock()
    never = asyncio.Event()

    async def stuck() -> None:
        await never.wait()

    async def sleeper() -> None:
        await clock.sleep(0.1)

    # The sleeper wakes; then only the stuck task is live, and it is not parked on the clock.
    # The driver spins on sleep(0) forever in that case, so bound it with a timeout.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(clock.run([stuck(), sleeper()]), timeout=0.2)


async def test_wait_on_a_future_resolved_by_a_spawned_task_preserves_time_order() -> None:
    clock = EventClock()
    log: list[tuple[str, float]] = []
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[str] = loop.create_future()

    async def producer() -> None:
        await clock.sleep(0.5)
        ready.set_result("token")
        await clock.sleep(0.5)
        log.append(("producer-done", clock.now_ms))

    async def consumer() -> None:
        clock.spawn(producer())
        value = await clock.wait(ready)
        log.append((value, clock.now_ms))

    async def bystander() -> None:
        await clock.sleep(0.7)
        log.append(("bystander", clock.now_ms))

    await clock.run([consumer(), bystander()])
    # The consumer must observe the token at t=500ms, before the bystander wakes at 700ms,
    # even though it was blocked on a future rather than the clock's heap.
    assert log == [("token", 500.0), ("bystander", 700.0), ("producer-done", 1000.0)]


async def test_wait_on_a_future_nobody_resolves_is_a_detected_deadlock() -> None:
    clock = EventClock()
    loop = asyncio.get_running_loop()
    never: asyncio.Future[None] = loop.create_future()

    async def stuck() -> None:
        await clock.wait(never)

    with pytest.raises(RuntimeError, match="blocked on something other than the clock"):
        await clock.run([stuck()])
