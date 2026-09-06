from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Coroutine, Iterable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class Clock(Protocol):
    async def sleep(self, seconds: float) -> None: ...

    def elapsed_ms(self) -> float:
        """Milliseconds since the clock started, in this clock's notion of time."""
        ...

    async def wait(self, future: asyncio.Future[T]) -> T:
        """Block on a future that another task on this clock will resolve."""
        ...

    def spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Start a background task that only ever blocks on this clock."""
        ...


class RealClock:
    def __init__(self) -> None:
        self._start = time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0

    async def wait(self, future: asyncio.Future[T]) -> T:
        return await future

    def spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        return asyncio.create_task(coro)


class VirtualClock:
    """Fast-forward clock for tests.

    Sleeps return immediately after yielding to the event loop. Simulated
    timings still come from the worker's cost model, so results are identical
    to a real-clock run; only wall time is skipped. This is not an
    event-ordered simulation: concurrent sleepers do not interleave in
    simulated-time order. Use EventClock for that.
    """

    def __init__(self) -> None:
        self.elapsed_s = 0.0

    async def sleep(self, seconds: float) -> None:
        self.elapsed_s += max(0.0, seconds)
        await asyncio.sleep(0)

    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0

    async def wait(self, future: asyncio.Future[T]) -> T:
        return await future

    def spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        return asyncio.create_task(coro)


class EventClock:
    """Discrete-event clock for benchmarks.

    Sleeping tasks park on a heap keyed by wake time. `run` lets every runnable
    task progress until all live tasks are parked, then advances simulated time
    to the earliest wake and releases it. Concurrent requests therefore
    interleave exactly as they would in simulated time, and a whole benchmark
    completes in wall time proportional to the number of events, not to the
    simulated duration.

    Assumption: tasks only block on this clock — `sleep`, or `wait` on a
    future that another clock task resolves — or on things that resolve
    without waiting, like uncontended locks or in-memory stores. A task that
    blocks on real I/O would stall the driver. Background tasks (a worker's
    batching engine, say) must be started with `spawn` so the driver counts
    them.
    """

    def __init__(self) -> None:
        self.now_ms = 0.0
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0
        self._live = 0
        self._parked = 0
        self._waits: set[asyncio.Future[Any]] = set()
        self._tasks: list[asyncio.Task[None]] = []
        self._failure: BaseException | None = None

    def elapsed_ms(self) -> float:
        return self.now_ms

    async def wait(self, future: asyncio.Future[T]) -> T:
        self._waits.add(future)
        try:
            return await future
        finally:
            self._waits.discard(future)

    def spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(self._track(coro))
        self._live += 1
        self._tasks.append(task)
        return task

    def _all_blocked(self) -> bool:
        # A future that has been resolved but whose waiter has not run yet counts as runnable.
        pending_waits = sum(1 for f in self._waits if not f.done())
        return self._parked + pending_waits >= self._live

    async def sleep(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._seq += 1
        heapq.heappush(self._heap, (self.now_ms + max(0.0, seconds) * 1000.0, self._seq, future))
        self._parked += 1
        try:
            await future
        finally:
            if future.cancelled():
                self._parked -= 1

    async def run(self, coros: Iterable[Coroutine[Any, Any, None]]) -> None:
        tasks = [asyncio.create_task(self._track(coro)) for coro in coros]
        self._tasks = tasks
        self._live = len(tasks)
        self._failure = None
        try:
            while self._live > 0 and self._failure is None:
                while self._live > 0 and not self._all_blocked() and self._failure is None:
                    await asyncio.sleep(0)
                if self._live == 0 or self._failure is not None:
                    break
                if not self._heap:
                    raise RuntimeError("live tasks are blocked on something other than the clock")
                wake_ms, _, future = heapq.heappop(self._heap)
                if future.cancelled():
                    continue
                self._parked -= 1
                self.now_ms = max(self.now_ms, wake_ms)
                future.set_result(None)
        finally:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._failure is not None:
            raise self._failure

    async def _track(self, coro: Coroutine[Any, Any, None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if self._failure is None:
                self._failure = exc
            raise
        finally:
            self._live -= 1
