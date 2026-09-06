from __future__ import annotations

import asyncio
import random
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from cachepilot.core.clock import Clock, RealClock
from cachepilot.core.models import InferenceRequest
from cachepilot.workers.base import DoneEvent, GenerationEvent, StatsListener, TokenEvent
from cachepilot.workers.simulated import SimulatedWorkerConfig, _DummyText


@dataclass
class _Sequence:
    request: InferenceRequest
    uncached_tokens: int
    reserved_tokens: int  # prompt + max output: what a paged KV allocator must be able to hold
    output_tokens: int
    arrival_ms: float
    text: _DummyText
    events: deque[GenerationEvent] = field(default_factory=deque)
    signal: asyncio.Future[None] | None = None
    admitted_ms: float | None = None
    first_token_ms: float | None = None
    produced: int = 0
    done: bool = False

    def emit(self, event: GenerationEvent) -> None:
        self.events.append(event)
        if self.signal is not None and not self.signal.done():
            self.signal.set_result(None)


class BatchedSimulatedWorker:
    """Continuous-batching timing model of an inference worker.

    Where `SimulatedWorker` charges a fixed penalty per pending request and
    lets every request proceed concurrently, this engine has a *capacity*:

    - a FIFO waiting queue and a running batch of at most `max_batch_size`
      sequences whose reserved KV (prompt + max output tokens) fits in
      `kv_budget_tokens`, admitted head-of-line like vLLM's scheduler;
    - one engine iteration = prefill of the newly admitted sequences
      (uncached tokens × prefill cost, plus the fixed overhead) followed by
      one decode step for the whole batch, whose cost grows with batch size
      because every running sequence's KV is read each step;
    - a request's first token is produced by its prefill; each later token
      costs one iteration; requests that arrive mid-iteration wait for the
      boundary.

    Consequently TTFT grows superlinearly once a worker is saturated, which
    is the behavior the linear model cannot express and the reason this
    class exists. Timings are simulated time on the injected clock, never
    wall time, so runs are deterministic under any clock.
    """

    backend = "simulated"

    def __init__(
        self,
        worker_id: str,
        model: str,
        config: SimulatedWorkerConfig | None = None,
        seed: int = 0,
        clock: Clock | None = None,
        on_stats_changed: StatsListener | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.model = model
        self.config = config or SimulatedWorkerConfig()
        self._rng = random.Random(seed)
        self._clock: Clock = clock or RealClock()
        self.on_stats_changed = on_stats_changed
        self._waiting: deque[_Sequence] = deque()
        self._running: list[_Sequence] = []
        self._engine: asyncio.Task[None] | None = None
        self.iterations = 0

    # ---- InferenceWorker -------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return len(self._waiting)

    @property
    def active_requests(self) -> int:
        return len(self._running)

    async def generate(
        self, request: InferenceRequest, cache_overlap: float = 0.0
    ) -> AsyncIterator[GenerationEvent]:
        if not 0.0 <= cache_overlap <= 1.0:
            raise ValueError(f"cache_overlap must be in [0, 1], got {cache_overlap}")
        prompt = request.prompt_tokens_estimate
        seq = _Sequence(
            request=request,
            uncached_tokens=round(prompt * (1.0 - cache_overlap)),
            reserved_tokens=prompt + request.max_tokens,
            output_tokens=request.max_tokens,
            arrival_ms=self._clock.elapsed_ms(),
            text=_DummyText(request),
        )
        self._waiting.append(seq)
        await self._notify()
        if self._engine is None or self._engine.done():
            self._engine = self._clock.spawn(self._run_engine())

        loop = asyncio.get_running_loop()
        try:
            while True:
                while seq.events:
                    event = seq.events.popleft()
                    yield event
                    if isinstance(event, DoneEvent):
                        return
                seq.signal = loop.create_future()
                await self._clock.wait(seq.signal)
                seq.signal = None
        finally:
            if not seq.done:  # consumer went away mid-stream: drop the sequence
                if seq in self._waiting:
                    self._waiting.remove(seq)
                elif seq in self._running:
                    self._running.remove(seq)
                seq.done = True
                await self._notify()

    async def health(self) -> bool:
        return True

    async def stats(self) -> dict[str, Any]:
        return {
            **self.heartbeat_snapshot(),
            "queue_model": "batched",
            "max_batch_size": self.config.max_batch_size,
            "kv_budget_tokens": self.config.kv_budget_tokens,
            "reserved_tokens": sum(s.reserved_tokens for s in self._running),
            "iterations": self.iterations,
        }

    def heartbeat_snapshot(self) -> dict[str, Any]:
        return {
            "queue_depth": self.queue_depth,
            "active_requests": self.active_requests,
            "tokens_per_second_estimate": self.tokens_per_second_estimate(),
        }

    def tokens_per_second_estimate(self) -> float:
        """Per-sequence decode rate at the current batch size."""
        step = self.decode_step_ms(max(1, len(self._running)))
        return 1000.0 / step if step > 0 else 0.0

    # ---- cost model -------------------------------------------------------

    def decode_step_ms(self, batch_size: int) -> float:
        cfg = self.config
        return (
            cfg.decode_step_base_ms + cfg.decode_step_per_seq_ms * batch_size
        ) / cfg.service_rate_multiplier

    def prefill_ms(self, uncached_tokens: int) -> float:
        cfg = self.config
        return uncached_tokens * cfg.prefill_cost_per_token_ms / cfg.service_rate_multiplier

    # ---- engine -----------------------------------------------------------

    def _admit(self) -> list[_Sequence]:
        cfg = self.config
        reserved = sum(s.reserved_tokens for s in self._running)
        admitted = []
        while self._waiting and len(self._running) < cfg.max_batch_size:
            head = self._waiting[0]
            if self._running and reserved + head.reserved_tokens > cfg.kv_budget_tokens:
                break  # head-of-line blocks until KV frees up, as a paged allocator would
            self._waiting.popleft()
            head.admitted_ms = self._clock.elapsed_ms()
            reserved += head.reserved_tokens
            self._running.append(head)
            admitted.append(head)
        return admitted

    async def _run_engine(self) -> None:
        cfg = self.config
        try:
            while self._waiting or self._running:
                admitted = self._admit()
                decoding = [s for s in self._running if s.first_token_ms is not None]
                if admitted:
                    await self._notify()

                step_ms = 0.0
                if admitted:
                    step_ms += cfg.fixed_overhead_ms / cfg.service_rate_multiplier
                    step_ms += self.prefill_ms(sum(s.uncached_tokens for s in admitted))
                if decoding:
                    step_ms += self.decode_step_ms(len(self._running))
                step_ms = max(0.0, step_ms + self._rng.gauss(0.0, cfg.latency_noise_ms_stddev))
                self.iterations += 1
                await self._clock.sleep(step_ms / 1000.0)
                now = self._clock.elapsed_ms()

                finished: list[_Sequence] = []
                for seq in admitted:
                    seq.first_token_ms = now
                    self._produce(seq, now, finished)
                for seq in decoding:
                    self._produce(seq, now, finished)
                for seq in finished:
                    self._running.remove(seq)
                if finished:
                    await self._notify()
        finally:
            self._engine = None

    def _produce(self, seq: _Sequence, now: float, finished: list[_Sequence]) -> None:
        seq.emit(TokenEvent(index=seq.produced, text=seq.text.next()))
        seq.produced += 1
        if seq.produced >= seq.output_tokens:
            assert seq.admitted_ms is not None and seq.first_token_ms is not None
            seq.done = True
            finished.append(seq)
            seq.emit(
                DoneEvent(
                    queue_wait_ms=seq.admitted_ms - seq.arrival_ms,
                    ttft_ms=seq.first_token_ms - seq.arrival_ms,
                    total_latency_ms=now - seq.arrival_ms,
                    output_tokens=seq.produced,
                    finish_reason="length",
                )
            )

    async def _notify(self) -> None:
        if self.on_stats_changed is not None:
            await self.on_stats_changed(self)
