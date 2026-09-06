from __future__ import annotations

import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from cachepilot.core.clock import Clock, RealClock
from cachepilot.core.models import InferenceRequest
from cachepilot.workers.base import DoneEvent, GenerationEvent, StatsListener, TokenEvent

MIB = 1024 * 1024

_VOCAB = (
    "cache", "prefix", "token", "worker", "queue", "latency", "prefill", "decode",
    "route", "batch", "block", "tensor", "kernel", "memory", "bandwidth", "schedule",
    "the", "a", "is", "of", "and", "to", "with", "for",
)  # fmt: skip


@dataclass(frozen=True)
class SimulatedWorkerConfig:
    prefill_cost_per_token_ms: float = 0.05
    decode_cost_per_token_ms: float = 8.0
    fixed_overhead_ms: float = 15.0
    kv_capacity_bytes: int = 8 * 1024 * MIB
    service_rate_multiplier: float = 1.0
    queue_wait_ms_per_pending: float = 12.0
    latency_noise_ms_stddev: float = 5.0
    # "linear": every request runs concurrently and pays a fixed penalty per pending request.
    # "batched": a continuous-batching engine with a bounded batch and KV budget; see
    # workers/batched.py. Only the batched model can saturate.
    queue_model: Literal["linear", "batched"] = "linear"
    max_batch_size: int = 16
    kv_budget_tokens: int = 65_536  # KV the engine can hold for running sequences
    decode_step_base_ms: float = 7.5  # per decode iteration: weights + kernel launches
    decode_step_per_seq_ms: float = 0.5  # per running sequence: its KV must be read each step


@dataclass(frozen=True)
class SimulationResult:
    queue_wait_ms: float
    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    total_latency_ms: float
    effective_prefill_tokens: int


class SimulatedWorker:
    """Seeded, deterministic timing model of an inference worker.

    This does not run a model. It exists so routing policies can be developed
    and benchmarked reproducibly without GPU hardware. Reported timings come
    from the cost model, never from measuring wall time, so results are
    identical under a real or virtual clock. KV residency is not tracked here;
    the gateway's KV directory plays that role for simulated workers.
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
        self.queue_depth = 0
        self.active_requests = 0

    def simulate_request(
        self,
        prompt_tokens: int,
        output_tokens: int,
        cache_overlap: float = 0.0,
    ) -> SimulationResult:
        if not 0.0 <= cache_overlap <= 1.0:
            raise ValueError(f"cache_overlap must be in [0, 1], got {cache_overlap}")

        cfg = self.config
        effective_prefill_tokens = round(prompt_tokens * (1.0 - cache_overlap))

        queue_wait_ms = (self.queue_depth + self.active_requests) * cfg.queue_wait_ms_per_pending
        prefill_ms = (
            effective_prefill_tokens * cfg.prefill_cost_per_token_ms / cfg.service_rate_multiplier
        )
        # The first token arrives at TTFT; each remaining token costs one decode step.
        decode_ms = max(0, output_tokens - 1) * self.decode_ms_per_token()

        noise_ms = self._rng.gauss(0.0, cfg.latency_noise_ms_stddev)
        ttft_ms = max(0.0, queue_wait_ms + prefill_ms + cfg.fixed_overhead_ms + noise_ms)

        return SimulationResult(
            queue_wait_ms=queue_wait_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            ttft_ms=ttft_ms,
            total_latency_ms=ttft_ms + decode_ms,
            effective_prefill_tokens=effective_prefill_tokens,
        )

    async def generate(
        self, request: InferenceRequest, cache_overlap: float = 0.0
    ) -> AsyncIterator[GenerationEvent]:
        output_tokens = request.max_tokens
        plan = self.simulate_request(request.prompt_tokens_estimate, output_tokens, cache_overlap)
        words = _DummyText(request)

        self.queue_depth += 1
        await self._notify()
        queued = True
        try:
            await self._clock.sleep(plan.queue_wait_ms / 1000)
            self.queue_depth -= 1
            self.active_requests += 1
            queued = False
            await self._notify()

            await self._clock.sleep((plan.ttft_ms - plan.queue_wait_ms) / 1000)
            per_token_s = self.decode_ms_per_token() / 1000
            for index in range(output_tokens):
                if index > 0:
                    await self._clock.sleep(per_token_s)
                yield TokenEvent(index=index, text=words.next())

            yield DoneEvent(
                queue_wait_ms=plan.queue_wait_ms,
                ttft_ms=plan.ttft_ms,
                total_latency_ms=plan.total_latency_ms,
                output_tokens=output_tokens,
                finish_reason="length",
            )
        finally:
            if queued:
                self.queue_depth -= 1
            else:
                self.active_requests -= 1
            await self._notify()

    async def health(self) -> bool:
        return True

    async def stats(self) -> dict[str, Any]:
        return self.heartbeat_snapshot()

    def decode_ms_per_token(self) -> float:
        return self.config.decode_cost_per_token_ms / self.config.service_rate_multiplier

    def tokens_per_second_estimate(self) -> float:
        per_token = self.decode_ms_per_token()
        return 1000.0 / per_token if per_token > 0 else 0.0

    def heartbeat_snapshot(self) -> dict[str, Any]:
        return {
            "queue_depth": self.queue_depth,
            "active_requests": self.active_requests,
            "tokens_per_second_estimate": self.tokens_per_second_estimate(),
        }

    async def _notify(self) -> None:
        if self.on_stats_changed is not None:
            await self.on_stats_changed(self)


class _DummyText:
    """Deterministic placeholder output, seeded by request id."""

    def __init__(self, request: InferenceRequest) -> None:
        self._rng = random.Random(request.request_id.int)
        self._first = True

    def next(self) -> str:
        word = self._rng.choice(_VOCAB)
        if self._first:
            self._first = False
            return word.capitalize()
        return f" {word}"
