from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from cachepilot.core.models import InferenceRequest
from cachepilot.workers.base import DoneEvent, GenerationEvent, StatsListener, TokenEvent

_PROM_LINE = re.compile(r"^(vllm:[A-Za-z0-9_]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")
_THROUGHPUT_ALPHA = 0.3


@dataclass
class ObservedStats:
    """What the backend itself reports. Absent fields mean the server did not expose them."""

    running: int | None = None
    waiting: int | None = None
    kv_usage_fraction: float | None = None
    tokens_per_second: float | None = None


class VLLMWorker:
    """Adapter for any OpenAI-compatible streaming server (vLLM, SGLang, llama.cpp, Ollama).

    Timings are *observed* on the wall clock from the gateway's side: TTFT is
    the arrival of the first content delta, total latency the end of the
    stream. Queue wait inside the server is not exposed by the OpenAI API and
    is reported as 0. The full prompt is always sent; the server does its own
    prefix caching, so `cache_overlap` here only reflects the gateway's
    *estimate* of what the server still holds.
    """

    backend = "vllm"

    def __init__(
        self,
        worker_id: str,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
        on_stats_changed: StatsListener | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.worker_id = worker_id
        self.model = model
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout_s, headers=headers
        )
        self._owns_client = client is None
        self.on_stats_changed = on_stats_changed
        self._now = monotonic
        self.active_requests = 0
        self.queue_depth = 0
        self.observed = ObservedStats()

    async def generate(
        self, request: InferenceRequest, cache_overlap: float = 0.0
    ) -> AsyncIterator[GenerationEvent]:
        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in request.messages],
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        self.active_requests += 1
        await self._notify()
        started = self._now()
        first_token_ms: float | None = None
        index = 0
        finish_reason = "stop"
        usage_tokens: int | None = None
        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        usage_tokens = int(usage["completion_tokens"])
                    for choice in chunk.get("choices", []):
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            if first_token_ms is None:
                                first_token_ms = (self._now() - started) * 1000.0
                            yield TokenEvent(index=index, text=content)
                            index += 1
            total_ms = (self._now() - started) * 1000.0
            output_tokens = usage_tokens if usage_tokens is not None else index
            ttft_ms = first_token_ms if first_token_ms is not None else total_ms
            self._update_throughput(output_tokens, total_ms - ttft_ms)
            yield DoneEvent(
                queue_wait_ms=0.0,
                ttft_ms=ttft_ms,
                total_latency_ms=total_ms,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )
        finally:
            self.active_requests -= 1
            await self._notify()

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            if resp.status_code == 200:
                return True
            resp = await self._client.get("/v1/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def stats(self) -> dict[str, Any]:
        """Scrape vLLM's Prometheus endpoint if present; tolerate servers without one."""
        try:
            resp = await self._client.get("/metrics")
        except httpx.HTTPError:
            return self.heartbeat_snapshot()
        if resp.status_code == 200:
            self.observed = parse_vllm_metrics(resp.text, self.observed)
        return {**self.heartbeat_snapshot(), "kv_usage_fraction": self.observed.kv_usage_fraction}

    def heartbeat_snapshot(self) -> dict[str, Any]:
        running = self.observed.running
        return {
            "queue_depth": self.observed.waiting if self.observed.waiting is not None else 0,
            "active_requests": max(self.active_requests, running or 0),
            "tokens_per_second_estimate": self.observed.tokens_per_second or 0.0,
        }

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _update_throughput(self, output_tokens: int, decode_ms: float) -> None:
        if output_tokens < 2 or decode_ms <= 0:
            return
        sample = (output_tokens - 1) / (decode_ms / 1000.0)
        previous = self.observed.tokens_per_second
        self.observed.tokens_per_second = (
            sample if previous is None else previous + _THROUGHPUT_ALPHA * (sample - previous)
        )

    async def _notify(self) -> None:
        if self.on_stats_changed is not None:
            await self.on_stats_changed(self)


def parse_vllm_metrics(text: str, previous: ObservedStats) -> ObservedStats:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _PROM_LINE.match(line.strip())
        if match:
            name, value = match.groups()
            totals[name] = totals.get(name, 0.0) + float(value)

    def get(name: str) -> float | None:
        return totals.get(name)

    running = get("vllm:num_requests_running")
    waiting = get("vllm:num_requests_waiting")
    kv = get("vllm:gpu_cache_usage_perc")
    return ObservedStats(
        running=int(running) if running is not None else previous.running,
        waiting=int(waiting) if waiting is not None else previous.waiting,
        kv_usage_fraction=kv if kv is not None else previous.kv_usage_fraction,
        tokens_per_second=previous.tokens_per_second,
    )
