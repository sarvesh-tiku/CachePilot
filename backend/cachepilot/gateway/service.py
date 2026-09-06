from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

from cachepilot.core.errors import UnknownPolicyError
from cachepilot.core.models import EventType, InferenceRequest, RequestResult, RoutingDecision
from cachepilot.kv.directory import InMemoryKVDirectory
from cachepilot.kv.fingerprint import Fingerprinter
from cachepilot.persistence.repositories import RunStore
from cachepilot.scheduler.base import Scheduler
from cachepilot.telemetry.events import Telemetry
from cachepilot.workers.base import DoneEvent, GenerationEvent, InferenceWorker, TokenEvent
from cachepilot.workers.registry import WorkerRegistry


@dataclass
class Submission:
    request: InferenceRequest
    decision: RoutingDecision
    cache_overlap: float
    events: AsyncIterator[GenerationEvent]


class InferenceGateway:
    """Routes one request:

    fingerprint -> schedule -> persist decision -> touch prefix -> stream from
    worker -> admit prefix to directory -> persist result. Every step emits an
    operational event.
    """

    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        workers: Mapping[str, InferenceWorker],
        schedulers: Mapping[str, Scheduler],
        fingerprinter: Fingerprinter,
        kv_directory: InMemoryKVDirectory,
        store: RunStore,
        telemetry: Telemetry,
        default_policy: str,
    ) -> None:
        if default_policy not in schedulers:
            raise UnknownPolicyError(default_policy, sorted(schedulers))
        self._registry = registry
        self._workers = workers
        self._schedulers = schedulers
        self._fingerprinter = fingerprinter
        self._kv = kv_directory
        self._store = store
        self._telemetry = telemetry
        self._default_policy = default_policy

    @property
    def policies(self) -> list[str]:
        return sorted(self._schedulers)

    @property
    def default_policy(self) -> str:
        return self._default_policy

    def describe_policies(self) -> dict[str, dict[str, float]]:
        return {name: self._schedulers[name].parameters() for name in self.policies}

    async def submit(self, request: InferenceRequest, policy: str | None = None) -> Submission:
        policy_name = policy or self._default_policy
        try:
            scheduler = self._schedulers[policy_name]
        except KeyError:
            raise UnknownPolicyError(policy_name, self.policies) from None

        emit = self._telemetry.emit
        rid = request.request_id
        await emit(
            EventType.REQUEST_RECEIVED,
            request_id=rid,
            model=request.model,
            prompt_tokens=request.prompt_tokens_estimate,
            max_tokens=request.max_tokens,
            policy=policy_name,
        )

        request = self._fingerprinter.apply(request)
        await emit(
            EventType.REQUEST_FINGERPRINTED,
            request_id=rid,
            prefix_chunks=len(request.prefix_hashes),
            prefix_fingerprint=request.prefix_fingerprint,
        )

        workers = await self._registry.list_all()
        decision = await scheduler.choose_worker(request, workers, self._kv)
        await self._store.save_request(request)
        await self._store.save_decision(decision)

        worker = self._workers[decision.selected_worker_id]
        overlap = self._kv.cache_overlap(request, worker.worker_id)
        selected = next(c for c in decision.candidates if c.worker_id == worker.worker_id)
        await emit(
            EventType.ROUTING_DECISION,
            request_id=rid,
            worker_id=worker.worker_id,
            decision_id=str(decision.decision_id),
            policy=decision.policy,
            cache_overlap=round(overlap, 4),
            queue_depth=selected.queue_depth,
            predicted_ttft_ms=round(selected.estimated_ttft_ms, 1),
            candidates=len(decision.candidates),
            reason=decision.reason,
        )
        return Submission(request, decision, overlap, self._run(worker, request, decision, overlap))

    async def _run(
        self,
        worker: InferenceWorker,
        request: InferenceRequest,
        decision: RoutingDecision,
        cache_overlap: float,
    ) -> AsyncIterator[GenerationEvent]:
        emit = self._telemetry.emit
        metrics = self._telemetry.metrics
        rid, wid = request.request_id, worker.worker_id

        matched = self._kv.touch(request, wid)
        if matched:
            metrics.cache_hits_total.labels(wid).inc()
            await emit(
                EventType.CACHE_HIT,
                request_id=rid,
                worker_id=wid,
                matched_chunks=matched,
                cache_overlap=round(cache_overlap, 4),
            )
        else:
            metrics.cache_misses_total.labels(wid).inc()
            await emit(EventType.CACHE_MISS, request_id=rid, worker_id=wid)

        first_token_seen = False
        async for event in worker.generate(request, cache_overlap):
            if isinstance(event, TokenEvent) and not first_token_seen:
                first_token_seen = True
                await emit(EventType.FIRST_TOKEN, request_id=rid, worker_id=wid)
            elif isinstance(event, DoneEvent):
                result = RequestResult(
                    request_id=rid,
                    worker_id=wid,
                    queue_wait_ms=event.queue_wait_ms,
                    ttft_ms=event.ttft_ms,
                    total_latency_ms=event.total_latency_ms,
                    output_tokens=event.output_tokens,
                    cache_hit=cache_overlap > 0.0,
                    cache_overlap=cache_overlap,
                )
                await self._store.save_result(result)
                metrics.observe_request(decision.policy, result)
                await emit(
                    EventType.REQUEST_COMPLETED,
                    request_id=rid,
                    worker_id=wid,
                    policy=decision.policy,
                    queue_wait_ms=round(event.queue_wait_ms, 1),
                    ttft_ms=round(event.ttft_ms, 1),
                    total_latency_ms=round(event.total_latency_ms, 1),
                    output_tokens=event.output_tokens,
                    cache_hit=cache_overlap > 0.0,
                )

                outcome = self._kv.admit(request, wid)
                await self._registry.update_kv_usage(wid, self._kv.used_bytes(wid))
                if outcome.added_chunks:
                    await emit(
                        EventType.CACHE_ADMITTED,
                        request_id=rid,
                        worker_id=wid,
                        added_chunks=outcome.added_chunks,
                        used_bytes=self._kv.used_bytes(wid),
                    )
                if outcome.evicted:
                    metrics.cache_evictions_total.labels(wid).inc(len(outcome.evicted))
                    await emit(
                        EventType.CACHE_EVICTED,
                        request_id=rid,
                        worker_id=wid,
                        evicted_chunks=len(outcome.evicted),
                        evicted_bytes=sum(e.estimated_bytes for e in outcome.evicted),
                        used_bytes=self._kv.used_bytes(wid),
                    )
            yield event
