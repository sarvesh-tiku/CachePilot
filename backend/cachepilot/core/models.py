from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class WorkerStatus(StrEnum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


class EventType(StrEnum):
    REQUEST_RECEIVED = "REQUEST_RECEIVED"
    REQUEST_FINGERPRINTED = "REQUEST_FINGERPRINTED"
    ROUTING_DECISION = "ROUTING_DECISION"
    REQUEST_QUEUED = "REQUEST_QUEUED"
    INFERENCE_STARTED = "INFERENCE_STARTED"
    FIRST_TOKEN = "FIRST_TOKEN"
    REQUEST_COMPLETED = "REQUEST_COMPLETED"
    CACHE_ADMITTED = "CACHE_ADMITTED"
    CACHE_HIT = "CACHE_HIT"
    CACHE_MISS = "CACHE_MISS"
    CACHE_EVICTED = "CACHE_EVICTED"
    WORKER_HEARTBEAT = "WORKER_HEARTBEAT"
    WORKER_UNHEALTHY = "WORKER_UNHEALTHY"
    WORKER_RECOVERED = "WORKER_RECOVERED"
    WORKER_DRAINING = "WORKER_DRAINING"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InferenceRequest(BaseModel):
    request_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    model: str
    messages: list[Message]
    prompt_tokens_estimate: int
    max_tokens: int
    created_at: datetime = Field(default_factory=utcnow)
    latency_slo_ms: int | None = None
    # prefix_hashes[i] identifies the first i+1 prefix chunks; prefix_fingerprint is the last.
    prefix_hashes: list[str] = Field(default_factory=list)
    prefix_fingerprint: str | None = None


class WorkerState(BaseModel):
    worker_id: str
    model: str
    backend: str = "simulated"
    status: WorkerStatus = WorkerStatus.HEALTHY
    queue_depth: int = 0
    active_requests: int = 0
    kv_capacity_bytes: int
    kv_used_bytes: int = 0
    tokens_per_second_estimate: float = 0.0
    last_heartbeat: datetime = Field(default_factory=utcnow)


class KVCacheEntry(BaseModel):
    prefix_hash: str
    worker_id: str
    model: str
    token_count: int
    estimated_bytes: int
    last_access: datetime = Field(default_factory=utcnow)
    hit_count: int = 0


class CandidateScore(BaseModel):
    worker_id: str
    cache_overlap: float
    queue_depth: int
    active_requests: int = 0
    estimated_ttft_ms: float
    kv_pressure: float
    final_score: float


class RoutingDecision(BaseModel):
    decision_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    request_id: uuid.UUID
    policy: str
    selected_worker_id: str
    candidates: list[CandidateScore]
    reason: str
    created_at: datetime = Field(default_factory=utcnow)


class RequestResult(BaseModel):
    request_id: uuid.UUID
    worker_id: str
    queue_wait_ms: float
    ttft_ms: float
    total_latency_ms: float
    output_tokens: int
    cache_hit: bool
    cache_overlap: float
    completed_at: datetime = Field(default_factory=utcnow)


class RequestRecord(BaseModel):
    """What is persisted about a request: metadata only, never prompt text."""

    request_id: uuid.UUID
    model: str
    message_count: int
    prompt_tokens_estimate: int
    max_tokens: int
    latency_slo_ms: int | None = None
    prefix_fingerprint: str | None = None
    created_at: datetime

    @classmethod
    def from_request(cls, request: InferenceRequest) -> RequestRecord:
        return cls(
            request_id=request.request_id,
            model=request.model,
            message_count=len(request.messages),
            prompt_tokens_estimate=request.prompt_tokens_estimate,
            max_tokens=request.max_tokens,
            latency_slo_ms=request.latency_slo_ms,
            prefix_fingerprint=request.prefix_fingerprint,
            created_at=request.created_at,
        )


class RequestView(BaseModel):
    request: RequestRecord
    decision: RoutingDecision | None = None
    result: RequestResult | None = None

    @property
    def status(self) -> str:
        if self.result is not None:
            return "completed"
        return "in_flight" if self.decision is not None else "received"


class SystemEvent(BaseModel):
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: EventType
    request_id: uuid.UUID | None = None
    worker_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)
