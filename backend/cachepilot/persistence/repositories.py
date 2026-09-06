from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from cachepilot.core.models import (
    CandidateScore,
    EventType,
    InferenceRequest,
    RequestRecord,
    RequestResult,
    RequestView,
    RoutingDecision,
    SystemEvent,
)
from cachepilot.persistence.schema import (
    EventRow,
    RequestResultRow,
    RequestRow,
    RoutingDecisionRow,
)


class RunStore(Protocol):
    async def save_request(self, request: InferenceRequest) -> None: ...

    async def save_decision(self, decision: RoutingDecision) -> None: ...

    async def save_result(self, result: RequestResult) -> None: ...

    async def get_decision(self, decision_id: uuid.UUID) -> RoutingDecision | None: ...

    async def get_result(self, request_id: uuid.UUID) -> RequestResult | None: ...

    async def list_decisions(
        self,
        *,
        policy: str | None = None,
        worker_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[RoutingDecision]: ...

    async def get_request_view(self, request_id: uuid.UUID) -> RequestView | None: ...

    async def list_request_views(
        self,
        *,
        policy: str | None = None,
        worker_id: str | None = None,
        limit: int = 100,
    ) -> list[RequestView]: ...

    async def list_results(self, *, since: datetime) -> list[RequestResult]: ...

    async def save_event(self, event: SystemEvent) -> None: ...

    async def list_events(
        self,
        *,
        request_id: uuid.UUID | None = None,
        event_type: EventType | None = None,
        limit: int = 200,
        ascending: bool = False,
    ) -> list[SystemEvent]: ...


class InMemoryRunStore:
    """Zero-I/O store for benchmarks and tests. Never blocks the event loop."""

    def __init__(self) -> None:
        self.requests: dict[uuid.UUID, RequestRecord] = {}
        self.decisions: dict[uuid.UUID, RoutingDecision] = {}
        self.results: dict[uuid.UUID, RequestResult] = {}
        self.events: list[SystemEvent] = []
        self._decision_by_request: dict[uuid.UUID, RoutingDecision] = {}

    async def save_request(self, request: InferenceRequest) -> None:
        self.requests[request.request_id] = RequestRecord.from_request(request)

    async def save_decision(self, decision: RoutingDecision) -> None:
        self.decisions[decision.decision_id] = decision
        self._decision_by_request[decision.request_id] = decision

    async def save_result(self, result: RequestResult) -> None:
        self.results[result.request_id] = result

    async def get_decision(self, decision_id: uuid.UUID) -> RoutingDecision | None:
        return self.decisions.get(decision_id)

    async def get_result(self, request_id: uuid.UUID) -> RequestResult | None:
        return self.results.get(request_id)

    async def list_decisions(
        self,
        *,
        policy: str | None = None,
        worker_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[RoutingDecision]:
        rows = sorted(self.decisions.values(), key=lambda d: d.created_at, reverse=True)
        if policy is not None:
            rows = [d for d in rows if d.policy == policy]
        if worker_id is not None:
            rows = [d for d in rows if d.selected_worker_id == worker_id]
        if since is not None:
            rows = [d for d in rows if d.created_at >= since]
        return rows[:limit]

    async def get_request_view(self, request_id: uuid.UUID) -> RequestView | None:
        record = self.requests.get(request_id)
        return self._view(record) if record else None

    async def list_request_views(
        self,
        *,
        policy: str | None = None,
        worker_id: str | None = None,
        limit: int = 100,
    ) -> list[RequestView]:
        views = [
            self._view(r)
            for r in sorted(self.requests.values(), key=lambda r: r.created_at, reverse=True)
        ]
        if policy is not None:
            views = [v for v in views if v.decision and v.decision.policy == policy]
        if worker_id is not None:
            views = [v for v in views if v.decision and v.decision.selected_worker_id == worker_id]
        return views[:limit]

    async def list_results(self, *, since: datetime) -> list[RequestResult]:
        return [r for r in self.results.values() if r.completed_at >= since]

    async def save_event(self, event: SystemEvent) -> None:
        self.events.append(event)

    async def list_events(
        self,
        *,
        request_id: uuid.UUID | None = None,
        event_type: EventType | None = None,
        limit: int = 200,
        ascending: bool = False,
    ) -> list[SystemEvent]:
        rows = self.events
        if request_id is not None:
            rows = [e for e in rows if e.request_id == request_id]
        if event_type is not None:
            rows = [e for e in rows if e.event_type == event_type]
        rows = rows if ascending else list(reversed(rows))
        return rows[:limit]

    def _view(self, record: RequestRecord) -> RequestView:
        return RequestView(
            request=record,
            decision=self._decision_by_request.get(record.request_id),
            result=self.results.get(record.request_id),
        )


class SqlRunStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        # A StaticPool (in-memory SQLite) is one shared connection: concurrent sessions
        # would see and roll back each other's transactions, so serialize them.
        self._lock = asyncio.Lock() if isinstance(engine.pool, StaticPool) else None

    @asynccontextmanager
    async def _session(self, *, write: bool) -> AsyncIterator[AsyncSession]:
        if self._lock is not None:
            async with self._lock:
                async with self._open(write) as session:
                    yield session
        else:
            async with self._open(write) as session:
                yield session

    @asynccontextmanager
    async def _open(self, write: bool) -> AsyncIterator[AsyncSession]:
        if write:
            async with self._sessions.begin() as session:
                yield session
        else:
            async with self._sessions() as session:
                yield session

    async def save_request(self, request: InferenceRequest) -> None:
        async with self._session(write=True) as session:
            session.add(
                RequestRow(
                    request_id=str(request.request_id),
                    model=request.model,
                    message_count=len(request.messages),
                    prompt_tokens_estimate=request.prompt_tokens_estimate,
                    max_tokens=request.max_tokens,
                    latency_slo_ms=request.latency_slo_ms,
                    prefix_fingerprint=request.prefix_fingerprint,
                    created_at=request.created_at,
                )
            )

    async def save_decision(self, decision: RoutingDecision) -> None:
        async with self._session(write=True) as session:
            session.add(
                RoutingDecisionRow(
                    decision_id=str(decision.decision_id),
                    request_id=str(decision.request_id),
                    policy=decision.policy,
                    selected_worker_id=decision.selected_worker_id,
                    candidates=[c.model_dump() for c in decision.candidates],
                    reason=decision.reason,
                    created_at=decision.created_at,
                )
            )

    async def save_result(self, result: RequestResult) -> None:
        async with self._session(write=True) as session:
            session.add(
                RequestResultRow(
                    request_id=str(result.request_id),
                    worker_id=result.worker_id,
                    queue_wait_ms=result.queue_wait_ms,
                    ttft_ms=result.ttft_ms,
                    total_latency_ms=result.total_latency_ms,
                    output_tokens=result.output_tokens,
                    cache_hit=result.cache_hit,
                    cache_overlap=result.cache_overlap,
                    completed_at=result.completed_at,
                )
            )

    async def get_decision(self, decision_id: uuid.UUID) -> RoutingDecision | None:
        async with self._session(write=False) as session:
            row = await session.get(RoutingDecisionRow, str(decision_id))
            return _decision_from_row(row) if row else None

    async def get_result(self, request_id: uuid.UUID) -> RequestResult | None:
        async with self._session(write=False) as session:
            row = await session.get(RequestResultRow, str(request_id))
            return _result_from_row(row) if row else None

    async def list_decisions(
        self,
        *,
        policy: str | None = None,
        worker_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[RoutingDecision]:
        stmt = select(RoutingDecisionRow).order_by(RoutingDecisionRow.created_at.desc())
        if policy is not None:
            stmt = stmt.where(RoutingDecisionRow.policy == policy)
        if worker_id is not None:
            stmt = stmt.where(RoutingDecisionRow.selected_worker_id == worker_id)
        if since is not None:
            stmt = stmt.where(RoutingDecisionRow.created_at >= since)
        stmt = stmt.limit(limit)
        async with self._session(write=False) as session:
            rows = (await session.scalars(stmt)).all()
        return [_decision_from_row(row) for row in rows]

    async def get_request_view(self, request_id: uuid.UUID) -> RequestView | None:
        stmt = self._view_query().where(RequestRow.request_id == str(request_id))
        async with self._session(write=False) as session:
            row = (await session.execute(stmt)).first()
        return _view_from_row(row[0], row[1], row[2]) if row else None

    async def list_request_views(
        self,
        *,
        policy: str | None = None,
        worker_id: str | None = None,
        limit: int = 100,
    ) -> list[RequestView]:
        stmt = self._view_query().order_by(RequestRow.created_at.desc()).limit(limit)
        if policy is not None:
            stmt = stmt.where(RoutingDecisionRow.policy == policy)
        if worker_id is not None:
            stmt = stmt.where(RoutingDecisionRow.selected_worker_id == worker_id)
        async with self._session(write=False) as session:
            rows = (await session.execute(stmt)).all()
        return [_view_from_row(r[0], r[1], r[2]) for r in rows]

    async def list_results(self, *, since: datetime) -> list[RequestResult]:
        stmt = select(RequestResultRow).where(RequestResultRow.completed_at >= since)
        async with self._session(write=False) as session:
            rows = (await session.scalars(stmt)).all()
        return [_result_from_row(row) for row in rows]

    async def save_event(self, event: SystemEvent) -> None:
        async with self._session(write=True) as session:
            session.add(
                EventRow(
                    event_id=str(event.event_id),
                    event_type=event.event_type.value,
                    request_id=str(event.request_id) if event.request_id else None,
                    worker_id=event.worker_id,
                    timestamp=event.timestamp,
                    payload=event.payload,
                )
            )

    async def list_events(
        self,
        *,
        request_id: uuid.UUID | None = None,
        event_type: EventType | None = None,
        limit: int = 200,
        ascending: bool = False,
    ) -> list[SystemEvent]:
        order = EventRow.timestamp.asc() if ascending else EventRow.timestamp.desc()
        stmt = select(EventRow).order_by(order, EventRow.event_id).limit(limit)
        if request_id is not None:
            stmt = stmt.where(EventRow.request_id == str(request_id))
        if event_type is not None:
            stmt = stmt.where(EventRow.event_type == event_type.value)
        async with self._session(write=False) as session:
            rows = (await session.scalars(stmt)).all()
        return [
            SystemEvent(
                event_id=uuid.UUID(row.event_id),
                event_type=EventType(row.event_type),
                request_id=uuid.UUID(row.request_id) if row.request_id else None,
                worker_id=row.worker_id,
                timestamp=_aware(row.timestamp),
                payload=row.payload,
            )
            for row in rows
        ]

    @staticmethod
    def _view_query():  # type: ignore[no-untyped-def]  # SQLAlchemy Select generics are unwieldy
        return (
            select(RequestRow, RoutingDecisionRow, RequestResultRow)
            .outerjoin(RoutingDecisionRow, RoutingDecisionRow.request_id == RequestRow.request_id)
            .outerjoin(RequestResultRow, RequestResultRow.request_id == RequestRow.request_id)
        )


def _aware(dt: datetime) -> datetime:
    # SQLite drops tzinfo; everything we store is UTC.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _record_from_row(row: RequestRow) -> RequestRecord:
    return RequestRecord(
        request_id=uuid.UUID(row.request_id),
        model=row.model,
        message_count=row.message_count,
        prompt_tokens_estimate=row.prompt_tokens_estimate,
        max_tokens=row.max_tokens,
        latency_slo_ms=row.latency_slo_ms,
        prefix_fingerprint=row.prefix_fingerprint,
        created_at=_aware(row.created_at),
    )


def _decision_from_row(row: RoutingDecisionRow) -> RoutingDecision:
    return RoutingDecision(
        decision_id=uuid.UUID(row.decision_id),
        request_id=uuid.UUID(row.request_id),
        policy=row.policy,
        selected_worker_id=row.selected_worker_id,
        candidates=[CandidateScore.model_validate(c) for c in row.candidates],
        reason=row.reason,
        created_at=_aware(row.created_at),
    )


def _result_from_row(row: RequestResultRow) -> RequestResult:
    return RequestResult(
        request_id=uuid.UUID(row.request_id),
        worker_id=row.worker_id,
        queue_wait_ms=row.queue_wait_ms,
        ttft_ms=row.ttft_ms,
        total_latency_ms=row.total_latency_ms,
        output_tokens=row.output_tokens,
        cache_hit=row.cache_hit,
        cache_overlap=row.cache_overlap,
        completed_at=_aware(row.completed_at),
    )


def _view_from_row(
    request: RequestRow,
    decision: RoutingDecisionRow | None,
    result: RequestResultRow | None,
) -> RequestView:
    return RequestView(
        request=_record_from_row(request),
        decision=_decision_from_row(decision) if decision else None,
        result=_result_from_row(result) if result else None,
    )
