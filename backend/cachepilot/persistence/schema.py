from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RequestRow(Base):
    """Request metadata only. Prompt text is never persisted."""

    __tablename__ = "requests"

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model: Mapped[str] = mapped_column(String(128))
    message_count: Mapped[int] = mapped_column(Integer)
    prompt_tokens_estimate: Mapped[int] = mapped_column(Integer)
    max_tokens: Mapped[int] = mapped_column(Integer)
    latency_slo_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prefix_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class RoutingDecisionRow(Base):
    __tablename__ = "routing_decisions"

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("requests.request_id"), index=True
    )
    policy: Mapped[str] = mapped_column(String(32), index=True)
    selected_worker_id: Mapped[str] = mapped_column(String(64), index=True)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class EventRow(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    request_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class RequestResultRow(Base):
    __tablename__ = "request_results"

    request_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("requests.request_id"), primary_key=True
    )
    worker_id: Mapped[str] = mapped_column(String(64), index=True)
    queue_wait_ms: Mapped[float] = mapped_column(Float)
    ttft_ms: Mapped[float] = mapped_column(Float)
    total_latency_ms: Mapped[float] = mapped_column(Float)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cache_hit: Mapped[bool] = mapped_column(Boolean)
    cache_overlap: Mapped[float] = mapped_column(Float)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
