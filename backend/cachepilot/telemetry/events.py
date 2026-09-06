from __future__ import annotations

import logging
import uuid
from typing import Any

from cachepilot.core.models import EventType, SystemEvent
from cachepilot.persistence.repositories import RunStore
from cachepilot.telemetry.prometheus import Metrics

INFO_EVENTS = frozenset(
    {
        EventType.ROUTING_DECISION,
        EventType.REQUEST_COMPLETED,
        EventType.CACHE_EVICTED,
        EventType.WORKER_UNHEALTHY,
        EventType.WORKER_RECOVERED,
        EventType.WORKER_DRAINING,
    }
)


class Telemetry:
    """Append-only operational events: persisted, logged as JSON, and counted."""

    def __init__(
        self, store: RunStore, metrics: Metrics, logger: logging.Logger | None = None
    ) -> None:
        self._store = store
        self.metrics = metrics
        self._logger = logger

    async def emit(
        self,
        event_type: EventType,
        *,
        request_id: uuid.UUID | None = None,
        worker_id: str | None = None,
        **payload: Any,
    ) -> SystemEvent:
        event = SystemEvent(
            event_type=event_type, request_id=request_id, worker_id=worker_id, payload=payload
        )
        await self._store.save_event(event)
        if self._logger is not None:
            level = logging.INFO if event_type in INFO_EVENTS else logging.DEBUG
            if self._logger.isEnabledFor(level):
                fields = {
                    "event_id": str(event.event_id),
                    "request_id": str(request_id) if request_id else None,
                    "worker": worker_id,
                    **payload,
                }
                self._logger.log(level, event_type.value.lower(), extra={"fields": fields})
        return event
