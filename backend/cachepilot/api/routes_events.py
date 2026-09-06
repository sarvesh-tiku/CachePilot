from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Request

from cachepilot.core.models import EventType, SystemEvent
from cachepilot.persistence.repositories import RunStore

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=list[SystemEvent])
async def list_events(
    request: Request,
    request_id: uuid.UUID | None = None,
    event_type: EventType | None = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 200,
    ascending: bool = False,
) -> list[SystemEvent]:
    store: RunStore = request.app.state.store
    return await store.list_events(
        request_id=request_id, event_type=event_type, limit=limit, ascending=ascending
    )
