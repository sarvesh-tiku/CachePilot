from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from cachepilot.core.models import RequestRecord, RequestResult, RequestView, RoutingDecision
from cachepilot.persistence.repositories import RunStore

router = APIRouter(prefix="/api/requests", tags=["requests"])


class RequestViewOut(BaseModel):
    request: RequestRecord
    decision: RoutingDecision | None
    result: RequestResult | None
    status: str

    @classmethod
    def from_view(cls, view: RequestView) -> RequestViewOut:
        return cls(
            request=view.request, decision=view.decision, result=view.result, status=view.status
        )


def _store(request: Request) -> RunStore:
    store: RunStore = request.app.state.store
    return store


@router.get("", response_model=list[RequestViewOut])
async def list_requests(
    request: Request,
    policy: str | None = None,
    worker_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RequestViewOut]:
    views = await _store(request).list_request_views(
        policy=policy, worker_id=worker_id, limit=limit
    )
    return [RequestViewOut.from_view(v) for v in views]


@router.get("/{request_id}", response_model=RequestViewOut)
async def get_request(request_id: uuid.UUID, request: Request) -> RequestViewOut:
    view = await _store(request).get_request_view(request_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"request not found: {request_id}")
    return RequestViewOut.from_view(view)
