from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from cachepilot.core.models import RequestResult, RoutingDecision
from cachepilot.persistence.repositories import RunStore

router = APIRouter(prefix="/api/decisions", tags=["decisions"])


class DecisionDetail(BaseModel):
    decision: RoutingDecision
    result: RequestResult | None


def _store(request: Request) -> RunStore:
    store: RunStore = request.app.state.store
    return store


@router.get("", response_model=list[RoutingDecision])
async def list_decisions(
    request: Request,
    policy: str | None = None,
    worker_id: str | None = None,
    since: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RoutingDecision]:
    return await _store(request).list_decisions(
        policy=policy, worker_id=worker_id, since=since, limit=limit
    )


@router.get("/{decision_id}", response_model=DecisionDetail)
async def get_decision(decision_id: uuid.UUID, request: Request) -> DecisionDetail:
    store = _store(request)
    decision = await store.get_decision(decision_id)
    if decision is None:
        raise HTTPException(status_code=404, detail=f"decision not found: {decision_id}")
    result = await store.get_result(decision.request_id)
    return DecisionDetail(decision=decision, result=result)
