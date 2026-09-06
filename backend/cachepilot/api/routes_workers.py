from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from cachepilot.core.errors import WorkerNotFoundError
from cachepilot.core.models import WorkerState
from cachepilot.workers.registry import WorkerRegistry

router = APIRouter(prefix="/api/workers", tags=["workers"])


def _registry(request: Request) -> WorkerRegistry:
    registry: WorkerRegistry = request.app.state.registry
    return registry


@router.get("", response_model=list[WorkerState])
async def list_workers(request: Request) -> list[WorkerState]:
    return await _registry(request).list_all()


@router.get("/{worker_id}", response_model=WorkerState)
async def get_worker(worker_id: str, request: Request) -> WorkerState:
    try:
        return await _registry(request).get(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{worker_id}/drain", response_model=WorkerState)
async def drain_worker(worker_id: str, request: Request) -> WorkerState:
    """Stop routing new requests to a worker; in-flight work finishes normally."""
    try:
        return await _registry(request).drain(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{worker_id}/restore", response_model=WorkerState)
async def restore_worker(worker_id: str, request: Request) -> WorkerState:
    try:
        return await _registry(request).restore(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
