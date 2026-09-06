from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from cachepilot.core.errors import WorkerNotFoundError
from cachepilot.core.models import KVCacheEntry
from cachepilot.kv.directory import InMemoryKVDirectory, WorkerCacheSummary

router = APIRouter(prefix="/api/kv", tags=["kv"])


def _directory(request: Request) -> InMemoryKVDirectory:
    directory: InMemoryKVDirectory = request.app.state.kv_directory
    return directory


@router.get("", response_model=list[WorkerCacheSummary])
async def list_cache_summaries(request: Request) -> list[WorkerCacheSummary]:
    return _directory(request).summaries()


@router.get("/{worker_id}", response_model=WorkerCacheSummary)
async def get_cache_summary(worker_id: str, request: Request) -> WorkerCacheSummary:
    try:
        return _directory(request).summary(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{worker_id}/entries", response_model=list[KVCacheEntry])
async def list_cache_entries(worker_id: str, request: Request) -> list[KVCacheEntry]:
    try:
        return _directory(request).entries(worker_id)
    except WorkerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
