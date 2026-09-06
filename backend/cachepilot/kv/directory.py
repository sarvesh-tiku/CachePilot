from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from pydantic import BaseModel

from cachepilot.core.errors import WorkerNotFoundError
from cachepilot.core.models import InferenceRequest, KVCacheEntry, utcnow


class WorkerCacheSummary(BaseModel):
    worker_id: str
    source: str  # "simulated": authoritative; "estimated": gateway-side guess about a real server
    capacity_bytes: int
    used_bytes: int
    entries: int
    hits: int
    misses: int
    evictions: int


@dataclass(frozen=True)
class AdmitOutcome:
    added_chunks: int
    evicted: list[KVCacheEntry]


@dataclass
class _WorkerCache:
    capacity_bytes: int
    source: str
    entries: OrderedDict[str, KVCacheEntry] = field(default_factory=OrderedDict)
    used_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class InMemoryKVDirectory:
    """Logical record of which prefix chunks each worker holds, with per-worker LRU.

    This is gateway-side residency metadata, not real KV tensors. For simulated
    workers it is authoritative; for real workers it will be an estimate.
    Eviction order depends only on access order, never on wall time.
    """

    def __init__(self, chunk_tokens: int, bytes_per_token: int) -> None:
        self.chunk_tokens = chunk_tokens
        self.chunk_bytes = chunk_tokens * bytes_per_token
        self._workers: dict[str, _WorkerCache] = {}

    def register_worker(
        self, worker_id: str, capacity_bytes: int, source: str = "simulated"
    ) -> None:
        self._workers[worker_id] = _WorkerCache(capacity_bytes=capacity_bytes, source=source)

    def cache_overlap(self, request: InferenceRequest, worker_id: str) -> float:
        if request.prompt_tokens_estimate <= 0:
            return 0.0
        matched = self._matched_chunks(request, self._cache(worker_id))
        return min(1.0, matched * self.chunk_tokens / request.prompt_tokens_estimate)

    def touch(self, request: InferenceRequest, worker_id: str) -> int:
        """Mark the reusable prefix as accessed. Returns the number of matched chunks."""
        cache = self._cache(worker_id)
        matched = self._matched_chunks(request, cache)
        now = utcnow()
        for prefix_hash in request.prefix_hashes[:matched]:
            entry = cache.entries[prefix_hash]
            entry.last_access = now
            entry.hit_count += 1
            cache.entries.move_to_end(prefix_hash)
        if matched:
            cache.hits += 1
        else:
            cache.misses += 1
        return matched

    def admit(self, request: InferenceRequest, worker_id: str) -> AdmitOutcome:
        """Record that every prefix chunk of the request is now resident, evicting LRU as needed."""
        cache = self._cache(worker_id)
        now = utcnow()
        added = 0
        for prefix_hash in request.prefix_hashes:
            existing = cache.entries.get(prefix_hash)
            if existing is not None:
                existing.last_access = now
                cache.entries.move_to_end(prefix_hash)
                continue
            cache.entries[prefix_hash] = KVCacheEntry(
                prefix_hash=prefix_hash,
                worker_id=worker_id,
                model=request.model,
                token_count=self.chunk_tokens,
                estimated_bytes=self.chunk_bytes,
                last_access=now,
            )
            cache.used_bytes += self.chunk_bytes
            added += 1

        evicted: list[KVCacheEntry] = []
        while cache.used_bytes > cache.capacity_bytes and cache.entries:
            _, victim = cache.entries.popitem(last=False)
            cache.used_bytes -= victim.estimated_bytes
            cache.evictions += 1
            evicted.append(victim)
        return AdmitOutcome(added_chunks=added, evicted=evicted)

    def used_bytes(self, worker_id: str) -> int:
        return self._cache(worker_id).used_bytes

    def entries(self, worker_id: str) -> list[KVCacheEntry]:
        return list(reversed(self._cache(worker_id).entries.values()))

    def summary(self, worker_id: str) -> WorkerCacheSummary:
        cache = self._cache(worker_id)
        return WorkerCacheSummary(
            worker_id=worker_id,
            source=cache.source,
            capacity_bytes=cache.capacity_bytes,
            used_bytes=cache.used_bytes,
            entries=len(cache.entries),
            hits=cache.hits,
            misses=cache.misses,
            evictions=cache.evictions,
        )

    def summaries(self) -> list[WorkerCacheSummary]:
        return [self.summary(worker_id) for worker_id in self._workers]

    def _cache(self, worker_id: str) -> _WorkerCache:
        try:
            return self._workers[worker_id]
        except KeyError:
            raise WorkerNotFoundError(worker_id) from None

    @staticmethod
    def _matched_chunks(request: InferenceRequest, cache: _WorkerCache) -> int:
        matched = 0
        for prefix_hash in request.prefix_hashes:
            if prefix_hash not in cache.entries:
                break
            matched += 1
        return matched
