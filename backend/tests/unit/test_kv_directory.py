import pytest

from cachepilot.core.errors import WorkerNotFoundError
from cachepilot.core.models import InferenceRequest, Message
from cachepilot.kv.directory import InMemoryKVDirectory

CHUNK_TOKENS = 10
BYTES_PER_TOKEN = 100
CHUNK_BYTES = CHUNK_TOKENS * BYTES_PER_TOKEN


def directory(capacity_chunks: int = 4) -> InMemoryKVDirectory:
    d = InMemoryKVDirectory(chunk_tokens=CHUNK_TOKENS, bytes_per_token=BYTES_PER_TOKEN)
    d.register_worker("worker-0", capacity_bytes=capacity_chunks * CHUNK_BYTES)
    d.register_worker("worker-1", capacity_bytes=capacity_chunks * CHUNK_BYTES)
    return d


def request(hashes: list[str], prompt_tokens: int | None = None) -> InferenceRequest:
    return InferenceRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        prompt_tokens_estimate=prompt_tokens
        if prompt_tokens is not None
        else len(hashes) * CHUNK_TOKENS,
        max_tokens=1,
        prefix_hashes=hashes,
        prefix_fingerprint=hashes[-1] if hashes else None,
    )


def test_empty_directory_has_zero_overlap() -> None:
    d = directory()
    assert d.cache_overlap(request(["a", "b"]), "worker-0") == 0.0
    assert d.summary("worker-0").used_bytes == 0


def test_admit_then_full_overlap_on_same_worker_only() -> None:
    d = directory()
    d.admit(request(["a", "b", "c"]), "worker-0")

    assert d.cache_overlap(request(["a", "b", "c"]), "worker-0") == 1.0
    assert d.cache_overlap(request(["a", "b", "c"]), "worker-1") == 0.0
    assert d.used_bytes("worker-0") == 3 * CHUNK_BYTES
    assert d.summary("worker-0").entries == 3


def test_longer_shared_prefix_gives_higher_overlap() -> None:
    d = directory()
    d.admit(request(["a", "b", "c", "d"]), "worker-0")

    one = d.cache_overlap(request(["a", "x", "y", "z"]), "worker-0")
    two = d.cache_overlap(request(["a", "b", "y", "z"]), "worker-0")
    three = d.cache_overlap(request(["a", "b", "c", "z"]), "worker-0")

    assert one == pytest.approx(0.25)
    assert two == pytest.approx(0.5)
    assert three == pytest.approx(0.75)


def test_partial_trailing_tokens_count_as_uncached() -> None:
    d = directory()
    d.admit(request(["a", "b"]), "worker-0")
    assert d.cache_overlap(request(["a", "b"], prompt_tokens=25), "worker-0") == pytest.approx(0.8)


def test_missing_middle_chunk_breaks_the_prefix_run() -> None:
    d = directory(capacity_chunks=10)
    d.admit(request(["a", "b", "c"]), "worker-0")
    d.admit(request(["a"]), "worker-0")
    d.admit(request(["q1", "q2", "q3", "q4", "q5", "q6", "q7"]), "worker-0")
    d.admit(request(["z1", "z2"]), "worker-0")
    # LRU evicted "b" and "c" (never touched again) but "a" survived.
    assert d.cache_overlap(request(["a", "b", "c"]), "worker-0") == pytest.approx(1 / 3)


def test_lru_eviction_respects_capacity_and_reports_victims() -> None:
    d = directory(capacity_chunks=3)
    d.admit(request(["a", "b", "c"]), "worker-0")
    outcome = d.admit(request(["d"]), "worker-0")

    assert d.used_bytes("worker-0") == 3 * CHUNK_BYTES
    assert [e.prefix_hash for e in outcome.evicted] == ["a"]
    assert outcome.added_chunks == 1
    assert d.summary("worker-0").evictions == 1
    assert d.cache_overlap(request(["a"]), "worker-0") == 0.0
    assert d.cache_overlap(request(["d"]), "worker-0") == 1.0


def test_touch_protects_recently_used_prefix_from_eviction() -> None:
    d = directory(capacity_chunks=3)
    d.admit(request(["a", "b", "c"]), "worker-0")
    d.touch(request(["a"]), "worker-0")
    outcome = d.admit(request(["d"]), "worker-0")

    assert [e.prefix_hash for e in outcome.evicted] == ["b"]
    assert d.cache_overlap(request(["a"]), "worker-0") == 1.0


def test_touch_counts_hits_misses_and_hit_count() -> None:
    d = directory()
    d.admit(request(["a", "b"]), "worker-0")

    assert d.touch(request(["a", "b", "c"]), "worker-0") == 2
    assert d.touch(request(["zzz"]), "worker-0") == 0

    summary = d.summary("worker-0")
    assert (summary.hits, summary.misses) == (1, 1)
    by_hash = {e.prefix_hash: e for e in d.entries("worker-0")}
    assert by_hash["a"].hit_count == 1 and by_hash["b"].hit_count == 1


def test_readmitting_existing_chunks_does_not_double_count() -> None:
    d = directory()
    d.admit(request(["a", "b"]), "worker-0")
    outcome = d.admit(request(["a", "b", "c"]), "worker-0")
    assert outcome.added_chunks == 1
    assert d.used_bytes("worker-0") == 3 * CHUNK_BYTES


def test_entries_are_most_recent_first() -> None:
    d = directory()
    d.admit(request(["a", "b", "c"]), "worker-0")
    d.touch(request(["a"]), "worker-0")
    assert [e.prefix_hash for e in d.entries("worker-0")] == ["a", "c", "b"]


def test_unknown_worker_raises() -> None:
    d = directory()
    with pytest.raises(WorkerNotFoundError):
        d.cache_overlap(request(["a"]), "nope")


def test_zero_prompt_tokens_gives_zero_overlap() -> None:
    d = directory()
    d.admit(request(["a"]), "worker-0")
    assert d.cache_overlap(request(["a"], prompt_tokens=0), "worker-0") == 0.0
