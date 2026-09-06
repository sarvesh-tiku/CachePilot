from hypothesis import given, settings
from hypothesis import strategies as st

from cachepilot.core.models import InferenceRequest, Message
from cachepilot.kv.directory import InMemoryKVDirectory
from cachepilot.kv.fingerprint import Fingerprinter

CHUNK_TOKENS = 4

hash_lists = st.lists(st.sampled_from(list("abcdefgh")), max_size=8)
prompts = st.lists(
    st.builds(
        Message,
        role=st.sampled_from(["system", "user", "assistant"]),
        content=st.text(alphabet="abc \n", max_size=120),
    ),
    min_size=1,
    max_size=4,
)


def request(hashes: list[str]) -> InferenceRequest:
    return InferenceRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        prompt_tokens_estimate=max(1, len(hashes) * CHUNK_TOKENS),
        max_tokens=1,
        prefix_hashes=hashes,
    )


@given(
    capacity_chunks=st.integers(min_value=1, max_value=6),
    ops=st.lists(st.tuples(st.sampled_from(["admit", "touch"]), hash_lists), max_size=30),
    probe=hash_lists,
)
@settings(max_examples=200)
def test_used_never_exceeds_capacity_and_overlap_in_unit_interval(
    capacity_chunks: int, ops: list[tuple[str, list[str]]], probe: list[str]
) -> None:
    d = InMemoryKVDirectory(chunk_tokens=CHUNK_TOKENS, bytes_per_token=1)
    d.register_worker("w", capacity_bytes=capacity_chunks * CHUNK_TOKENS)

    for op, hashes in ops:
        if op == "admit":
            d.admit(request(hashes), "w")
        else:
            d.touch(request(hashes), "w")
        summary = d.summary("w")
        assert 0 <= summary.used_bytes <= summary.capacity_bytes
        assert summary.used_bytes == summary.entries * d.chunk_bytes

    assert 0.0 <= d.cache_overlap(request(probe), "w") <= 1.0


@given(messages=prompts)
@settings(max_examples=200)
def test_fingerprint_is_deterministic_and_prefix_consistent(messages: list[Message]) -> None:
    fp = Fingerprinter(CHUNK_TOKENS)
    a = fp.fingerprint(messages)
    b = fp.fingerprint(list(messages))
    assert a == b

    extended = fp.fingerprint(messages + [Message(role="user", content="tail")])
    assert extended.chunk_hashes[: len(a.chunk_hashes)] == a.chunk_hashes
