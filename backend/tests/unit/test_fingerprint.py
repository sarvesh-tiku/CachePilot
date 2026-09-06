import pytest

from cachepilot.core.models import InferenceRequest, Message
from cachepilot.core.tokens import CHARS_PER_TOKEN
from cachepilot.kv.fingerprint import Fingerprinter

CHUNK_TOKENS = 8
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
FP = Fingerprinter(CHUNK_TOKENS)


def messages(system_chars: int, user_text: str = "") -> list[Message]:
    return [
        Message(role="system", content="s" * system_chars),
        Message(role="user", content=user_text),
    ]


def test_same_prefix_gives_same_hashes() -> None:
    a = FP.fingerprint(messages(200, "question A"))
    b = FP.fingerprint(messages(200, "question A"))
    assert a == b
    assert a.full is not None


def test_only_full_chunks_are_fingerprinted() -> None:
    assert FP.fingerprint([Message(role="user", content="short")]).chunk_hashes == ()
    exact = [Message(role="user", content="x" * (CHUNK_CHARS - len("user:\n")))]
    assert len(FP.fingerprint(exact).chunk_hashes) == 1


def test_shared_prefix_shares_leading_hashes_only() -> None:
    shared = "s" * (CHUNK_CHARS * 3)
    a = FP.fingerprint([Message(role="system", content=shared + "A" * CHUNK_CHARS)])
    b = FP.fingerprint([Message(role="system", content=shared + "B" * CHUNK_CHARS)])

    assert len(a.chunk_hashes) == len(b.chunk_hashes) == 4
    assert a.chunk_hashes[:3] == b.chunk_hashes[:3]
    assert a.chunk_hashes[3] != b.chunk_hashes[3]
    assert a.full != b.full


def test_hashes_are_cumulative_not_per_chunk() -> None:
    """The same chunk text at a different position must hash differently."""
    repeated = FP.fingerprint([Message(role="system", content="r" * (CHUNK_CHARS * 2))])
    assert repeated.chunk_hashes[0] != repeated.chunk_hashes[1]


def test_chunk_size_changes_hashes() -> None:
    m = messages(400)
    assert (
        Fingerprinter(8).fingerprint(m).chunk_hashes[0]
        != Fingerprinter(16).fingerprint(m).chunk_hashes[0]
    )


def test_apply_populates_request_fields() -> None:
    request = InferenceRequest(
        model="m", messages=messages(200), prompt_tokens_estimate=52, max_tokens=4
    )
    stamped = FP.apply(request)
    assert stamped.request_id == request.request_id
    assert stamped.prefix_hashes
    assert stamped.prefix_fingerprint == stamped.prefix_hashes[-1]
    assert request.prefix_hashes == []


def test_prompt_tokens_matches_estimate() -> None:
    fp = FP.fingerprint(messages(200, "q"))
    assert fp.prompt_tokens == -(-len("system:" + "s" * 200 + "\nuser:q\n") // CHARS_PER_TOKEN)


def test_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        Fingerprinter(0)
