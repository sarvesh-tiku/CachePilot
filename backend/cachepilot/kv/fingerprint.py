from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cachepilot.core.models import InferenceRequest, Message
from cachepilot.core.tokens import CHARS_PER_TOKEN, estimate_text_tokens, normalize_prompt

_HASH_HEX_CHARS = 32


@dataclass(frozen=True)
class PrefixFingerprint:
    chunk_tokens: int
    chunk_hashes: tuple[str, ...]
    prompt_tokens: int

    @property
    def full(self) -> str | None:
        return self.chunk_hashes[-1] if self.chunk_hashes else None


class Fingerprinter:
    """Deterministic cumulative hashes over fixed-size prefix chunks.

    chunk_hashes[i] covers chunks 0..i, so two prompts share exactly the leading
    hashes that their shared prefix covers. Only full chunks count: a partial
    trailing chunk is never reusable, matching how block-based KV caches behave.
    """

    def __init__(self, chunk_tokens: int) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        self.chunk_tokens = chunk_tokens
        self._chunk_chars = chunk_tokens * CHARS_PER_TOKEN

    def fingerprint(self, messages: list[Message]) -> PrefixFingerprint:
        text = normalize_prompt(messages)
        digest = hashlib.sha256(f"cachepilot:v1:{self.chunk_tokens}:".encode())
        hashes: list[str] = []
        for i in range(len(text) // self._chunk_chars):
            digest.update(text[i * self._chunk_chars : (i + 1) * self._chunk_chars].encode())
            hashes.append(digest.hexdigest()[:_HASH_HEX_CHARS])
        return PrefixFingerprint(
            chunk_tokens=self.chunk_tokens,
            chunk_hashes=tuple(hashes),
            prompt_tokens=estimate_text_tokens(text),
        )

    def apply(self, request: InferenceRequest) -> InferenceRequest:
        fp = self.fingerprint(request.messages)
        return request.model_copy(
            update={"prefix_hashes": list(fp.chunk_hashes), "prefix_fingerprint": fp.full}
        )
