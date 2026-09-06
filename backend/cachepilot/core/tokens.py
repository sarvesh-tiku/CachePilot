from __future__ import annotations

import math

from cachepilot.core.models import Message

# Deterministic approximation; a real tokenizer is deliberately out of scope for v1.
CHARS_PER_TOKEN = 4


def normalize_prompt(messages: list[Message]) -> str:
    """Canonical prompt text. Token estimates and prefix fingerprints both derive from it."""
    return "".join(f"{m.role}:{m.content}\n" for m in messages)


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_prompt_tokens(messages: list[Message]) -> int:
    return estimate_text_tokens(normalize_prompt(messages))
