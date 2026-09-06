from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from cachepilot.core.models import Message
from cachepilot.core.tokens import CHARS_PER_TOKEN

WORKLOAD_DIR = Path(__file__).parent / "workloads"

_WORDS = (
    "policy", "section", "employee", "receipt", "approval", "travel", "expense", "manager",
    "quarter", "budget", "vendor", "contract", "review", "deadline", "invoice", "compliance",
    "the", "shall", "must", "within", "before", "after", "unless", "including",
)  # fmt: skip


class ArrivalSpec(BaseModel):
    kind: Literal["poisson", "bursty"] = "poisson"
    rate_rps: float = Field(default=20.0, gt=0)
    burst_size: int = Field(default=10, ge=1)
    burst_spacing_ms: float = Field(default=5.0, ge=0)
    burst_gap_mean_ms: float = Field(default=500.0, gt=0)


class WorkloadSpec(BaseModel):
    name: str
    description: str = ""
    prefix_tokens: int = Field(default=2048, ge=1)
    question_tokens: int = Field(default=32, ge=1)
    output_tokens: int = Field(default=64, ge=1)
    shared_prefix_count: int = Field(default=0, ge=0)
    shared_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    shared_skew: float = Field(default=0.0, ge=0.0)
    arrival: ArrivalSpec = Field(default_factory=ArrivalSpec)
    kv_capacity_bytes_per_worker: int | None = Field(default=None, ge=1)


class BenchmarkRequest(BaseModel):
    index: int
    arrival_ms: float
    prefix_id: str
    messages: list[Message]
    max_tokens: int


def load_workload(name_or_path: str) -> WorkloadSpec:
    path = Path(name_or_path)
    if not path.suffix:
        path = WORKLOAD_DIR / f"{name_or_path}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in WORKLOAD_DIR.glob("*.json")))
        raise FileNotFoundError(f"workload {name_or_path!r} not found; available: {available}")
    return WorkloadSpec.model_validate(json.loads(path.read_text()))


def generate(spec: WorkloadSpec, *, seed: int, count: int) -> list[BenchmarkRequest]:
    """Materialize a deterministic request stream from a workload spec and seed."""
    rng = random.Random(f"{spec.name}:{seed}")
    prefix_texts: dict[str, str] = {}

    def prefix_text(prefix_id: str) -> str:
        if prefix_id not in prefix_texts:
            prefix_texts[prefix_id] = _words(
                random.Random(f"{seed}:{prefix_id}"), spec.prefix_tokens * CHARS_PER_TOKEN
            )
        return prefix_texts[prefix_id]

    shared_weights = [
        1.0 / (rank + 1) ** spec.shared_skew for rank in range(spec.shared_prefix_count)
    ]

    def pick_shared() -> str:
        rank = rng.choices(range(spec.shared_prefix_count), weights=shared_weights)[0]
        return f"shared-{rank}"

    def pick_prefix(index: int) -> str:
        if spec.shared_prefix_count and rng.random() < spec.shared_fraction:
            return pick_shared()
        return f"unique-{index}"

    requests: list[BenchmarkRequest] = []
    if spec.arrival.kind == "poisson":
        t = 0.0
        for index in range(count):
            t += rng.expovariate(spec.arrival.rate_rps) * 1000.0
            requests.append(_request(spec, rng, index, t, pick_prefix(index), prefix_text))
    else:
        t = 0.0
        index = 0
        while index < count:
            burst_prefix = pick_shared() if spec.shared_prefix_count else None
            for slot in range(spec.arrival.burst_size):
                if index >= count:
                    break
                arrival = t + slot * spec.arrival.burst_spacing_ms
                prefix_id = (
                    burst_prefix
                    if burst_prefix is not None and rng.random() < spec.shared_fraction
                    else f"unique-{index}"
                )
                requests.append(_request(spec, rng, index, arrival, prefix_id, prefix_text))
                index += 1
            t += spec.arrival.burst_size * spec.arrival.burst_spacing_ms
            t += rng.expovariate(1.0 / spec.arrival.burst_gap_mean_ms)
    return requests


def _request(
    spec: WorkloadSpec,
    rng: random.Random,
    index: int,
    arrival_ms: float,
    prefix_id: str,
    prefix_text: Callable[[str], str],
) -> BenchmarkRequest:
    question = f"Q{index}: " + _words(rng, spec.question_tokens * CHARS_PER_TOKEN)
    return BenchmarkRequest(
        index=index,
        arrival_ms=arrival_ms,
        prefix_id=prefix_id,
        messages=[
            Message(role="system", content=prefix_text(prefix_id)),
            Message(role="user", content=question),
        ],
        max_tokens=spec.output_tokens,
    )


def _words(rng: random.Random, chars: int) -> str:
    parts: list[str] = []
    length = -1  # join adds one fewer separator than words
    while length < chars:
        word = rng.choice(_WORDS)
        parts.append(word)
        length += len(word) + 1
    return " ".join(parts)[:chars]
