from collections import Counter
from pathlib import Path

import pytest

from benchmark.workloads import WORKLOAD_DIR, ArrivalSpec, WorkloadSpec, generate, load_workload
from cachepilot.core.tokens import CHARS_PER_TOKEN, estimate_prompt_tokens


def spec(**overrides: object) -> WorkloadSpec:
    base = {
        "name": "t",
        "prefix_tokens": 64,
        "question_tokens": 8,
        "output_tokens": 4,
        "shared_prefix_count": 4,
        "shared_fraction": 0.8,
    }
    return WorkloadSpec.model_validate({**base, **overrides})


def test_generation_is_deterministic_for_seed() -> None:
    a = generate(spec(), seed=1, count=50)
    b = generate(spec(), seed=1, count=50)
    c = generate(spec(), seed=2, count=50)
    assert a == b
    assert a != c


def test_arrivals_are_monotonic_and_indexed() -> None:
    reqs = generate(spec(), seed=3, count=100)
    assert [r.index for r in reqs] == list(range(100))
    assert all(x.arrival_ms <= y.arrival_ms for x, y in zip(reqs, reqs[1:], strict=False))
    assert reqs[0].arrival_ms > 0


def test_shared_fraction_roughly_respected() -> None:
    reqs = generate(spec(shared_fraction=0.8), seed=4, count=1000)
    shared = sum(1 for r in reqs if r.prefix_id.startswith("shared-"))
    assert 0.74 < shared / 1000 < 0.86
    assert len({r.prefix_id for r in reqs if r.prefix_id.startswith("shared-")}) == 4


def test_independent_workload_has_unique_prefixes() -> None:
    reqs = generate(spec(shared_prefix_count=0, shared_fraction=0.0), seed=5, count=100)
    assert len({r.prefix_id for r in reqs}) == 100
    assert len({r.messages[0].content for r in reqs}) == 100


def test_same_prefix_id_gives_identical_prefix_text() -> None:
    reqs = generate(spec(shared_prefix_count=1, shared_fraction=1.0), seed=6, count=20)
    assert len({r.messages[0].content for r in reqs}) == 1
    assert len({r.messages[1].content for r in reqs}) == 20


def test_prefix_and_question_sizes_match_spec() -> None:
    reqs = generate(spec(prefix_tokens=64, question_tokens=8), seed=7, count=5)
    for r in reqs:
        assert len(r.messages[0].content) == 64 * CHARS_PER_TOKEN
        assert estimate_prompt_tokens(r.messages) == pytest.approx(64 + 8 + 4, abs=2)
        assert r.max_tokens == 4


def test_skew_concentrates_on_first_prefix() -> None:
    uniform = generate(spec(shared_skew=0.0, shared_fraction=1.0), seed=8, count=2000)
    skewed = generate(spec(shared_skew=2.0, shared_fraction=1.0), seed=8, count=2000)
    top_uniform = Counter(r.prefix_id for r in uniform).most_common(1)[0][1]
    top_skewed = Counter(r.prefix_id for r in skewed).most_common(1)[0][1]
    assert top_skewed > top_uniform * 1.5


def test_bursty_arrivals_group_same_prefix() -> None:
    arrival = ArrivalSpec(kind="bursty", burst_size=5, burst_spacing_ms=2.0, burst_gap_mean_ms=100)
    reqs = generate(spec(arrival=arrival, shared_fraction=1.0), seed=9, count=25)
    bursts = [reqs[i : i + 5] for i in range(0, 25, 5)]
    for burst in bursts:
        assert len({r.prefix_id for r in burst}) == 1
        gaps = [y.arrival_ms - x.arrival_ms for x, y in zip(burst, burst[1:], strict=False)]
        assert all(g == pytest.approx(2.0) for g in gaps)


@pytest.mark.parametrize("path", sorted(WORKLOAD_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_shipped_workloads_load_and_generate(path: Path) -> None:
    loaded = load_workload(str(path))
    assert loaded.name == path.stem
    assert len(generate(loaded, seed=42, count=20)) == 20


def test_unknown_workload_lists_available() -> None:
    with pytest.raises(FileNotFoundError, match="shared_prefix_heavy"):
        load_workload("nope")
