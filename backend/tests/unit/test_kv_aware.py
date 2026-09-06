import pytest

from cachepilot.core.errors import NoHealthyWorkersError
from cachepilot.core.models import InferenceRequest, Message, WorkerState, WorkerStatus
from cachepilot.scheduler.base import KVDirectory
from cachepilot.scheduler.kv_aware import KVAwareScheduler, KVAwareWeights
from cachepilot.scheduler.policies import build_all_schedulers, build_scheduler

PROMPT_TOKENS = 4000


class FixedOverlap:
    def __init__(self, overlaps: dict[str, float]) -> None:
        self._overlaps = overlaps

    def cache_overlap(self, request: InferenceRequest, worker_id: str) -> float:
        return self._overlaps.get(worker_id, 0.0)


def make_request() -> InferenceRequest:
    return InferenceRequest(
        model="sim-model",
        messages=[Message(role="user", content="hi")],
        prompt_tokens_estimate=PROMPT_TOKENS,
        max_tokens=16,
    )


def worker(
    worker_id: str,
    queue: int = 0,
    active: int = 0,
    kv_used: int = 0,
    status: WorkerStatus = WorkerStatus.HEALTHY,
) -> WorkerState:
    return WorkerState(
        worker_id=worker_id,
        model="sim-model",
        status=status,
        queue_depth=queue,
        active_requests=active,
        kv_capacity_bytes=1000,
        kv_used_bytes=kv_used,
    )


async def choose(
    workers: list[WorkerState], kv: KVDirectory, weights: KVAwareWeights | None = None
) -> tuple[str, str, dict[str, float]]:
    decision = await KVAwareScheduler(weights).choose_worker(make_request(), workers, kv)
    return (
        decision.selected_worker_id,
        decision.reason,
        {c.worker_id: c.final_score for c in decision.candidates},
    )


async def test_prefers_cached_worker_when_load_is_equal() -> None:
    workers = [worker("worker-0", queue=2), worker("worker-1", queue=2)]
    selected, reason, _ = await choose(workers, FixedOverlap({"worker-1": 0.9}))
    assert selected == "worker-1"
    assert "cache reuse and lowest load both favor worker-1" in reason


async def test_without_any_cached_prefix_it_behaves_like_least_loaded() -> None:
    workers = [
        worker("worker-0", queue=3),
        worker("worker-1", queue=1),
        worker("worker-2", queue=2),
    ]
    selected, reason, _ = await choose(workers, FixedOverlap({}))
    assert selected == "worker-1"
    assert reason.startswith("no cached prefix on any worker")


async def test_cache_reuse_outweighs_moderate_queue_depth() -> None:
    workers = [worker("worker-0", queue=4), worker("worker-1", queue=1)]
    selected, reason, scores = await choose(workers, FixedOverlap({"worker-0": 0.91}))
    assert selected == "worker-0"
    assert "cache reuse on worker-0" in reason and "outweighed lower load on worker-1" in reason
    assert scores["worker-0"] > scores["worker-1"]


async def test_severe_queue_depth_overrides_cache_reuse() -> None:
    workers = [worker("worker-0", queue=60), worker("worker-1", queue=0)]
    selected, reason, _ = await choose(workers, FixedOverlap({"worker-0": 0.91}))
    assert selected == "worker-1"
    assert reason.startswith("load on worker-0")
    assert "chose worker-1" in reason


async def test_weights_change_the_tradeoff() -> None:
    workers = [worker("worker-0", queue=4), worker("worker-1", queue=1)]
    kv = FixedOverlap({"worker-0": 0.5})

    cache_hungry, _, _ = await choose(workers, kv, KVAwareWeights(alpha=10, beta=0.1, delta=0))
    load_averse, _, _ = await choose(workers, kv, KVAwareWeights(alpha=0.1, beta=10, delta=0))
    assert cache_hungry == "worker-0"
    assert load_averse == "worker-1"


async def test_kv_pressure_breaks_ties_against_full_workers() -> None:
    workers = [worker("worker-0", kv_used=950), worker("worker-1", kv_used=100)]
    selected, _, scores = await choose(workers, FixedOverlap({}))
    assert selected == "worker-1"
    assert scores["worker-1"] > scores["worker-0"]


async def test_all_candidates_scored_and_ordered_by_input_id() -> None:
    workers = [worker("worker-2"), worker("worker-0", queue=1), worker("worker-1")]
    decision = await KVAwareScheduler().choose_worker(
        make_request(), workers, FixedOverlap({"worker-2": 0.3})
    )
    assert [c.worker_id for c in decision.candidates] == ["worker-0", "worker-1", "worker-2"]
    assert all(isinstance(c.final_score, float) for c in decision.candidates)
    assert decision.selected_worker_id == "worker-2"


async def test_score_formula_matches_documented_terms() -> None:
    weights = KVAwareWeights(alpha=2.0, beta=0.8, gamma=0.4, delta=0.002)
    workers = [worker("worker-0", queue=2, kv_used=500), worker("worker-1", queue=4)]
    decision = await KVAwareScheduler(weights).choose_worker(
        make_request(), workers, FixedOverlap({"worker-0": 0.75})
    )
    c0 = next(c for c in decision.candidates if c.worker_id == "worker-0")
    expected = 2.0 * 0.75 - 0.8 * (2 / 4) - 0.4 * 0.5 - 0.002 * c0.estimated_ttft_ms
    assert c0.final_score == pytest.approx(expected)
    assert "alpha=2.0" in decision.reason


async def test_skips_unhealthy_even_with_perfect_cache() -> None:
    workers = [worker("worker-0", status=WorkerStatus.DRAINING), worker("worker-1", queue=9)]
    selected, _, _ = await choose(workers, FixedOverlap({"worker-0": 1.0}))
    assert selected == "worker-1"


async def test_no_healthy_workers_raises() -> None:
    with pytest.raises(NoHealthyWorkersError):
        await choose([worker("worker-0", status=WorkerStatus.UNHEALTHY)], FixedOverlap({}))


async def test_identical_state_gives_identical_decision() -> None:
    workers = [worker("worker-0", queue=1), worker("worker-1", queue=2)]
    kv = FixedOverlap({"worker-1": 0.4})
    request = make_request()
    a = await KVAwareScheduler().choose_worker(request, workers, kv)
    b = await KVAwareScheduler().choose_worker(request, workers, kv)
    assert (a.selected_worker_id, a.candidates, a.reason) == (
        b.selected_worker_id,
        b.candidates,
        b.reason,
    )


def test_factory_exposes_weights_as_parameters() -> None:
    scheduler = build_scheduler("kv_aware", weights=KVAwareWeights(alpha=3.0))
    assert scheduler.parameters() == {"alpha": 3.0, "beta": 0.8, "gamma": 0.4, "delta": 0.002}
    assert set(build_all_schedulers()) == {"round_robin", "least_loaded", "kv_aware"}
    assert build_scheduler("round_robin").parameters() == {}
