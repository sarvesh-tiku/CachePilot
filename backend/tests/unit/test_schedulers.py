from collections import Counter

import pytest

from cachepilot.core.errors import NoHealthyWorkersError, UnknownPolicyError
from cachepilot.core.models import InferenceRequest, Message, WorkerState, WorkerStatus
from cachepilot.scheduler.base import NullKVDirectory
from cachepilot.scheduler.least_loaded import LeastLoadedScheduler
from cachepilot.scheduler.policies import build_all_schedulers, build_scheduler
from cachepilot.scheduler.round_robin import RoundRobinScheduler
from cachepilot.scheduler.scoring import TTFTEstimator

KV = NullKVDirectory()


def make_request() -> InferenceRequest:
    return InferenceRequest(
        model="sim-model",
        messages=[Message(role="user", content="hi")],
        prompt_tokens_estimate=1000,
        max_tokens=16,
    )


def worker(
    worker_id: str,
    queue: int = 0,
    active: int = 0,
    status: WorkerStatus = WorkerStatus.HEALTHY,
) -> WorkerState:
    return WorkerState(
        worker_id=worker_id,
        model="sim-model",
        status=status,
        queue_depth=queue,
        active_requests=active,
        kv_capacity_bytes=1024,
        kv_used_bytes=256,
    )


async def test_round_robin_distributes_evenly() -> None:
    scheduler = RoundRobinScheduler()
    workers = [worker("worker-0"), worker("worker-1"), worker("worker-2")]

    picks = [
        (await scheduler.choose_worker(make_request(), workers, KV)).selected_worker_id
        for _ in range(9)
    ]

    assert Counter(picks) == {"worker-0": 3, "worker-1": 3, "worker-2": 3}
    assert picks[:3] == ["worker-0", "worker-1", "worker-2"]


async def test_round_robin_skips_unhealthy_workers() -> None:
    scheduler = RoundRobinScheduler()
    workers = [
        worker("worker-0"),
        worker("worker-1", status=WorkerStatus.UNHEALTHY),
        worker("worker-2", status=WorkerStatus.DRAINING),
    ]
    picks = {
        (await scheduler.choose_worker(make_request(), workers, KV)).selected_worker_id
        for _ in range(4)
    }
    assert picks == {"worker-0"}


async def test_round_robin_is_independent_of_input_order() -> None:
    scheduler = RoundRobinScheduler()
    shuffled = [worker("worker-2"), worker("worker-0"), worker("worker-1")]
    decision = await scheduler.choose_worker(make_request(), shuffled, KV)
    assert decision.selected_worker_id == "worker-0"


async def test_round_robin_records_all_healthy_candidates() -> None:
    workers = [worker("worker-0", queue=2), worker("worker-1", status=WorkerStatus.UNHEALTHY)]
    decision = await RoundRobinScheduler().choose_worker(make_request(), workers, KV)

    assert [c.worker_id for c in decision.candidates] == ["worker-0"]
    candidate = decision.candidates[0]
    assert candidate.queue_depth == 2
    assert candidate.cache_overlap == 0.0
    assert candidate.kv_pressure == pytest.approx(0.25)
    assert candidate.estimated_ttft_ms == pytest.approx(2 * 12.0 + 1000 * 0.05 + 15.0)
    assert decision.policy == "round_robin"
    assert "round robin" in decision.reason


async def test_least_loaded_picks_minimum_load() -> None:
    workers = [
        worker("worker-0", queue=3, active=1),
        worker("worker-1", queue=1, active=0),
        worker("worker-2", queue=2, active=0),
    ]
    decision = await LeastLoadedScheduler().choose_worker(make_request(), workers, KV)

    assert decision.selected_worker_id == "worker-1"
    scores = {c.worker_id: c.final_score for c in decision.candidates}
    assert scores == {"worker-0": -4.0, "worker-1": -1.0, "worker-2": -2.0}
    assert "lowest load 1" in decision.reason
    assert "runner-up worker-2" in decision.reason


async def test_least_loaded_breaks_ties_by_worker_id() -> None:
    workers = [
        worker("worker-2", queue=1),
        worker("worker-1", queue=1),
        worker("worker-0", queue=5),
    ]
    decision = await LeastLoadedScheduler().choose_worker(make_request(), workers, KV)
    assert decision.selected_worker_id == "worker-1"


async def test_least_loaded_skips_unhealthy_even_if_idle() -> None:
    workers = [worker("worker-0", status=WorkerStatus.UNHEALTHY), worker("worker-1", queue=9)]
    decision = await LeastLoadedScheduler().choose_worker(make_request(), workers, KV)
    assert decision.selected_worker_id == "worker-1"


@pytest.mark.parametrize("scheduler", [RoundRobinScheduler(), LeastLoadedScheduler()])
async def test_no_healthy_workers_raises(
    scheduler: RoundRobinScheduler | LeastLoadedScheduler,
) -> None:
    with pytest.raises(NoHealthyWorkersError):
        await scheduler.choose_worker(
            make_request(), [worker("worker-0", status=WorkerStatus.UNHEALTHY)], KV
        )


@pytest.mark.parametrize("scheduler", [RoundRobinScheduler(), LeastLoadedScheduler()])
async def test_identical_state_gives_identical_decision(
    scheduler: RoundRobinScheduler | LeastLoadedScheduler,
) -> None:
    request = make_request()
    workers = [worker("worker-0", queue=1), worker("worker-1", queue=2)]
    a = await type(scheduler)().choose_worker(request, workers, KV)
    b = await type(scheduler)().choose_worker(request, workers, KV)
    assert a.selected_worker_id == b.selected_worker_id
    assert a.candidates == b.candidates
    assert a.reason == b.reason


def test_ttft_estimator_formula() -> None:
    est = TTFTEstimator(queue_wait_ms_per_pending=10, prefill_ms_per_token=0.1, fixed_overhead_ms=5)
    w = worker("w", queue=2, active=1)
    assert est.predict(w, prompt_tokens=1000, cache_overlap=0.0) == pytest.approx(30 + 100 + 5)
    assert est.predict(w, prompt_tokens=1000, cache_overlap=0.75) == pytest.approx(30 + 25 + 5)


def test_policy_factory() -> None:
    assert build_scheduler("round_robin").policy == "round_robin"
    assert build_scheduler("least_loaded").policy == "least_loaded"
    assert set(build_all_schedulers()) == {"round_robin", "least_loaded", "kv_aware"}
    with pytest.raises(UnknownPolicyError):
        build_scheduler("magic")
