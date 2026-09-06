import uuid
from datetime import UTC, datetime, timedelta

import pytest

from cachepilot.core.models import (
    CandidateScore,
    InferenceRequest,
    Message,
    RequestResult,
    RoutingDecision,
)
from cachepilot.persistence.db import create_schema, make_engine
from cachepilot.persistence.repositories import SqlRunStore


@pytest.fixture
async def store() -> SqlRunStore:
    engine = make_engine("sqlite+aiosqlite://")
    await create_schema(engine)
    return SqlRunStore(engine)


def make_request() -> InferenceRequest:
    return InferenceRequest(
        model="sim-model",
        messages=[Message(role="user", content="secret prompt text")],
        prompt_tokens_estimate=42,
        max_tokens=16,
    )


def make_decision(
    request: InferenceRequest, policy: str = "round_robin", worker: str = "worker-0"
) -> RoutingDecision:
    return RoutingDecision(
        request_id=request.request_id,
        policy=policy,
        selected_worker_id=worker,
        candidates=[
            CandidateScore(
                worker_id="worker-0",
                cache_overlap=0.0,
                queue_depth=1,
                estimated_ttft_ms=29.1,
                kv_pressure=0.25,
                final_score=1.0,
            ),
            CandidateScore(
                worker_id="worker-1",
                cache_overlap=0.0,
                queue_depth=0,
                estimated_ttft_ms=17.1,
                kv_pressure=0.0,
                final_score=0.0,
            ),
        ],
        reason="test",
    )


async def test_decision_round_trips_with_candidates(store: SqlRunStore) -> None:
    request = make_request()
    decision = make_decision(request)
    await store.save_request(request)
    await store.save_decision(decision)

    loaded = await store.get_decision(decision.decision_id)

    assert loaded == decision
    assert loaded is not None and loaded.created_at.tzinfo is not None


async def test_result_round_trips(store: SqlRunStore) -> None:
    request = make_request()
    await store.save_request(request)
    result = RequestResult(
        request_id=request.request_id,
        worker_id="worker-0",
        queue_wait_ms=12.0,
        ttft_ms=30.5,
        total_latency_ms=158.5,
        output_tokens=16,
        cache_hit=False,
        cache_overlap=0.0,
    )
    await store.save_result(result)
    assert await store.get_result(request.request_id) == result


async def test_missing_rows_return_none(store: SqlRunStore) -> None:
    assert await store.get_decision(uuid.uuid4()) is None
    assert await store.get_result(uuid.uuid4()) is None


async def test_list_decisions_filters_and_orders_newest_first(store: SqlRunStore) -> None:
    base = datetime.now(UTC) - timedelta(minutes=10)
    decisions = []
    for i, (policy, worker) in enumerate(
        [("round_robin", "worker-0"), ("least_loaded", "worker-1"), ("round_robin", "worker-1")]
    ):
        request = make_request()
        await store.save_request(request)
        decision = make_decision(request, policy, worker).model_copy(
            update={"created_at": base + timedelta(minutes=i)}
        )
        await store.save_decision(decision)
        decisions.append(decision)

    everything = await store.list_decisions()
    assert [d.decision_id for d in everything] == [d.decision_id for d in reversed(decisions)]

    rr = await store.list_decisions(policy="round_robin")
    assert {d.decision_id for d in rr} == {decisions[0].decision_id, decisions[2].decision_id}

    w1 = await store.list_decisions(worker_id="worker-1")
    assert {d.decision_id for d in w1} == {decisions[1].decision_id, decisions[2].decision_id}

    recent = await store.list_decisions(since=base + timedelta(minutes=1))
    assert {d.decision_id for d in recent} == {decisions[1].decision_id, decisions[2].decision_id}

    assert len(await store.list_decisions(limit=1)) == 1
