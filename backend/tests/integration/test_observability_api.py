from typing import Any

from httpx import AsyncClient

BODY: dict[str, Any] = {
    "model": "sim-model",
    "messages": [
        {"role": "system", "content": "Policy text. " * 200},
        {"role": "user", "content": "Q"},
    ],
    "max_tokens": 4,
}
HEADERS = {"X-CachePilot-Policy": "kv_aware"}


async def test_request_emits_ordered_event_trail(client: AsyncClient) -> None:
    first = await client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
    second = await client.post("/v1/chat/completions", json=BODY, headers=HEADERS)

    cold = (
        await client.get(
            "/api/events",
            params={"request_id": first.headers["X-CachePilot-Request-Id"], "ascending": "true"},
        )
    ).json()
    assert [e["event_type"] for e in cold] == [
        "REQUEST_RECEIVED",
        "REQUEST_FINGERPRINTED",
        "ROUTING_DECISION",
        "CACHE_MISS",
        "FIRST_TOKEN",
        "REQUEST_COMPLETED",
        "CACHE_ADMITTED",
    ]
    decision = next(e for e in cold if e["event_type"] == "ROUTING_DECISION")
    assert decision["worker_id"] == first.headers["X-CachePilot-Worker"]
    assert decision["payload"]["policy"] == "kv_aware"
    assert decision["payload"]["cache_overlap"] == 0.0
    assert "reason" in decision["payload"]
    completed = next(e for e in cold if e["event_type"] == "REQUEST_COMPLETED")
    assert completed["payload"]["output_tokens"] == 4
    assert completed["payload"]["ttft_ms"] > 0

    warm = (
        await client.get(
            "/api/events",
            params={"request_id": second.headers["X-CachePilot-Request-Id"], "ascending": "true"},
        )
    ).json()
    types = [e["event_type"] for e in warm]
    assert "CACHE_HIT" in types and "CACHE_MISS" not in types
    assert "CACHE_ADMITTED" not in types  # every chunk was already resident
    hit = next(e for e in warm if e["event_type"] == "CACHE_HIT")
    assert hit["payload"]["matched_chunks"] > 0

    only_hits = (await client.get("/api/events", params={"event_type": "CACHE_HIT"})).json()
    assert len(only_hits) == 1
    newest_first = (await client.get("/api/events", params={"limit": 3})).json()
    assert len(newest_first) == 3
    assert newest_first[0]["timestamp"] >= newest_first[-1]["timestamp"]


async def test_prometheus_endpoint_exposes_request_and_worker_metrics(client: AsyncClient) -> None:
    for _ in range(3):
        assert (
            await client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
        ).status_code == 200

    response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert (
        'cachepilot_requests_total{cache_hit="false",policy="kv_aware",worker="worker-0"} 1.0'
        in text
    )
    assert (
        'cachepilot_requests_total{cache_hit="true",policy="kv_aware",worker="worker-0"} 2.0'
        in text
    )
    assert 'cachepilot_ttft_ms_count{policy="kv_aware"} 3.0' in text
    assert 'cachepilot_cache_hits_total{worker="worker-0"} 2.0' in text
    assert 'cachepilot_cache_misses_total{worker="worker-0"} 1.0' in text
    assert 'cachepilot_worker_healthy{worker="worker-1"} 1.0' in text
    assert 'cachepilot_worker_kv_utilization{worker="worker-0"}' in text


async def test_drain_excludes_worker_and_restore_readmits(client: AsyncClient) -> None:
    drained = await client.post("/api/workers/worker-0/drain")
    assert drained.status_code == 200
    assert drained.json()["status"] == "draining"

    picks = set()
    for _ in range(4):
        response = await client.post(
            "/v1/chat/completions", json=BODY, headers={"X-CachePilot-Policy": "round_robin"}
        )
        picks.add(response.headers["X-CachePilot-Worker"])
    assert picks == {"worker-1", "worker-2"}

    events = (await client.get("/api/events", params={"event_type": "WORKER_DRAINING"})).json()
    assert events[0]["worker_id"] == "worker-0"
    assert events[0]["payload"] == {"previous": "healthy", "status": "draining"}

    metrics = (await client.get("/metrics")).text
    assert 'cachepilot_worker_healthy{worker="worker-0"} 0.0' in metrics

    restored = await client.post("/api/workers/worker-0/restore")
    assert restored.json()["status"] == "healthy"
    recovered = (await client.get("/api/events", params={"event_type": "WORKER_RECOVERED"})).json()
    assert recovered[0]["worker_id"] == "worker-0"

    assert (await client.post("/api/workers/nope/drain")).status_code == 404
