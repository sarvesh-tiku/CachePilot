import json
from collections import Counter
from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient

from cachepilot.workers.registry import WorkerRegistry

BODY: dict[str, Any] = {
    "model": "sim-model",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Explain KV caching."},
    ],
    "max_tokens": 8,
}


async def test_non_streaming_completion_has_openai_shape(client: AsyncClient) -> None:
    response = await client.post("/v1/chat/completions", json=BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == "sim-model"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"]
    assert choice["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 8
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 8

    routing = body["cachepilot"]
    assert routing["policy"] == "round_robin"
    assert routing["worker_id"] == "worker-0"
    assert response.headers["X-CachePilot-Worker"] == "worker-0"
    assert response.headers["X-CachePilot-Request-Id"] == routing["request_id"]
    assert response.headers["X-CachePilot-Decision-Id"] == routing["decision_id"]


async def test_streaming_completion_emits_sse_chunks(client: AsyncClient) -> None:
    async with client.stream(
        "POST", "/v1/chat/completions", json={**BODY, "stream": True}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = (await response.aread()).decode()

    frames = [line[len("data: ") :] for line in raw.split("\n\n") if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]

    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[0]["cachepilot"]["worker_id"] == "worker-0"

    content_chunks = [c for c in chunks[1:-1] if "content" in c["choices"][0]["delta"]]
    assert len(content_chunks) == 8
    assert "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)

    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


async def test_round_robin_spreads_nine_requests_over_three_workers(
    client: AsyncClient,
) -> None:
    picks = []
    for _ in range(9):
        response = await client.post("/v1/chat/completions", json=BODY)
        assert response.status_code == 200
        picks.append(response.headers["X-CachePilot-Worker"])

    assert Counter(picks) == {"worker-0": 3, "worker-1": 3, "worker-2": 3}


async def test_policy_header_selects_scheduler(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions", json=BODY, headers={"X-CachePilot-Policy": "least_loaded"}
    )
    assert response.status_code == 200
    assert response.json()["cachepilot"]["policy"] == "least_loaded"
    assert response.headers["X-CachePilot-Policy"] == "least_loaded"


async def test_unknown_policy_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions", json=BODY, headers={"X-CachePilot-Policy": "magic"}
    )
    assert response.status_code == 400
    assert "magic" in response.json()["detail"]


async def test_max_tokens_over_limit_is_rejected(client: AsyncClient) -> None:
    response = await client.post("/v1/chat/completions", json={**BODY, "max_tokens": 65})
    assert response.status_code == 400


async def test_no_healthy_workers_returns_503(app: FastAPI, client: AsyncClient) -> None:
    registry: WorkerRegistry = app.state.registry
    for worker in await registry.list_all():
        await registry.mark_unhealthy(worker.worker_id)

    response = await client.post("/v1/chat/completions", json=BODY)
    assert response.status_code == 503


async def test_decision_and_result_are_persisted_and_queryable(client: AsyncClient) -> None:
    for policy in ("round_robin", "round_robin", "least_loaded"):
        response = await client.post(
            "/v1/chat/completions", json=BODY, headers={"X-CachePilot-Policy": policy}
        )
        assert response.status_code == 200
    decision_id = response.headers["X-CachePilot-Decision-Id"]

    everything = (await client.get("/api/decisions")).json()
    assert len(everything) == 3

    rr = (await client.get("/api/decisions", params={"policy": "round_robin"})).json()
    assert len(rr) == 2
    assert {d["selected_worker_id"] for d in rr} == {"worker-0", "worker-1"}

    on_w0 = (await client.get("/api/decisions", params={"worker_id": "worker-0"})).json()
    assert all(d["selected_worker_id"] == "worker-0" for d in on_w0)

    detail = await client.get(f"/api/decisions/{decision_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["decision"]["policy"] == "least_loaded"
    assert len(payload["decision"]["candidates"]) == 3
    assert payload["decision"]["reason"]
    result = payload["result"]
    assert result["output_tokens"] == 8
    assert result["worker_id"] == payload["decision"]["selected_worker_id"]
    assert result["ttft_ms"] > 0
    assert result["total_latency_ms"] > result["ttft_ms"]
    assert result["cache_hit"] is False

    missing = await client.get("/api/decisions/00000000-0000-0000-0000-000000000000")
    assert missing.status_code == 404


async def test_worker_counters_return_to_idle_after_requests(client: AsyncClient) -> None:
    for _ in range(3):
        await client.post("/v1/chat/completions", json=BODY)

    workers = (await client.get("/api/workers")).json()
    assert all(w["queue_depth"] == 0 and w["active_requests"] == 0 for w in workers)
