from typing import Any

from httpx import AsyncClient

# ~3000 chars = ~750 tokens = 5 full chunks at the test chunk size of 128 tokens.
SHARED_DOCUMENT = "Company travel policy, section %d: employees must file receipts. " * 45
HEADERS = {"X-CachePilot-Policy": "least_loaded"}  # idle workers tie-break to worker-0


def body(question: str) -> dict[str, Any]:
    return {
        "model": "sim-model",
        "messages": [
            {"role": "system", "content": SHARED_DOCUMENT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 4,
    }


async def test_second_request_with_shared_prefix_hits_cache_on_same_worker(
    client: AsyncClient,
) -> None:
    first = await client.post("/v1/chat/completions", json=body("Question A"), headers=HEADERS)
    second = await client.post("/v1/chat/completions", json=body("Question B"), headers=HEADERS)
    assert (
        first.headers["X-CachePilot-Worker"] == second.headers["X-CachePilot-Worker"] == "worker-0"
    )

    first_detail = (
        await client.get(f"/api/decisions/{first.headers['X-CachePilot-Decision-Id']}")
    ).json()
    second_detail = (
        await client.get(f"/api/decisions/{second.headers['X-CachePilot-Decision-Id']}")
    ).json()

    first_overlaps = {
        c["worker_id"]: c["cache_overlap"] for c in first_detail["decision"]["candidates"]
    }
    assert first_overlaps == {"worker-0": 0.0, "worker-1": 0.0, "worker-2": 0.0}
    assert first_detail["result"]["cache_hit"] is False

    second_overlaps = {
        c["worker_id"]: c["cache_overlap"] for c in second_detail["decision"]["candidates"]
    }
    assert 0.8 < second_overlaps["worker-0"] < 1.0
    assert second_overlaps["worker-1"] == second_overlaps["worker-2"] == 0.0
    assert second_detail["result"]["cache_hit"] is True
    assert second_detail["result"]["cache_overlap"] == second_overlaps["worker-0"]
    assert second_detail["result"]["ttft_ms"] < first_detail["result"]["ttft_ms"]

    w0_second = next(
        c for c in second_detail["decision"]["candidates"] if c["worker_id"] == "worker-0"
    )
    w0_first = next(
        c for c in first_detail["decision"]["candidates"] if c["worker_id"] == "worker-0"
    )
    assert w0_second["estimated_ttft_ms"] < w0_first["estimated_ttft_ms"]


async def test_kv_api_reports_residency_and_worker_usage(client: AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=body("Question A"), headers=HEADERS)
    await client.post("/v1/chat/completions", json=body("Question B"), headers=HEADERS)

    summaries = {s["worker_id"]: s for s in (await client.get("/api/kv")).json()}
    w0 = summaries["worker-0"]
    assert w0["entries"] == 5
    assert w0["used_bytes"] == 5 * 128 * 524_288
    assert (w0["hits"], w0["misses"]) == (1, 1)
    assert w0["evictions"] == 0
    assert summaries["worker-1"]["entries"] == 0

    entries = (await client.get("/api/kv/worker-0/entries")).json()
    assert len(entries) == 5
    assert all(e["worker_id"] == "worker-0" and e["token_count"] == 128 for e in entries)
    assert sum(e["hit_count"] for e in entries) == 5

    worker = (await client.get("/api/workers/worker-0")).json()
    assert worker["kv_used_bytes"] == w0["used_bytes"]

    assert (await client.get("/api/kv/nope")).status_code == 404


async def test_unrelated_prompts_do_not_share_cache(client: AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=body("Question A"), headers=HEADERS)
    other = {
        "model": "sim-model",
        "messages": [{"role": "user", "content": "Completely different text. " * 60}],
        "max_tokens": 4,
    }
    response = await client.post("/v1/chat/completions", json=other, headers=HEADERS)
    detail = (
        await client.get(f"/api/decisions/{response.headers['X-CachePilot-Decision-Id']}")
    ).json()
    assert all(c["cache_overlap"] == 0.0 for c in detail["decision"]["candidates"])
    assert detail["result"]["cache_hit"] is False
