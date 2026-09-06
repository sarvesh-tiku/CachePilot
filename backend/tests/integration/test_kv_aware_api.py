from collections import Counter
from typing import Any

from httpx import AsyncClient

SHARED_DOCUMENT = "Company travel policy, section %d: employees must file receipts. " * 45


def body(question: str) -> dict[str, Any]:
    return {
        "model": "sim-model",
        "messages": [
            {"role": "system", "content": SHARED_DOCUMENT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 4,
    }


async def run(client: AsyncClient, policy: str, n: int) -> list[str]:
    picks = []
    for i in range(n):
        response = await client.post(
            "/v1/chat/completions", json=body(f"Q{i}"), headers={"X-CachePilot-Policy": policy}
        )
        assert response.status_code == 200
        picks.append(response.headers["X-CachePilot-Worker"])
    return picks


async def test_kv_aware_clusters_shared_prefix_where_round_robin_scatters(
    client: AsyncClient,
) -> None:
    scattered = await run(client, "round_robin", 6)
    assert Counter(scattered) == {"worker-0": 2, "worker-1": 2, "worker-2": 2}

    clustered = await run(client, "kv_aware", 6)
    assert len(set(clustered)) == 1

    decisions = (await client.get("/api/decisions", params={"policy": "kv_aware"})).json()
    reasons = [d["reason"] for d in decisions]
    assert all("cache reuse" in r for r in reasons)
    overlaps = [
        next(c["cache_overlap"] for c in d["candidates"] if c["worker_id"] == clustered[0])
        for d in decisions
    ]
    assert all(o > 0.8 for o in overlaps)


async def test_policies_endpoint_exposes_weights(client: AsyncClient) -> None:
    policies = {p["name"]: p for p in (await client.get("/api/policies")).json()}
    assert set(policies) == {"round_robin", "least_loaded", "kv_aware"}
    assert policies["round_robin"]["default"] is True
    assert policies["kv_aware"]["parameters"] == {
        "alpha": 2.0,
        "beta": 0.8,
        "gamma": 0.4,
        "delta": 0.002,
    }
