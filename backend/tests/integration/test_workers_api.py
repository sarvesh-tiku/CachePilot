import asyncio

from httpx import AsyncClient


async def test_three_workers_run_concurrently_and_report_state(client: AsyncClient) -> None:
    await asyncio.sleep(0.05)

    response = await client.get("/api/workers")
    assert response.status_code == 200
    workers = response.json()
    assert [w["worker_id"] for w in workers] == ["worker-0", "worker-1", "worker-2"]
    for worker in workers:
        assert worker["status"] == "healthy"
        assert worker["kv_capacity_bytes"] > 0
        assert 0 <= worker["kv_used_bytes"] <= worker["kv_capacity_bytes"]
        assert worker["tokens_per_second_estimate"] > 0

    single = await client.get("/api/workers/worker-1")
    assert single.status_code == 200
    assert single.json()["worker_id"] == "worker-1"

    missing = await client.get("/api/workers/nope")
    assert missing.status_code == 404


async def test_health_endpoint(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
