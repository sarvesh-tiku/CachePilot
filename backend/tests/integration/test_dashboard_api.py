import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cachepilot.core.clock import VirtualClock
from cachepilot.core.config import Settings
from cachepilot.main import create_app

BODY: dict[str, Any] = {
    "model": "sim-model",
    "messages": [
        {"role": "system", "content": "Policy text. " * 200},
        {"role": "user", "content": "Q"},
    ],
    "max_tokens": 4,
}


async def test_requests_list_and_detail_join_decision_and_result(client: AsyncClient) -> None:
    for policy in ("round_robin", "kv_aware"):
        response = await client.post(
            "/v1/chat/completions", json=BODY, headers={"X-CachePilot-Policy": policy}
        )
        assert response.status_code == 200
    request_id = response.headers["X-CachePilot-Request-Id"]

    rows = (await client.get("/api/requests")).json()
    assert len(rows) == 2
    assert rows[0]["request"]["request_id"] == request_id  # newest first
    assert rows[0]["status"] == "completed"
    assert rows[0]["decision"]["policy"] == "kv_aware"
    assert rows[0]["result"]["output_tokens"] == 4
    assert "messages" not in rows[0]["request"]
    assert rows[0]["request"]["message_count"] == 2

    only_rr = (await client.get("/api/requests", params={"policy": "round_robin"})).json()
    assert [r["decision"]["policy"] for r in only_rr] == ["round_robin"]

    detail = await client.get(f"/api/requests/{request_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["request"]["prefix_fingerprint"]
    assert len(payload["decision"]["candidates"]) == 3
    assert payload["result"]["worker_id"] == payload["decision"]["selected_worker_id"]

    assert (await client.get(f"/api/requests/{uuid.uuid4()}")).status_code == 404


async def test_metrics_summary_reflects_recent_traffic(client: AsyncClient) -> None:
    empty = (await client.get("/api/metrics/summary")).json()
    assert empty["completed_requests"] == 0
    assert empty["healthy_workers"] == 3
    assert len(empty["workers"]) == 3

    for _ in range(5):
        assert (await client.post("/v1/chat/completions", json=BODY)).status_code == 200

    summary = (await client.get("/api/metrics/summary", params={"window_s": 60})).json()
    assert summary["completed_requests"] == 5
    assert summary["requests_per_s"] == pytest.approx(5 / 60)
    assert summary["output_tokens_per_s"] == pytest.approx(20 / 60)
    assert summary["ttft_ms"]["p50"] > 0
    assert summary["total_latency_ms"]["p99"] >= summary["ttft_ms"]["p99"]
    assert summary["active_requests"] == 0
    assert sum(w["completed_in_window"] for w in summary["workers"]) == 5
    assert any(w["cache_entries"] > 0 for w in summary["workers"])
    assert 0 <= summary["cache_hit_rate"] <= 1


async def test_benchmark_runs_api_lists_triggers_and_fetches(tmp_path: Path) -> None:
    settings = Settings(
        database_url="sqlite+aiosqlite://",
        worker_heartbeat_interval_s=0.01,
        benchmark_results_dir=str(tmp_path),
    )
    app: FastAPI = create_app(settings, clock=VirtualClock())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            workloads = (await client.get("/api/benchmark-runs/workloads")).json()
            assert {w["name"] for w in workloads} >= {"independent", "shared_prefix_heavy"}

            assert (await client.get("/api/benchmark-runs")).json() == []

            created = await client.post(
                "/api/benchmark-runs",
                json={
                    "workload": "shared_prefix_small",
                    "policies": ["round_robin", "kv_aware"],
                    "requests": 20,
                    "seed": 3,
                },
            )
            assert created.status_code == 201, created.text
            runs = created.json()
            assert [r["policy"] for r in runs] == ["round_robin", "kv_aware"]
            assert all(r["metrics"]["completed"] == 20 for r in runs)
            assert len(list(tmp_path.glob("*.json"))) == 2

            listed = (await client.get("/api/benchmark-runs")).json()
            assert len(listed) == 2

            one = await client.get(f"/api/benchmark-runs/{runs[0]['run_id']}")
            assert one.status_code == 200
            assert one.json()["workload"]["name"] == "shared_prefix_small"

            assert (await client.get(f"/api/benchmark-runs/{uuid.uuid4()}")).status_code == 404
            bad = await client.post("/api/benchmark-runs", json={"workload": "nope", "requests": 5})
            assert bad.status_code == 404
            bad_policy = await client.post(
                "/api/benchmark-runs", json={"workload": "independent", "policies": ["x"]}
            )
            assert bad_policy.status_code == 400
