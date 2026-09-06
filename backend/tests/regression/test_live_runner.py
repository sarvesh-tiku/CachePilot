from pathlib import Path

from httpx import ASGITransport, AsyncClient

from benchmark.live import run_live
from benchmark.report import write_outputs
from benchmark.workloads import ArrivalSpec, WorkloadSpec
from cachepilot.core.clock import VirtualClock
from cachepilot.core.config import Settings
from cachepilot.main import create_app

TINY = WorkloadSpec(
    name="tiny_live",
    prefix_tokens=512,
    question_tokens=8,
    output_tokens=4,
    shared_prefix_count=2,
    shared_fraction=0.8,
    arrival=ArrivalSpec(rate_rps=1000.0),
)


async def test_live_runner_drives_a_running_gateway(tmp_path: Path) -> None:
    settings = Settings(database_url="sqlite+aiosqlite://", worker_heartbeat_interval_s=0.01)
    app = create_app(settings, clock=VirtualClock())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            run, rows = await run_live(
                client=client, spec=TINY, policy="kv_aware", requests=20, seed=3, warmup=1
            )

    assert run.mode == "live"
    assert run.gateway_url == "http://gateway.test"
    assert run.worker_backends == {f"worker-{i}": "simulated" for i in range(3)}
    assert run.metrics.completed == 20
    assert [r.index for r in rows] == list(range(20))
    assert all(r.ttft_ms >= 0 and r.total_latency_ms >= r.ttft_ms for r in rows)
    assert run.metrics.cache_hit_rate > 0
    assert set(run.metrics.ttft_ms_by_cache) == {"hit", "miss"}
    assert run.policy_parameters["alpha"] == 2.0

    json_path, _ = write_outputs(run, rows, tmp_path)
    assert json_path.name == "live_tiny_live_kv_aware_seed3.json"
