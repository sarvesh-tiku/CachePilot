import pytest

from cachepilot.core.errors import WorkerAlreadyRegisteredError, WorkerNotFoundError
from cachepilot.core.models import WorkerStatus
from cachepilot.workers.registry import WorkerRegistry


async def test_register_worker_defaults_to_healthy() -> None:
    registry = WorkerRegistry()
    state = await registry.register("worker-0", "sim-model", kv_capacity_bytes=1024)
    assert state.worker_id == "worker-0"
    assert state.status == WorkerStatus.HEALTHY
    assert state.queue_depth == 0
    assert state.kv_used_bytes == 0


async def test_register_duplicate_raises() -> None:
    registry = WorkerRegistry()
    await registry.register("worker-0", "sim-model", kv_capacity_bytes=1024)
    with pytest.raises(WorkerAlreadyRegisteredError):
        await registry.register("worker-0", "sim-model", kv_capacity_bytes=1024)


async def test_heartbeat_updates_queue_stats() -> None:
    registry = WorkerRegistry()
    registered = await registry.register("worker-0", "sim-model", kv_capacity_bytes=1024)
    before = registered.last_heartbeat
    state = await registry.heartbeat(
        "worker-0", queue_depth=3, active_requests=1, tokens_per_second_estimate=120.5
    )
    assert state.queue_depth == 3
    assert state.active_requests == 1
    assert state.tokens_per_second_estimate == 120.5
    assert state.last_heartbeat >= before


async def test_update_kv_usage_is_independent_of_heartbeat() -> None:
    registry = WorkerRegistry()
    await registry.register("worker-0", "sim-model", kv_capacity_bytes=1024)
    await registry.update_kv_usage("worker-0", 512)
    state = await registry.heartbeat(
        "worker-0", queue_depth=0, active_requests=0, tokens_per_second_estimate=1.0
    )
    assert state.kv_used_bytes == 512


async def test_heartbeat_unknown_worker_raises() -> None:
    registry = WorkerRegistry()
    with pytest.raises(WorkerNotFoundError):
        await registry.heartbeat(
            "missing", queue_depth=0, active_requests=0, tokens_per_second_estimate=0.0
        )


async def test_get_unknown_worker_raises() -> None:
    registry = WorkerRegistry()
    with pytest.raises(WorkerNotFoundError):
        await registry.get("missing")


async def test_mark_unhealthy_excludes_from_healthy_list() -> None:
    registry = WorkerRegistry()
    await registry.register("worker-0", "sim-model", kv_capacity_bytes=1024)
    await registry.register("worker-1", "sim-model", kv_capacity_bytes=1024)

    await registry.mark_unhealthy("worker-0")

    healthy = await registry.list_healthy()
    assert [w.worker_id for w in healthy] == ["worker-1"]
    assert len(await registry.list_all()) == 2


async def test_drain_and_restore_notify_listener_only_on_change() -> None:
    changes: list[tuple[str, str, str]] = []

    async def listener(state: object, previous: WorkerStatus) -> None:
        assert hasattr(state, "worker_id")
        changes.append((state.worker_id, previous.value, state.status.value))  # type: ignore[attr-defined]

    registry = WorkerRegistry(on_status_change=listener)
    await registry.register("worker-0", "sim-model", kv_capacity_bytes=1024)

    drained = await registry.drain("worker-0")
    assert drained.status == WorkerStatus.DRAINING
    assert await registry.list_healthy() == []

    # A heartbeat must not un-drain an operator-drained worker.
    await registry.heartbeat(
        "worker-0", queue_depth=0, active_requests=0, tokens_per_second_estimate=1.0
    )
    assert (await registry.get("worker-0")).status == WorkerStatus.DRAINING

    await registry.restore("worker-0")
    await registry.restore("worker-0")  # no-op, no second notification
    assert changes == [
        ("worker-0", "healthy", "draining"),
        ("worker-0", "draining", "healthy"),
    ]


async def test_heartbeat_recovers_unhealthy_worker() -> None:
    registry = WorkerRegistry()
    await registry.register("worker-0", "sim-model", kv_capacity_bytes=1024)
    await registry.mark_unhealthy("worker-0")

    state = await registry.heartbeat(
        "worker-0", queue_depth=0, active_requests=0, tokens_per_second_estimate=10.0
    )
    assert state.status == WorkerStatus.HEALTHY
