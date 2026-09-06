import json
import logging
import uuid

from cachepilot.core.models import EventType, RequestResult, WorkerState, WorkerStatus
from cachepilot.persistence.repositories import InMemoryRunStore
from cachepilot.telemetry.events import Telemetry
from cachepilot.telemetry.logging import JsonFormatter
from cachepilot.telemetry.prometheus import Metrics


async def test_emit_persists_event_with_payload() -> None:
    store = InMemoryRunStore()
    telemetry = Telemetry(store, Metrics())
    rid = uuid.uuid4()

    event = await telemetry.emit(
        EventType.ROUTING_DECISION, request_id=rid, worker_id="worker-1", policy="kv_aware"
    )

    assert event.payload == {"policy": "kv_aware"}
    listed = await store.list_events(request_id=rid)
    assert [e.event_id for e in listed] == [event.event_id]
    assert await store.list_events(event_type=EventType.CACHE_HIT) == []


async def test_emit_logs_json_with_fields(caplog: object) -> None:
    logger = logging.getLogger("test.telemetry")
    logger.setLevel(logging.DEBUG)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        telemetry = Telemetry(InMemoryRunStore(), Metrics(), logger)
        rid = uuid.uuid4()
        await telemetry.emit(EventType.REQUEST_COMPLETED, request_id=rid, ttft_ms=12.5)
        await telemetry.emit(EventType.CACHE_MISS, request_id=rid)
    finally:
        logger.removeHandler(handler)

    assert [r.levelno for r in records] == [logging.INFO, logging.DEBUG]
    line = json.loads(JsonFormatter().format(records[0]))
    assert line["event"] == "request_completed"
    assert line["request_id"] == str(rid)
    assert line["ttft_ms"] == 12.5
    assert line["level"] == "INFO"
    assert "ts" in line


def test_metrics_render_includes_spec_names() -> None:
    metrics = Metrics()
    metrics.observe_request(
        "kv_aware",
        RequestResult(
            request_id=uuid.uuid4(),
            worker_id="worker-0",
            queue_wait_ms=12.0,
            ttft_ms=40.0,
            total_latency_ms=200.0,
            output_tokens=16,
            cache_hit=True,
            cache_overlap=0.9,
        ),
    )
    metrics.cache_hits_total.labels("worker-0").inc()
    metrics.cache_evictions_total.labels("worker-0").inc(3)
    metrics.sync_workers(
        [
            WorkerState(
                worker_id="worker-0",
                model="m",
                status=WorkerStatus.DRAINING,
                queue_depth=2,
                kv_capacity_bytes=100,
                kv_used_bytes=25,
                tokens_per_second_estimate=125.0,
            )
        ]
    )

    text = metrics.render().decode()
    assert (
        'cachepilot_requests_total{cache_hit="true",policy="kv_aware",worker="worker-0"} 1.0'
        in text
    )
    assert "cachepilot_request_duration_ms_bucket" in text
    assert 'cachepilot_ttft_ms_sum{policy="kv_aware"} 40.0' in text
    assert 'cachepilot_queue_wait_ms_count{policy="kv_aware"} 1.0' in text
    assert 'cachepilot_cache_hits_total{worker="worker-0"} 1.0' in text
    assert 'cachepilot_cache_evictions_total{worker="worker-0"} 3.0' in text
    assert 'cachepilot_worker_queue_depth{worker="worker-0"} 2.0' in text
    assert 'cachepilot_worker_kv_utilization{worker="worker-0"} 0.25' in text
    assert 'cachepilot_worker_tokens_per_second{worker="worker-0"} 125.0' in text
    assert 'cachepilot_worker_healthy{worker="worker-0"} 0.0' in text


def test_two_metrics_instances_do_not_collide() -> None:
    a, b = Metrics(), Metrics()
    a.cache_hits_total.labels("w").inc()
    assert 'cachepilot_cache_hits_total{worker="w"} 1.0' in a.render().decode()
    assert 'cachepilot_cache_hits_total{worker="w"}' not in b.render().decode()
