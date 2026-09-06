import json
from collections.abc import Iterator

import httpx
import pytest

from cachepilot.core.models import InferenceRequest, Message
from cachepilot.workers.base import DoneEvent, TokenEvent
from cachepilot.workers.vllm import ObservedStats, VLLMWorker, parse_vllm_metrics

METRICS_TEXT = """
# HELP vllm:num_requests_running Number of requests currently running on GPU.
vllm:num_requests_running{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 2.0
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 3.0
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 0.42
vllm:prompt_tokens_total{model_name="Qwen/Qwen2.5-0.5B-Instruct"} 12345.0
"""


def sse(chunks: list[dict[str, object]]) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def fake_server(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/chat/completions":
            body = sse(
                [
                    {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                    {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
                    {"choices": [{"index": 0, "delta": {"content": " world"}}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
                ]
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/metrics":
            return httpx.Response(200, text=METRICS_TEXT)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def ticking(*values: float) -> Iterator[float]:
    yield from values
    while True:
        yield values[-1]


def make_worker(
    seen: list[httpx.Request], clock_values: tuple[float, ...] = (0.0, 0.05, 0.3)
) -> VLLMWorker:
    clock = ticking(*clock_values)
    client = httpx.AsyncClient(transport=fake_server(seen), base_url="http://vllm.test")
    return VLLMWorker(
        "vllm-0",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "http://vllm.test",
        client=client,
        monotonic=lambda: next(clock),
    )


def request() -> InferenceRequest:
    return InferenceRequest(
        model="sim-model",
        messages=[Message(role="system", content="S"), Message(role="user", content="U")],
        prompt_tokens_estimate=9,
        max_tokens=8,
    )


async def test_generate_streams_deltas_and_measures_observed_timings() -> None:
    seen: list[httpx.Request] = []
    worker = make_worker(seen)

    events = [e async for e in worker.generate(request(), cache_overlap=0.5)]

    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert [t.text for t in tokens] == ["Hello", " world"]
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.ttft_ms == pytest.approx(50.0)
    assert done.total_latency_ms == pytest.approx(300.0)
    assert done.output_tokens == 2
    assert done.finish_reason == "stop"
    assert done.queue_wait_ms == 0.0

    sent = json.loads(seen[0].content)
    assert sent["model"] == "Qwen/Qwen2.5-0.5B-Instruct"  # the served name, not the client's
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]  # full prompt always sent
    assert worker.observed.tokens_per_second == pytest.approx(1 / 0.25)


async def test_active_requests_tracked_and_released_on_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    seen: list[tuple[int, int]] = []
    worker = VLLMWorker(
        "vllm-0",
        "m",
        "http://vllm.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://vllm.test"
        ),
    )

    async def listener(w: object) -> None:
        seen.append((w.active_requests, w.queue_depth))

    worker.on_stats_changed = listener
    with pytest.raises(httpx.HTTPStatusError):
        async for _ in worker.generate(request()):
            pass
    assert seen == [(1, 0), (0, 0)]
    assert worker.active_requests == 0


async def test_health_and_stats_use_backend_metrics() -> None:
    seen: list[httpx.Request] = []
    worker = make_worker(seen)

    assert await worker.health() is True
    stats = await worker.stats()
    assert stats["queue_depth"] == 3
    assert stats["active_requests"] == 2
    assert stats["kv_usage_fraction"] == pytest.approx(0.42)
    assert worker.heartbeat_snapshot()["queue_depth"] == 3


async def test_health_false_when_server_unreachable() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    worker = VLLMWorker(
        "vllm-0",
        "m",
        "http://down.test",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://down.test"
        ),
    )
    assert await worker.health() is False
    assert await worker.stats() == worker.heartbeat_snapshot()


def test_parse_vllm_metrics_sums_label_sets_and_keeps_previous() -> None:
    text = (
        'vllm:num_requests_running{model_name="a"} 1.0\n'
        'vllm:num_requests_running{model_name="b"} 2.0\n'
        "unrelated_metric 5\n"
    )
    observed = parse_vllm_metrics(text, ObservedStats(waiting=7, kv_usage_fraction=0.1))
    assert observed.running == 3
    assert observed.waiting == 7
    assert observed.kv_usage_fraction == pytest.approx(0.1)
