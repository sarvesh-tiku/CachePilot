import json

import httpx

from cachepilot.core.clock import VirtualClock
from cachepilot.core.models import EventType, InferenceRequest, Message
from cachepilot.gateway.factory import build_stack
from cachepilot.persistence.repositories import InMemoryRunStore
from cachepilot.scheduler.kv_aware import KVAwareWeights
from cachepilot.workers.base import DoneEvent
from cachepilot.workers.vllm import VLLMWorker

PROMPT = "Company policy. " * 300  # ~1200 tokens: 9 full 128-token chunks


def openai_like_server(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        n = json.loads(request.content)["max_tokens"]
        chunks = [{"choices": [{"index": 0, "delta": {"content": f"t{i} "}}]} for i in range(n)]
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    return httpx.MockTransport(handler)


async def test_request_flows_through_gateway_to_a_real_worker() -> None:
    seen: list[httpx.Request] = []
    worker = VLLMWorker(
        "vllm-0",
        "served-model",
        "http://llm.test",
        client=httpx.AsyncClient(transport=openai_like_server(seen), base_url="http://llm.test"),
    )
    store = InMemoryRunStore()
    stack = await build_stack(
        simulated_workers=0,
        clock=VirtualClock(),
        store=store,
        default_policy="kv_aware",
        weights=KVAwareWeights(),
        chunk_tokens=128,
        bytes_per_token=1024,
        real_workers=[(worker, 64 * 1024 * 1024)],
    )

    def make_request(question: str) -> InferenceRequest:
        messages = [Message(role="system", content=PROMPT), Message(role="user", content=question)]
        return InferenceRequest(
            model="whatever-the-client-said",
            messages=messages,
            prompt_tokens_estimate=1200,
            max_tokens=3,
        )

    first = await stack.gateway.submit(make_request("A"))
    tokens = [e async for e in first.events]
    assert isinstance(tokens[-1], DoneEvent)
    assert "".join(t.text for t in tokens[:-1]) == "t0 t1 t2 "
    assert first.decision.selected_worker_id == "vllm-0"
    assert first.cache_overlap == 0.0

    second = await stack.gateway.submit(make_request("B"))
    async for _ in second.events:
        pass
    assert second.cache_overlap > 0.8  # gateway *estimates* the server still holds the prefix

    result = await store.get_result(second.request.request_id)
    assert result is not None
    assert result.worker_id == "vllm-0"
    assert result.cache_hit is True
    assert result.output_tokens == 3
    assert result.queue_wait_ms == 0.0  # not observable through the OpenAI API

    states = await stack.registry.list_all()
    assert [(s.worker_id, s.backend) for s in states] == [("vllm-0", "vllm")]
    summary = stack.kv_directory.summary("vllm-0")
    assert summary.source == "estimated"
    assert summary.entries == 9

    sent = [json.loads(r.content) for r in seen]
    assert all(s["model"] == "served-model" for s in sent)
    assert all(len(s["messages"]) == 2 for s in sent)  # full prompt regardless of overlap

    hits = await store.list_events(event_type=EventType.CACHE_HIT)
    assert len(hits) == 1 and hits[0].worker_id == "vllm-0"
