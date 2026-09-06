from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from cachepilot.api.schemas import (
    AssistantMessage,
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatDelta,
    RoutingInfo,
    Usage,
)
from cachepilot.core.config import Settings
from cachepilot.core.errors import NoHealthyWorkersError, UnknownPolicyError
from cachepilot.core.models import InferenceRequest
from cachepilot.core.tokens import estimate_prompt_tokens
from cachepilot.gateway.service import InferenceGateway, Submission
from cachepilot.workers.base import DoneEvent, TokenEvent

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    x_cachepilot_policy: Annotated[str | None, Header()] = None,
) -> Response:
    settings: Settings = request.app.state.settings
    gateway: InferenceGateway = request.app.state.gateway

    if body.max_tokens > settings.max_tokens_limit:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens {body.max_tokens} exceeds limit {settings.max_tokens_limit}",
        )

    inference = InferenceRequest(
        model=body.model,
        messages=body.messages,
        prompt_tokens_estimate=estimate_prompt_tokens(body.messages),
        max_tokens=body.max_tokens,
        latency_slo_ms=body.latency_slo_ms,
    )

    try:
        submission = await gateway.submit(inference, policy=x_cachepilot_policy)
    except UnknownPolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoHealthyWorkersError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    headers = {
        "X-CachePilot-Request-Id": str(inference.request_id),
        "X-CachePilot-Decision-Id": str(submission.decision.decision_id),
        "X-CachePilot-Policy": submission.decision.policy,
        "X-CachePilot-Worker": submission.decision.selected_worker_id,
    }

    if body.stream:
        return StreamingResponse(_sse(submission), media_type="text/event-stream", headers=headers)

    completion = await _collect(submission)
    return JSONResponse(completion.model_dump(mode="json"), headers=headers)


def _routing_info(submission: Submission) -> RoutingInfo:
    return RoutingInfo(
        request_id=submission.request.request_id,
        decision_id=submission.decision.decision_id,
        policy=submission.decision.policy,
        worker_id=submission.decision.selected_worker_id,
    )


def _completion_id(submission: Submission) -> str:
    return f"chatcmpl-{submission.request.request_id.hex}"


def _created(submission: Submission) -> int:
    return int(submission.request.created_at.timestamp())


async def _collect(submission: Submission) -> ChatCompletion:
    parts: list[str] = []
    done: DoneEvent | None = None
    async for event in submission.events:
        if isinstance(event, TokenEvent):
            parts.append(event.text)
        else:
            done = event
    if done is None:
        raise RuntimeError("worker stream ended without a DoneEvent")

    return ChatCompletion(
        id=_completion_id(submission),
        created=_created(submission),
        model=submission.request.model,
        choices=[
            ChatChoice(
                message=AssistantMessage(content="".join(parts)),
                finish_reason=done.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=submission.request.prompt_tokens_estimate,
            completion_tokens=done.output_tokens,
            total_tokens=submission.request.prompt_tokens_estimate + done.output_tokens,
        ),
        cachepilot=_routing_info(submission),
    )


async def _sse(submission: Submission) -> AsyncIterator[str]:
    def chunk(choice: ChatChunkChoice, routing: RoutingInfo | None = None) -> str:
        payload = ChatCompletionChunk(
            id=_completion_id(submission),
            created=_created(submission),
            model=submission.request.model,
            choices=[choice],
            cachepilot=routing,
        )
        return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"

    yield chunk(ChatChunkChoice(delta=ChatDelta(role="assistant")), _routing_info(submission))
    async for event in submission.events:
        if isinstance(event, TokenEvent):
            yield chunk(ChatChunkChoice(delta=ChatDelta(content=event.text)))
        else:
            yield chunk(ChatChunkChoice(delta=ChatDelta(), finish_reason=event.finish_reason))
    yield "data: [DONE]\n\n"
