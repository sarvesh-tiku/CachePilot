from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from cachepilot.gateway.service import InferenceGateway
from cachepilot.scheduler.scoring import TTFTEstimator

router = APIRouter(prefix="/api/policies", tags=["policies"])


class PolicyInfo(BaseModel):
    name: str
    default: bool
    parameters: dict[str, float]
    estimator: dict[str, float] = Field(
        default_factory=dict,
        description="TTFT estimator coefficients shared by every policy's candidate scores",
    )


@router.get("", response_model=list[PolicyInfo])
async def list_policies(request: Request) -> list[PolicyInfo]:
    gateway: InferenceGateway = request.app.state.gateway
    estimator: TTFTEstimator = request.app.state.estimator
    return [
        PolicyInfo(
            name=name,
            default=name == gateway.default_policy,
            parameters=params,
            estimator=estimator.parameters(),
        )
        for name, params in gateway.describe_policies().items()
    ]
