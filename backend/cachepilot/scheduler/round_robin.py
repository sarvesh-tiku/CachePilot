from __future__ import annotations

from cachepilot.core.models import InferenceRequest, RoutingDecision, WorkerState
from cachepilot.scheduler.base import KVDirectory
from cachepilot.scheduler.scoring import TTFTEstimator, describe_candidates, healthy_workers


class RoundRobinScheduler:
    """Baseline: rotate through healthy workers, ignoring load and cache state."""

    policy = "round_robin"

    def __init__(self, estimator: TTFTEstimator | None = None) -> None:
        self._estimator = estimator or TTFTEstimator()
        self._cursor = 0

    def parameters(self) -> dict[str, float]:
        return {}

    async def choose_worker(
        self,
        request: InferenceRequest,
        workers: list[WorkerState],
        kv_directory: KVDirectory,
    ) -> RoutingDecision:
        healthy = healthy_workers(workers)
        position = self._cursor % len(healthy)
        self._cursor += 1
        selected = healthy[position]

        candidates = [
            c.model_copy(update={"final_score": 1.0 if c.worker_id == selected.worker_id else 0.0})
            for c in describe_candidates(request, healthy, kv_directory, self._estimator)
        ]
        return RoutingDecision(
            request_id=request.request_id,
            policy=self.policy,
            selected_worker_id=selected.worker_id,
            candidates=candidates,
            reason=(
                f"round robin cursor at position {position} of {len(healthy)} healthy workers; "
                "load and cache state not considered"
            ),
        )
