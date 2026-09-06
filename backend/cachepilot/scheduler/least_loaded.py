from __future__ import annotations

from cachepilot.core.models import InferenceRequest, RoutingDecision, WorkerState
from cachepilot.scheduler.base import KVDirectory
from cachepilot.scheduler.scoring import (
    TTFTEstimator,
    describe_candidates,
    healthy_workers,
    load,
)


class LeastLoadedScheduler:
    """Baseline: pick the healthy worker with the fewest queued + active requests."""

    policy = "least_loaded"

    def __init__(self, estimator: TTFTEstimator | None = None) -> None:
        self._estimator = estimator or TTFTEstimator()

    def parameters(self) -> dict[str, float]:
        return {}

    async def choose_worker(
        self,
        request: InferenceRequest,
        workers: list[WorkerState],
        kv_directory: KVDirectory,
    ) -> RoutingDecision:
        healthy = healthy_workers(workers)
        ranked = sorted(healthy, key=lambda w: (load(w), w.worker_id))
        selected = ranked[0]

        candidates = [
            c.model_copy(update={"final_score": float(-load(w))})
            for c, w in zip(
                describe_candidates(request, healthy, kv_directory, self._estimator),
                healthy,
                strict=True,
            )
        ]
        reason = (
            f"lowest load {load(selected)} "
            f"(queue {selected.queue_depth} + active {selected.active_requests})"
        )
        if len(ranked) > 1:
            runner_up = ranked[1]
            reason += f"; runner-up {runner_up.worker_id} at load {load(runner_up)}"
        return RoutingDecision(
            request_id=request.request_id,
            policy=self.policy,
            selected_worker_id=selected.worker_id,
            candidates=candidates,
            reason=reason,
        )
