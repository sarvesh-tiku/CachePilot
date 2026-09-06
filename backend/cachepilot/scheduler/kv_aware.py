from __future__ import annotations

from dataclasses import asdict, dataclass

from cachepilot.core.models import CandidateScore, InferenceRequest, RoutingDecision, WorkerState
from cachepilot.scheduler.base import KVDirectory
from cachepilot.scheduler.scoring import (
    TTFTEstimator,
    describe_candidates,
    healthy_workers,
    load,
)


@dataclass(frozen=True)
class KVAwareWeights:
    alpha: float = 2.0  # reward per unit of cache overlap
    beta: float = 0.8  # penalty per unit of load, normalized to the busiest candidate
    gamma: float = 0.4  # penalty per unit of KV pressure
    delta: float = 0.002  # penalty per ms of predicted TTFT


class KVAwareScheduler:
    """Trade cache locality against load.

    score = alpha * cache_overlap
          - beta  * load / max_load_among_candidates
          - gamma * kv_pressure
          - delta * predicted_ttft_ms

    Load is normalized against the busiest healthy candidate so the beta term
    only expresses *relative* load; absolute queueing cost enters through the
    predicted TTFT term.
    """

    policy = "kv_aware"

    def __init__(
        self,
        weights: KVAwareWeights | None = None,
        estimator: TTFTEstimator | None = None,
    ) -> None:
        self.weights = weights or KVAwareWeights()
        self._estimator = estimator or TTFTEstimator()

    def parameters(self) -> dict[str, float]:
        return asdict(self.weights)

    async def choose_worker(
        self,
        request: InferenceRequest,
        workers: list[WorkerState],
        kv_directory: KVDirectory,
    ) -> RoutingDecision:
        healthy = healthy_workers(workers)
        w = self.weights
        max_load = max(load(worker) for worker in healthy) or 1

        scored: list[CandidateScore] = []
        for candidate, worker in zip(
            describe_candidates(request, healthy, kv_directory, self._estimator),
            healthy,
            strict=True,
        ):
            score = (
                w.alpha * candidate.cache_overlap
                - w.beta * load(worker) / max_load
                - w.gamma * candidate.kv_pressure
                - w.delta * candidate.estimated_ttft_ms
            )
            scored.append(candidate.model_copy(update={"final_score": score}))

        loads = {worker.worker_id: load(worker) for worker in healthy}
        best = min(scored, key=lambda c: (-c.final_score, c.worker_id))
        return RoutingDecision(
            request_id=request.request_id,
            policy=self.policy,
            selected_worker_id=best.worker_id,
            candidates=scored,
            reason=self._explain(best, scored, loads),
        )

    def _explain(
        self, best: CandidateScore, scored: list[CandidateScore], loads: dict[str, int]
    ) -> str:
        least_loaded = min(scored, key=lambda c: (loads[c.worker_id], c.worker_id))
        most_cached = max(scored, key=lambda c: (c.cache_overlap, -loads[c.worker_id]))
        w = self.weights

        def describe(c: CandidateScore) -> str:
            return (
                f"{c.worker_id} (overlap {c.cache_overlap:.2f}, load {loads[c.worker_id]}, "
                f"est TTFT {c.estimated_ttft_ms:.0f}ms)"
            )

        if most_cached.cache_overlap == 0.0:
            headline = f"no cached prefix on any worker; lowest predicted TTFT at {describe(best)}"
        elif (
            best.worker_id == most_cached.worker_id
            and loads[best.worker_id] == loads[least_loaded.worker_id]
        ):
            headline = f"cache reuse and lowest load both favor {describe(best)}"
        elif best.worker_id == most_cached.worker_id:
            headline = (
                f"cache reuse on {describe(best)} outweighed lower load on {describe(least_loaded)}"
            )
        else:
            headline = (
                f"load on {describe(most_cached)} outweighed its cache reuse; "
                f"chose {describe(best)}"
            )
        return (
            f"{headline}; score {best.final_score:.3f} "
            f"(alpha={w.alpha}, beta={w.beta}, gamma={w.gamma}, delta={w.delta})"
        )
