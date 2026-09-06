from __future__ import annotations

from cachepilot.core.errors import UnknownPolicyError
from cachepilot.scheduler.base import Scheduler
from cachepilot.scheduler.kv_aware import KVAwareScheduler, KVAwareWeights
from cachepilot.scheduler.least_loaded import LeastLoadedScheduler
from cachepilot.scheduler.round_robin import RoundRobinScheduler
from cachepilot.scheduler.scoring import TTFTEstimator

POLICY_NAMES = (
    RoundRobinScheduler.policy,
    LeastLoadedScheduler.policy,
    KVAwareScheduler.policy,
)


def build_scheduler(
    policy: str,
    *,
    weights: KVAwareWeights | None = None,
    estimator: TTFTEstimator | None = None,
) -> Scheduler:
    if policy == RoundRobinScheduler.policy:
        return RoundRobinScheduler(estimator)
    if policy == LeastLoadedScheduler.policy:
        return LeastLoadedScheduler(estimator)
    if policy == KVAwareScheduler.policy:
        return KVAwareScheduler(weights, estimator)
    raise UnknownPolicyError(policy, list(POLICY_NAMES))


def build_all_schedulers(
    *,
    weights: KVAwareWeights | None = None,
    estimator: TTFTEstimator | None = None,
) -> dict[str, Scheduler]:
    return {
        name: build_scheduler(name, weights=weights, estimator=estimator) for name in POLICY_NAMES
    }
