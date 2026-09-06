from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace

from cachepilot.core.errors import NoHealthyWorkersError
from cachepilot.core.models import CandidateScore, InferenceRequest, WorkerState, WorkerStatus
from cachepilot.scheduler.base import KVDirectory


@dataclass(frozen=True)
class TTFTSample:
    """One observed request: what the scheduler knew when it decided, and what happened."""

    pending: int
    prompt_tokens: int
    cache_overlap: float
    ttft_ms: float

    @property
    def uncached_tokens(self) -> float:
        return self.prompt_tokens * (1.0 - self.cache_overlap)


@dataclass(frozen=True)
class FitReport:
    samples: int
    mae_ms: float
    bias_ms: float  # mean(predicted - observed); positive means the estimator runs high
    r2: float


@dataclass(frozen=True)
class TTFTEstimator:
    """Predicted TTFT = queue wait + uncached prefill + fixed overhead.

    Coefficients default to the simulator's cost model. For real workers they
    should be fitted from observed timings with `fit`; until then estimates are
    exactly as good as the simulator is realistic, which is to say: an estimate.
    """

    queue_wait_ms_per_pending: float = 12.0
    prefill_ms_per_token: float = 0.05
    fixed_overhead_ms: float = 15.0

    def predict(self, worker: WorkerState, prompt_tokens: int, cache_overlap: float) -> float:
        pending = worker.queue_depth + worker.active_requests
        return self.predict_from(pending, prompt_tokens * (1.0 - cache_overlap))

    def predict_from(self, pending: float, uncached_tokens: float) -> float:
        return (
            pending * self.queue_wait_ms_per_pending
            + uncached_tokens * self.prefill_ms_per_token
            + self.fixed_overhead_ms
        )

    def parameters(self) -> dict[str, float]:
        return asdict(self)

    def fit(self, samples: Sequence[TTFTSample]) -> TTFTEstimator:
        """Least-squares fit of the three coefficients to observed TTFTs.

        Coefficients are constrained to be non-negative (a negative prefill
        cost is noise, not physics). A feature that never varies in the data
        (say, pending is always 0 on a single idle worker) cannot be fitted;
        its coefficient keeps this estimator's current value and its
        contribution is subtracted before fitting the rest.
        """
        if len(samples) < 2:
            raise ValueError("need at least two samples to fit")
        current = [self.queue_wait_ms_per_pending, self.prefill_ms_per_token, 1.0]
        columns = [
            [float(s.pending) for s in samples],
            [s.uncached_tokens for s in samples],
            [1.0] * len(samples),
        ]
        target = [s.ttft_ms for s in samples]

        free = [i for i in (0, 1) if max(columns[i]) > min(columns[i])]
        held = [(i, current[i]) for i in (0, 1) if i not in free]  # unidentifiable: keep as is
        for i, coefficient in held:
            target = [t - coefficient * x for t, x in zip(target, columns[i], strict=True)]

        active = [*free, 2]
        coefficients: dict[int, float] = {i: value for i, value in held}
        while True:
            solved = _least_squares([columns[i] for i in active], target)
            negatives = [(value, i) for i, value in zip(active, solved, strict=True) if value < 0]
            if not negatives:
                coefficients.update(zip(active, solved, strict=True))
                break
            _, worst = min(negatives)
            coefficients[worst] = 0.0
            active.remove(worst)
            if not active:
                break
        return replace(
            self,
            queue_wait_ms_per_pending=coefficients.get(0, 0.0),
            prefill_ms_per_token=coefficients.get(1, 0.0),
            fixed_overhead_ms=coefficients.get(2, 0.0),
        )

    def evaluate(self, samples: Sequence[TTFTSample]) -> FitReport:
        if not samples:
            raise ValueError("no samples to evaluate")
        predicted = [self.predict_from(s.pending, s.uncached_tokens) for s in samples]
        observed = [s.ttft_ms for s in samples]
        errors = [p - o for p, o in zip(predicted, observed, strict=True)]
        mean_observed = sum(observed) / len(observed)
        ss_res = sum(e * e for e in errors)
        ss_tot = sum((o - mean_observed) ** 2 for o in observed)
        return FitReport(
            samples=len(samples),
            mae_ms=sum(abs(e) for e in errors) / len(errors),
            bias_ms=sum(errors) / len(errors),
            r2=1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0),
        )


def _least_squares(columns: Sequence[Sequence[float]], target: Sequence[float]) -> list[float]:
    """Ordinary least squares via the normal equations; tiny systems only."""
    k = len(columns)
    gram = [[_dot(columns[i], columns[j]) for j in range(k)] for i in range(k)]
    rhs = [_dot(columns[i], target) for i in range(k)]
    return _solve(gram, rhs)


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(rhs)
    rows = [[*matrix[i], rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            raise ValueError("samples do not determine the coefficients (singular fit)")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r == col:
                continue
            factor = rows[r][col] / rows[col][col]
            if factor:
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col], strict=True)]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def healthy_workers(workers: list[WorkerState]) -> list[WorkerState]:
    healthy = sorted(
        (w for w in workers if w.status == WorkerStatus.HEALTHY), key=lambda w: w.worker_id
    )
    if not healthy:
        raise NoHealthyWorkersError()
    return healthy


def load(worker: WorkerState) -> int:
    return worker.queue_depth + worker.active_requests


def kv_pressure(worker: WorkerState) -> float:
    if worker.kv_capacity_bytes <= 0:
        return 0.0
    return worker.kv_used_bytes / worker.kv_capacity_bytes


def describe_candidates(
    request: InferenceRequest,
    workers: list[WorkerState],
    kv_directory: KVDirectory,
    estimator: TTFTEstimator,
) -> list[CandidateScore]:
    """Common per-worker facts every policy records, before it applies its own score."""
    candidates = []
    for worker in workers:
        overlap = kv_directory.cache_overlap(request, worker.worker_id)
        candidates.append(
            CandidateScore(
                worker_id=worker.worker_id,
                cache_overlap=overlap,
                queue_depth=worker.queue_depth,
                active_requests=worker.active_requests,
                estimated_ttft_ms=estimator.predict(
                    worker, request.prompt_tokens_estimate, overlap
                ),
                kv_pressure=kv_pressure(worker),
                final_score=0.0,
            )
        )
    return candidates
