# Benchmarking

Every number here comes from **three deterministic simulated inference workers**, not GPUs. The simulator's cost model is described in the README; the benchmark replays a seeded workload through the real gateway and schedulers on a discrete-event clock, so results are exactly reproducible and independent of wall-clock speed.

## Reproduce

```bash
make bench WORKLOAD=shared_prefix_heavy REQUESTS=1000 SEED=42
# or, one policy at a time:
python -m benchmark.runner --workload shared_prefix_heavy --policy kv_aware --requests 1000 --seed 42
python -m benchmark.analysis benchmark/results/shared_prefix_heavy_*_seed42.json
```

Outputs land in `benchmark/results/` as one JSON (run metadata + aggregate metrics) and one CSV (per-request rows) per run. The JSON files for the runs below are committed; CSVs are not.

## Metric definitions

- **TTFT** — queue wait + uncached prefill + fixed overhead + seeded noise, from the simulator's cost model.
- **Lat** — TTFT + (output_tokens − 1) × decode step.
- **Queue mean** — mean simulated queue wait.
- **Hit rate** — fraction of requests whose selected worker held ≥ 1 leading prefix chunk.
- **Prefill saved** — 1 − (effective prefill tokens ÷ prompt tokens), summed over all requests.
- **Imbalance** — requests on the busiest worker ÷ mean per worker (1.00 = perfectly even).
- **Evictions** — LRU evictions across all workers' KV directories.

Deltas in parentheses are relative to `round_robin` on the identical seeded request stream.

## Results — 1000 requests, seed 42, 3 workers, default weights (α=2.0, β=0.8, γ=0.4, δ=0.002)

#### bursty (1000 requests, seed 42, 3 workers)

| Policy | TTFT p50 | TTFT p99 | Lat p99 | Queue mean | Hit rate | Prefill saved | Imbalance | Evictions |
|---|---|---|---|---|---|---|---|---|
| kv_aware | 179 ms (-30.7%) | 365 ms (-14.2%) | 869 ms (-6.5%) | 75 ms (+4.6%) | 57.7% (+18.4 pp) | 57.2% (+18.2 pp) | 1.15 (+14.4%) | 7424 |
| least_loaded | 257 ms (-0.7%) | 405 ms (-4.9%) | 909 ms (-2.3%) | 70 ms (-2.0%) | 40.6% (+1.3 pp) | 40.2% (+1.3 pp) | 1.01 (+1.2%) | 9248 |
| round_robin | 258 ms | 426 ms | 930 ms | 72 ms | 39.3% | 38.9% | 1.00 | 9248 |

_Same-prefix requests arrive in bursts of 12 spaced 5ms apart, then a pause. Tests locality against queue overload when the cache is not yet warm for the burst._

#### cache_pressure (1000 requests, seed 42, 3 workers)

| Policy | TTFT p50 | TTFT p99 | Lat p99 | Queue mean | Hit rate | Prefill saved | Imbalance | Evictions |
|---|---|---|---|---|---|---|---|---|
| kv_aware | 155 ms (-7.7%) | 199 ms (-1.8%) | 703 ms (-0.5%) | 46 ms (-7.6%) | 29.1% (+18.4 pp) | 28.6% (+18.1 pp) | 1.05 (+4.8%) | 11952 |
| least_loaded | 166 ms (-1.2%) | 202 ms (-0.1%) | 706 ms (-0.0%) | 47 ms (-5.1%) | 9.4% (-1.3 pp) | 9.2% (-1.3 pp) | 1.06 (+5.7%) | 13664 |
| round_robin | 168 ms | 203 ms | 707 ms | 50 ms | 10.7% | 10.5% | 1.00 | 13728 |

_40 distinct 2K-token prefixes (1 GiB of KV each) cycle uniformly against 4 GiB of KV per worker: the working set is ~3x total capacity. Tests eviction behavior._

#### hotspot (1000 requests, seed 42, 3 workers)

| Policy | TTFT p50 | TTFT p99 | Lat p99 | Queue mean | Hit rate | Prefill saved | Imbalance | Evictions |
|---|---|---|---|---|---|---|---|---|
| kv_aware | 135 ms (-3.6%) | 369 ms (-2.0%) | 873 ms (-0.9%) | 99 ms (-3.5%) | 61.0% (+0.3 pp) | 60.5% (+0.3 pp) | 1.01 (+0.9%) | 11712 |
| least_loaded | 135 ms (-3.5%) | 369 ms (-2.2%) | 873 ms (-1.0%) | 99 ms (-3.3%) | 60.6% (-0.1 pp) | 60.1% (-0.1 pp) | 1.04 (+3.3%) | 11712 |
| round_robin | 140 ms | 377 ms | 881 ms | 103 ms | 60.7% | 60.2% | 1.00 | 11680 |

_60% of a 40 rps stream shares one 4K-token prefix. Tests whether cache affinity concentrates load on one worker and hurts tail latency._

#### hotspot_tight (1000 requests, seed 42, 3 workers)

| Policy | TTFT p50 | TTFT p99 | Lat p99 | Queue mean | Hit rate | Prefill saved | Imbalance | Evictions |
|---|---|---|---|---|---|---|---|---|
| kv_aware | 262 ms (-18.3%) | 358 ms (-14.4%) | 862 ms (-6.5%) | 140 ms (+17.5%) | 55.2% (+21.6 pp) | 54.7% (+21.4 pp) | 1.60 (+59.9%) | 14848 |
| least_loaded | 319 ms (-0.8%) | 403 ms (-3.6%) | 907 ms (-1.6%) | 115 ms (-3.8%) | 33.2% (-0.4 pp) | 32.9% (-0.4 pp) | 1.02 (+1.5%) | 20416 |
| round_robin | 321 ms | 418 ms | 922 ms | 119 ms | 33.6% | 33.3% | 1.00 | 21760 |

_Same 60%-hot 40 rps stream as hotspot, but each worker can hold only ~one 4K-token prefix (2 GiB of KV). The hot prefix cannot replicate across workers, so cache affinity must concentrate load on whichever worker holds it. This is the stress test for the failure hypothesis._

#### independent (1000 requests, seed 42, 3 workers)

| Policy | TTFT p50 | TTFT p99 | Lat p99 | Queue mean | Hit rate | Prefill saved | Imbalance | Evictions |
|---|---|---|---|---|---|---|---|---|
| kv_aware | 164 ms (-1.5%) | 214 ms (-0.2%) | 718 ms (-0.0%) | 46 ms (-5.8%) | 0.0% (+0.0 pp) | 0.0% (+0.0 pp) | 1.06 (+6.0%) | 15616 |
| least_loaded | 164 ms (-1.8%) | 214 ms (-0.1%) | 718 ms (-0.0%) | 46 ms (-6.2%) | 0.0% (+0.0 pp) | 0.0% (+0.0 pp) | 1.06 (+6.0%) | 15616 |
| round_robin | 167 ms | 215 ms | 719 ms | 49 ms | 0.0% | 0.0% | 1.00 | 15616 |

_Every request has a unique 2K-token prefix. Locality cannot help; kv_aware should converge to load-aware routing._

#### shared_prefix_heavy (1000 requests, seed 42, 3 workers)

| Policy | TTFT p50 | TTFT p99 | Lat p99 | Queue mean | Hit rate | Prefill saved | Imbalance | Evictions |
|---|---|---|---|---|---|---|---|---|
| kv_aware | 66 ms (-16.1%) | 283 ms (-4.7%) | 787 ms (-1.7%) | 44 ms (-5.9%) | 79.2% (+19.8 pp) | 78.5% (+19.6 pp) | 1.13 (+12.6%) | 6144 |
| least_loaded | 77 ms (-2.7%) | 295 ms (-0.5%) | 799 ms (-0.2%) | 43 ms (-7.0%) | 58.6% (-0.8 pp) | 58.1% (-0.8 pp) | 1.06 (+6.0%) | 10496 |
| round_robin | 79 ms | 296 ms | 800 ms | 46 ms | 59.4% | 58.9% | 1.00 | 10656 |

_80% of traffic belongs to four 4K-token shared prefixes. Expect a clear locality benefit if queue imbalance stays bounded._

#### shared_prefix_small (1000 requests, seed 42, 3 workers)

| Policy | TTFT p50 | TTFT p99 | Lat p99 | Queue mean | Hit rate | Prefill saved | Imbalance | Evictions |
|---|---|---|---|---|---|---|---|---|
| kv_aware | 160 ms (-2.2%) | 203 ms (-1.6%) | 707 ms (-0.5%) | 48 ms (-6.2%) | 24.8% (+1.6 pp) | 24.4% (+1.6 pp) | 1.05 (+5.1%) | 11632 |
| least_loaded | 161 ms (-1.3%) | 201 ms (-2.9%) | 705 ms (-0.8%) | 48 ms (-6.0%) | 23.4% (+0.2 pp) | 23.0% (+0.2 pp) | 1.05 (+4.8%) | 11744 |
| round_robin | 164 ms | 207 ms | 711 ms | 51 ms | 23.2% | 22.8% | 1.00 | 11776 |

_25% of traffic shares one 2K-token prefix; the rest is unique. Expect a moderate locality benefit._

## Weight sensitivity (α/β/γ/δ), one at a time

`python -m benchmark.sweep --all-workloads` replays the *same* seed-42 streams under 15 `kv_aware` settings: the defaults (α=2, β=0.8, γ=0.4, δ=0.002) and, for each weight in turn, the other three held at their defaults while it takes each value in

| Weight | Values swept |
| --- | --- |
| α (cache overlap reward) | 0, 0.5, 1, **2**, 4, 8 |
| β (relative load penalty) | 0, 0.2, **0.8**, 2, 4 |
| γ (KV pressure penalty) | 0, **0.4**, 2 |
| δ (predicted-TTFT penalty, per ms) | 0, **0.002**, 0.01, 0.05 |

Deltas are against the **default** row; `round_robin` and `least_loaded` are shown for reference. Run JSONs are in `benchmark/results/sweep/`; `python -m benchmark.sweep --table benchmark/results/sweep/*.json` regenerates these tables.

```bash
python -m benchmark.sweep --all-workloads --requests 1000 --seed 42
python -m benchmark.sweep --workload bursty --grid beta=0,0.4,0.8,1.6,3.2   # custom grid for one weight
```

### The three workloads where weights matter

#### bursty (1000 requests, seed 42, 3 workers)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 257 ms (+43.2%) | 405 ms (+10.9%) | 70 ms (-6.4%) | 40.6% (-17.1 pp) | 1.01 (-11.5%) | 9248 |
| round_robin (reference) | 258 ms (+44.3%) | 426 ms (+16.6%) | 72 ms (-4.4%) | 39.3% (-18.4 pp) | 1.00 (-12.6%) | 9248 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 179 ms | 365 ms | 75 ms | 57.7% | 1.15 | 7424 |
| alpha=0 | 222 ms (+23.9%) | 368 ms (+0.7%) | 67 ms (-10.8%) | 49.4% (-8.3 pp) | 1.03 (-10.5%) | 8768 |
| alpha=0.5 | 222 ms (+23.8%) | 377 ms (+3.0%) | 72 ms (-3.9%) | 50.5% (-7.2 pp) | 1.24 (+8.1%) | 8288 |
| alpha=1 | 179 ms (+0.0%) | 365 ms (+0.0%) | 75 ms (+0.0%) | 57.7% (+0.0 pp) | 1.15 (+0.0%) | 7424 |
| alpha=4 | 179 ms (+0.0%) | 365 ms (+0.0%) | 75 ms (+0.0%) | 57.7% (+0.0 pp) | 1.15 (+0.0%) | 7424 |
| alpha=8 | 179 ms (+0.0%) | 365 ms (+0.0%) | 75 ms (+0.0%) | 57.7% (+0.0 pp) | 1.15 (+0.0%) | 7424 |
| beta=0 | 166 ms (-7.5%) | 363 ms (-0.6%) | 75 ms (+0.4%) | 59.5% (+1.8 pp) | 1.21 (+5.5%) | 7264 |
| beta=0.2 | 226 ms (+26.2%) | 385 ms (+5.4%) | 78 ms (+3.9%) | 50.6% (-7.1 pp) | 1.31 (+14.4%) | 8288 |
| beta=2 | 187 ms (+4.1%) | 361 ms (-1.3%) | 81 ms (+7.5%) | 60.7% (+3.0 pp) | 1.39 (+20.9%) | 7296 |
| beta=4 | 226 ms (+26.3%) | 426 ms (+16.6%) | 69 ms (-7.6%) | 49.1% (-8.6 pp) | 1.04 (-9.7%) | 8576 |
| gamma=0 | 195 ms (+8.8%) | 388 ms (+6.1%) | 73 ms (-2.9%) | 52.9% (-4.8 pp) | 1.13 (-1.3%) | 7904 |
| gamma=2 | 167 ms (-6.9%) | 372 ms (+1.7%) | 76 ms (+1.0%) | 61.1% (+3.4 pp) | 1.22 (+6.5%) | 7072 |
| delta=0 | 214 ms (+19.2%) | 371 ms (+1.7%) | 75 ms (-0.2%) | 54.2% (-3.5 pp) | 1.21 (+5.5%) | 7648 |
| delta=0.01 | 173 ms (-3.3%) | 354 ms (-3.3%) | 81 ms (+7.6%) | 62.5% (+4.8 pp) | 1.22 (+6.0%) | 7136 |
| delta=0.05 | 206 ms (+15.2%) | 378 ms (+3.3%) | 76 ms (+1.3%) | 54.9% (-2.8 pp) | 1.36 (+18.8%) | 7872 |

#### hotspot_tight (1000 requests, seed 42, 3 workers)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 319 ms (+21.5%) | 403 ms (+12.6%) | 115 ms (-18.1%) | 33.2% (-22.0 pp) | 1.02 (-36.5%) | 20416 |
| round_robin (reference) | 321 ms (+22.4%) | 418 ms (+16.8%) | 119 ms (-14.9%) | 33.6% (-21.6 pp) | 1.00 (-37.5%) | 21760 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 262 ms | 358 ms | 140 ms | 55.2% | 1.60 | 14848 |
| alpha=0 | 275 ms (+4.8%) | 379 ms (+5.8%) | 109 ms (-22.3%) | 47.8% (-7.4 pp) | 1.01 (-37.1%) | 19328 |
| alpha=0.5 | 214 ms (-18.3%) | 370 ms (+3.4%) | 114 ms (-18.8%) | 54.1% (-1.1 pp) | 1.22 (-24.0%) | 16544 |
| alpha=1 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| alpha=4 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| alpha=8 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| beta=0 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| beta=0.2 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| beta=2 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| beta=4 | 193 ms (-26.3%) | 374 ms (+4.4%) | 109 ms (-21.9%) | 53.9% (-1.3 pp) | 1.12 (-30.0%) | 17664 |
| gamma=0 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| gamma=2 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| delta=0 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| delta=0.01 | 262 ms (+0.0%) | 358 ms (+0.0%) | 140 ms (+0.0%) | 55.2% (+0.0 pp) | 1.60 (+0.0%) | 14848 |
| delta=0.05 | 252 ms (-3.8%) | 363 ms (+1.4%) | 122 ms (-12.8%) | 54.8% (-0.4 pp) | 1.34 (-16.3%) | 16320 |

#### shared_prefix_heavy (1000 requests, seed 42, 3 workers)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 77 ms (+15.9%) | 295 ms (+4.4%) | 43 ms (-1.2%) | 58.6% (-20.6 pp) | 1.06 (-5.9%) | 10496 |
| round_robin (reference) | 79 ms (+19.2%) | 296 ms (+4.9%) | 46 ms (+6.2%) | 59.4% (-19.8 pp) | 1.00 (-11.2%) | 10656 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 66 ms | 283 ms | 44 ms | 79.2% | 1.13 | 6144 |
| alpha=0 | 64 ms (-2.6%) | 285 ms (+0.9%) | 41 ms (-6.6%) | 74.9% (-4.3 pp) | 1.07 (-5.6%) | 7968 |
| alpha=0.5 | 66 ms (-0.3%) | 282 ms (-0.4%) | 42 ms (-3.3%) | 78.3% (-0.9 pp) | 1.10 (-2.7%) | 6464 |
| alpha=1 | 66 ms (+0.0%) | 283 ms (+0.0%) | 44 ms (+0.0%) | 79.2% (+0.0 pp) | 1.13 (+0.0%) | 6144 |
| alpha=4 | 66 ms (+0.0%) | 283 ms (+0.0%) | 44 ms (+0.0%) | 79.2% (+0.0 pp) | 1.13 (+0.0%) | 6144 |
| alpha=8 | 66 ms (+0.0%) | 283 ms (+0.0%) | 44 ms (+0.0%) | 79.2% (+0.0 pp) | 1.13 (+0.0%) | 6144 |
| beta=0 | 80 ms (+21.5%) | 275 ms (-2.7%) | 51 ms (+17.4%) | 79.7% (+0.5 pp) | 1.27 (+12.5%) | 5952 |
| beta=0.2 | 67 ms (+2.0%) | 282 ms (-0.3%) | 44 ms (+0.9%) | 79.7% (+0.5 pp) | 1.14 (+1.3%) | 5984 |
| beta=2 | 67 ms (+1.6%) | 283 ms (+0.3%) | 43 ms (-1.8%) | 78.6% (-0.6 pp) | 1.13 (+0.5%) | 6176 |
| beta=4 | 66 ms (-0.5%) | 286 ms (+1.1%) | 41 ms (-6.5%) | 75.3% (-3.9 pp) | 1.08 (-4.0%) | 7296 |
| gamma=0 | 67 ms (+0.9%) | 279 ms (-1.4%) | 43 ms (-1.8%) | 79.1% (-0.1 pp) | 1.07 (-5.6%) | 6112 |
| gamma=2 | 67 ms (+2.0%) | 282 ms (-0.3%) | 44 ms (+0.9%) | 79.7% (+0.5 pp) | 1.14 (+1.3%) | 5984 |
| delta=0 | 66 ms (+0.0%) | 283 ms (+0.0%) | 44 ms (+0.0%) | 79.2% (+0.0 pp) | 1.13 (+0.0%) | 6144 |
| delta=0.01 | 67 ms (+1.6%) | 283 ms (+0.3%) | 43 ms (-1.8%) | 78.6% (-0.6 pp) | 1.13 (+0.5%) | 6176 |
| delta=0.05 | 67 ms (+1.6%) | 283 ms (+0.3%) | 43 ms (-1.8%) | 78.6% (-0.6 pp) | 1.13 (+0.5%) | 6176 |

### The four where they do not

Across all 15 settings, the largest TTFT p50 movement on each of these is within the noise of a single seed:

| Workload | Largest ΔTTFT p50 vs default | Setting | Hit-rate range |
| --- | --- | --- | --- |
| independent | −0.2% | γ=0 | 0% everywhere |
| hotspot (8 GiB/worker) | +1.7% | γ=2 | 61.0% everywhere |
| cache_pressure | +2.4% | α=0 | 21.7% – 29.1% |
| shared_prefix_small | −2.0% | β=0.2 | 23.7% – 24.8% |

<details>
<summary>Full tables for the insensitive workloads</summary>

#### independent (1000 requests, seed 42, 3 workers)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 164 ms (-0.2%) | 214 ms (+0.0%) | 46 ms (-0.4%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| round_robin (reference) | 167 ms (+1.6%) | 215 ms (+0.2%) | 49 ms (+6.2%) | 0.0% (+0.0 pp) | 1.00 (-5.6%) | 15616 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 164 ms | 214 ms | 46 ms | 0.0% | 1.06 | 15616 |
| alpha=0 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| alpha=0.5 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| alpha=1 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| alpha=4 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| alpha=8 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| beta=0 | 164 ms (-0.1%) | 214 ms (+0.0%) | 46 ms (+0.5%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| beta=0.2 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| beta=2 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| beta=4 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| gamma=0 | 164 ms (-0.2%) | 214 ms (+0.0%) | 46 ms (-0.4%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| gamma=2 | 164 ms (+0.0%) | 214 ms (+0.1%) | 46 ms (+0.2%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| delta=0 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| delta=0.01 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |
| delta=0.05 | 164 ms (+0.0%) | 214 ms (+0.0%) | 46 ms (+0.0%) | 0.0% (+0.0 pp) | 1.06 (+0.0%) | 15616 |

#### hotspot (1000 requests, seed 42, 3 workers)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 135 ms (+0.1%) | 369 ms (-0.2%) | 99 ms (+0.2%) | 60.6% (-0.4 pp) | 1.04 (+2.4%) | 11712 |
| round_robin (reference) | 140 ms (+3.7%) | 377 ms (+2.1%) | 103 ms (+3.6%) | 60.7% (-0.3 pp) | 1.00 (-0.9%) | 11680 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 135 ms | 369 ms | 99 ms | 61.0% | 1.01 | 11712 |
| alpha=0 | 135 ms (+0.0%) | 369 ms (+0.0%) | 99 ms (+0.0%) | 61.0% (+0.0 pp) | 1.01 (+0.0%) | 11712 |
| alpha=0.5 | 135 ms (+0.0%) | 369 ms (+0.0%) | 99 ms (+0.0%) | 61.0% (+0.0 pp) | 1.01 (+0.0%) | 11712 |
| alpha=1 | 135 ms (+0.0%) | 369 ms (+0.0%) | 99 ms (+0.0%) | 61.0% (+0.0 pp) | 1.01 (+0.0%) | 11712 |
| alpha=4 | 135 ms (+0.0%) | 369 ms (+0.0%) | 99 ms (+0.0%) | 61.0% (+0.0 pp) | 1.01 (+0.0%) | 11712 |
| alpha=8 | 135 ms (+0.0%) | 369 ms (+0.0%) | 99 ms (+0.0%) | 61.0% (+0.0 pp) | 1.01 (+0.0%) | 11712 |
| beta=0 | 137 ms (+1.4%) | 375 ms (+1.7%) | 100 ms (+0.4%) | 61.0% (+0.0 pp) | 1.02 (+0.9%) | 11712 |
| beta=0.2 | 137 ms (+1.4%) | 369 ms (-0.0%) | 99 ms (+0.1%) | 61.0% (+0.0 pp) | 1.02 (+1.2%) | 11648 |
| beta=2 | 134 ms (-0.6%) | 369 ms (-0.1%) | 99 ms (-0.0%) | 61.0% (+0.0 pp) | 1.02 (+0.6%) | 11712 |
| beta=4 | 135 ms (+0.0%) | 371 ms (+0.5%) | 100 ms (+0.4%) | 61.0% (+0.0 pp) | 1.03 (+2.1%) | 11744 |
| gamma=0 | 134 ms (-0.6%) | 376 ms (+1.7%) | 99 ms (-0.1%) | 61.0% (+0.0 pp) | 1.02 (+0.9%) | 11776 |
| gamma=2 | 137 ms (+1.7%) | 379 ms (+2.7%) | 100 ms (+0.4%) | 61.0% (+0.0 pp) | 1.02 (+0.6%) | 11808 |
| delta=0 | 136 ms (+1.0%) | 369 ms (-0.0%) | 99 ms (-0.0%) | 61.0% (+0.0 pp) | 1.03 (+2.1%) | 11680 |
| delta=0.01 | 136 ms (+1.0%) | 369 ms (-0.1%) | 99 ms (+0.1%) | 61.0% (+0.0 pp) | 1.03 (+1.5%) | 11840 |
| delta=0.05 | 135 ms (+0.0%) | 371 ms (+0.5%) | 100 ms (+0.4%) | 61.0% (+0.0 pp) | 1.03 (+2.1%) | 11744 |

#### cache_pressure (1000 requests, seed 42, 3 workers)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 166 ms (+7.0%) | 202 ms (+1.7%) | 47 ms (+2.8%) | 9.4% (-19.7 pp) | 1.06 (+0.9%) | 13664 |
| round_robin (reference) | 168 ms (+8.3%) | 203 ms (+1.8%) | 50 ms (+8.3%) | 10.7% (-18.4 pp) | 1.00 (-4.6%) | 13728 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 155 ms | 199 ms | 46 ms | 29.1% | 1.05 | 11952 |
| alpha=0 | 159 ms (+2.4%) | 200 ms (+0.3%) | 46 ms (-0.9%) | 21.7% (-7.4 pp) | 1.05 (-0.3%) | 12944 |
| alpha=0.5 | 155 ms (+0.0%) | 199 ms (+0.0%) | 46 ms (+0.0%) | 29.1% (+0.0 pp) | 1.05 (+0.0%) | 11952 |
| alpha=1 | 155 ms (+0.0%) | 199 ms (+0.0%) | 46 ms (+0.0%) | 29.1% (+0.0 pp) | 1.05 (+0.0%) | 11952 |
| alpha=4 | 155 ms (+0.0%) | 199 ms (+0.0%) | 46 ms (+0.0%) | 29.1% (+0.0 pp) | 1.05 (+0.0%) | 11952 |
| alpha=8 | 155 ms (+0.0%) | 199 ms (+0.0%) | 46 ms (+0.0%) | 29.1% (+0.0 pp) | 1.05 (+0.0%) | 11952 |
| beta=0 | 156 ms (+0.5%) | 199 ms (-0.1%) | 46 ms (-0.9%) | 26.7% (-2.4 pp) | 1.06 (+0.9%) | 12496 |
| beta=0.2 | 155 ms (+0.2%) | 199 ms (-0.1%) | 46 ms (-0.9%) | 27.3% (-1.8 pp) | 1.06 (+0.9%) | 12448 |
| beta=2 | 155 ms (+0.0%) | 199 ms (+0.0%) | 46 ms (+0.0%) | 29.1% (+0.0 pp) | 1.05 (+0.0%) | 11952 |
| beta=4 | 156 ms (+0.7%) | 196 ms (-1.3%) | 46 ms (-0.5%) | 25.9% (-3.2 pp) | 1.06 (+0.9%) | 12624 |
| gamma=0 | 156 ms (+0.4%) | 202 ms (+1.6%) | 46 ms (-0.4%) | 26.9% (-2.2 pp) | 1.08 (+3.1%) | 12528 |
| gamma=2 | 155 ms (-0.1%) | 199 ms (-0.1%) | 46 ms (-0.6%) | 28.4% (-0.7 pp) | 1.06 (+1.1%) | 12448 |
| delta=0 | 155 ms (+0.0%) | 199 ms (+0.0%) | 46 ms (+0.0%) | 29.1% (+0.0 pp) | 1.05 (+0.0%) | 11952 |
| delta=0.01 | 155 ms (+0.0%) | 199 ms (+0.0%) | 46 ms (+0.0%) | 29.1% (+0.0 pp) | 1.05 (+0.0%) | 11952 |
| delta=0.05 | 155 ms (+0.0%) | 199 ms (+0.0%) | 46 ms (+0.0%) | 29.1% (+0.0 pp) | 1.05 (+0.0%) | 11952 |

#### shared_prefix_small (1000 requests, seed 42, 3 workers)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 161 ms (+0.9%) | 201 ms (-1.3%) | 48 ms (+0.3%) | 23.4% (-1.4 pp) | 1.05 (-0.3%) | 11744 |
| round_robin (reference) | 164 ms (+2.3%) | 207 ms (+1.7%) | 51 ms (+6.6%) | 23.2% (-1.6 pp) | 1.00 (-4.8%) | 11776 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 160 ms | 203 ms | 48 ms | 24.8% | 1.05 | 11632 |
| alpha=0 | 161 ms (+0.6%) | 202 ms (-0.4%) | 48 ms (+0.1%) | 23.7% (-1.1 pp) | 1.04 (-1.4%) | 11792 |
| alpha=0.5 | 160 ms (+0.0%) | 203 ms (+0.0%) | 48 ms (+0.0%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |
| alpha=1 | 160 ms (+0.0%) | 203 ms (+0.0%) | 48 ms (+0.0%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |
| alpha=4 | 160 ms (+0.0%) | 203 ms (+0.0%) | 48 ms (+0.0%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |
| alpha=8 | 160 ms (+0.0%) | 203 ms (+0.0%) | 48 ms (+0.0%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |
| beta=0 | 160 ms (+0.0%) | 202 ms (-0.4%) | 48 ms (+0.6%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |
| beta=0.2 | 157 ms (-2.0%) | 200 ms (-1.5%) | 49 ms (+1.8%) | 24.8% (+0.0 pp) | 1.17 (+11.4%) | 11632 |
| beta=2 | 160 ms (+0.0%) | 203 ms (+0.0%) | 48 ms (+0.0%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |
| beta=4 | 161 ms (+0.4%) | 203 ms (-0.2%) | 48 ms (+0.7%) | 24.7% (-0.1 pp) | 1.05 (-0.6%) | 11648 |
| gamma=0 | 160 ms (-0.0%) | 201 ms (-1.3%) | 48 ms (+0.4%) | 24.8% (+0.0 pp) | 1.06 (+0.3%) | 11632 |
| gamma=2 | 159 ms (-0.9%) | 201 ms (-1.3%) | 49 ms (+2.6%) | 24.8% (+0.0 pp) | 1.13 (+7.4%) | 11632 |
| delta=0 | 160 ms (+0.0%) | 203 ms (+0.0%) | 48 ms (+0.0%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |
| delta=0.01 | 160 ms (+0.0%) | 203 ms (+0.0%) | 48 ms (+0.0%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |
| delta=0.05 | 160 ms (+0.0%) | 203 ms (+0.0%) | 48 ms (+0.0%) | 24.8% (+0.0 pp) | 1.05 (+0.0%) | 11632 |

</details>

### Replication on two more seeds

Because routing is discrete, one seed can flatter or punish a setting by accident. `bursty` and `hotspot_tight` were re-swept on seeds 7 and 123 (`benchmark/results/sweep/*_seed7.json`, `*_seed123.json`). TTFT p50 / hit rate / imbalance:

| Workload | Setting | seed 42 | seed 7 | seed 123 |
| --- | --- | --- | --- | --- |
| hotspot_tight | default | 262 ms / 55.2% / **1.60** | 152 ms / 59.9% / 1.18 | 189 ms / 53.7% / 1.13 |
| hotspot_tight | β=4 | 193 ms (−26%) / 53.9% / 1.12 | 151 ms (−0.6%) / 59.3% / 1.11 | 187 ms (−1.0%) / 52.6% / 1.11 |
| hotspot_tight | α=0.5 | 214 ms (−18%) / 54.1% / 1.22 | 152 ms (−0.1%) / 59.2% / 1.09 | 193 ms (+2.3%) / 54.1% / 1.15 |
| hotspot_tight | α=0 | 275 ms (+5%) / 47.8% / 1.01 | 152 ms (0%) / 55.1% / 1.08 | 227 ms (+20%) / 49.9% / 1.02 |
| hotspot_tight | δ=0.05 | 252 ms (−3.8%) / 54.8% / 1.34 | 148 ms (−2.8%) / 59.6% / 1.10 | 189 ms (0%) / 53.7% / 1.13 |
| bursty | default | 179 ms / 57.7% / 1.15 | 166 ms / 56.8% / 1.05 | 225 ms / 46.9% / 1.17 |
| bursty | β=0 | 166 ms (−7.5%) / 59.5% / 1.21 | 154 ms (−6.8%) / 67.3% / **1.63**, queue +25% | 225 ms (0%) / 46.9% / 1.17 |
| bursty | β=4 | 226 ms (+26%) / 49.1% / 1.04 | 172 ms (+4%) / 54.6% / 1.09 | 222 ms (−1%) / 47.5% / 1.01 |
| bursty | δ=0 | 214 ms (+19%) / 54.2% / 1.21 | 168 ms (+1.6%) / 56.8% / 1.05 | 225 ms (0%) / 46.9% / 1.17 |
| bursty | α=0 | 222 ms (+24%) / 49.4% / 1.03 | 160 ms (−3.4%) / 54.6% / 1.05 | 219 ms (−2.5%) / 49.0% / 1.07 |

What replicates: the α plateau (α ≥ 1 is byte-identical to the default on every seed and both workloads); β=4 on `hotspot_tight` pulls imbalance to ≈1.11 on every seed, costs at most 1.3 points of hit rate, and never raises TTFT; α=0 on `hotspot_tight` costs 4–7 points of hit rate on every seed and 0–20% TTFT. What does not: the −26% TTFT for β=4 was specific to seed 42, where the *default* happened to concentrate at 1.60 (1.13–1.18 on the other seeds); δ=0's +19% and β=0's −7% on `bursty` are seed-dependent, and on seed 7 β=0 buys its TTFT with a 1.63 imbalance and +25% queue wait. The spread of the **default** itself across seeds (bursty 166–225 ms, hotspot_tight 152–262 ms) is larger than every weight effect except α=0 — seed variance, not weights, is the first-order term in this simulator.

### What the sweep says

- **α is a threshold, not a dial.** Every α ≥ 1 — and on most workloads every α ≥ 0.5 — routes *byte-identically* to α=2, on every seed tried. Overlap in this simulator is nearly binary (a prefix is resident or it is not) and the penalties are bounded (β·1 + γ·1 ≤ 1.2 at defaults, δ·TTFT ≈ 0.03–0.6), so once α clears their sum the cached worker wins every comparison and raising α further changes nothing. α=0 is the one α that costs: 4–7 points of hit rate on every seed and 0–20% TTFT on `hotspot_tight`. It still keeps most of the locality (`shared_prefix_heavy` 74.9% vs 79.2%) because predicted TTFT already prices uncached prefill — δ carries locality on its own.
- **β=4 is a safe way to cap hotspot concentration.** On `hotspot_tight` it pulls imbalance to ≈1.11 on all three seeds, gives up at most 1.3 points of hit rate, and never raises TTFT p50. How much TTFT it *saves* depends on how badly the default happened to concentrate on that seed: −26% on seed 42 (default imbalance 1.60), ≈0% on seeds 7 and 123 (1.18, 1.13). On `bursty` β=4 is never better and was +26% on one seed; β=0 is the mirror image, −7% TTFT on two seeds but with a 1.63 imbalance and +25% queue wait on one of them. No single β wins both workloads.
- **δ and γ effects do not survive replication.** δ=0 cost +19% on `bursty` seed 42 and +1.6% / 0% on the other seeds; δ=0.01's −3.3% and γ=2's −6.9% are seed-42 only. Their honest reading: harmless at the defaults, and not worth tuning on simulated data.
- **The response is not monotonic, and seed variance is the first-order term.** On `bursty` seed 42, β = 0 / 0.2 / 0.8 / 2 / 4 gives TTFT p50 of 166 / 226 / 179 / 187 / 226 ms. Routing is discrete and path-dependent — one early decision changes which worker warms which prefix and everything downstream — and the default's own spread across seeds (bursty 166–225 ms, hotspot_tight 152–262 ms) exceeds every weight effect except α=0. Single-seed deltas under ~10% are noise. See the replication table above.
- **Where the policy does not matter, the weights do not either.** The four insensitive workloads are exactly the ones where `kv_aware` was already within a few percent of the baselines.

## Under saturation: continuous-batching workers

Everything above uses the *linear* worker model: 12 ms per pending request, unbounded concurrency. It cannot saturate, so it cannot show the failure the hotspot workloads were written to provoke. `--queue-model batched` swaps in a continuous-batching engine (`workers/batched.py`, described in [`kv-cache-design.md`](kv-cache-design.md#5-saturation-why-a-second-simulator-was-necessary)): a FIFO queue, a running batch of at most 16 sequences within a 64K-token KV budget, one prefill pass for newly admitted sequences followed by one decode step for the whole batch whose cost grows with batch size. A cold 4K-token prefill stalls the batch for ≈205 ms, so cache hits raise the worker's *capacity*, not only the latency of the hit.

```bash
python -m benchmark.loadscan --workload hotspot_tight --workload shared_prefix_heavy --rates 10,20,30,40,50 --requests 600
python -m benchmark.sweep --workload hotspot_tight --queue-model batched --rate 10 --point beta=0,gamma=0,delta=0
python -m benchmark.runner --workload hotspot --policy kv_aware --queue-model batched --max-batch-size 8 --rate 12
```

Run JSONs are in `benchmark/results/batched/` (`batched16_<workload>_r<rate>_<policy>_seed42.json`) and `benchmark/results/batched/sweep/`.

### Load scan — 600 requests, seed 42, 3 batched workers, offered load 10 → 50 req/s

With 64-token outputs the three workers together sustain about 16–17 req/s; the knee is between 10 and 20 req/s. Deltas are against `round_robin` at the same offered load. Throughput is completed requests over the makespan, so above the knee it is the capacity the policy achieved.

#### hotspot_tight (600 requests, seed 42, 3 batched workers, max batch 16)

| Offered load | Policy | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Throughput |
|---|---|---|---|---|---|---|---|
| 10 req/s | round_robin | 240 ms | 902 ms | 65 ms | 28.2% | 1.00 | 10.0 req/s |
| 10 req/s | least_loaded | 244 ms (+1.7%) | 994 ms (+10.2%) | 85 ms (+30.4%) | 25.2% (-3.0 pp) | 1.02 (+2.5%) | 10.0 req/s (+0.7%) |
| 10 req/s | kv_aware | 219 ms (-8.7%) | 604 ms (-33.0%) | 33 ms (-49.9%) | 50.2% (+22.0 pp) | 1.59 (+59.0%) | 10.1 req/s (+1.9%) |
| 20 req/s | round_robin | 8093 ms | 15424 ms | 7342 ms | 28.7% | 1.00 | 13.1 req/s |
| 20 req/s | least_loaded | 7851 ms (-3.0%) | 14723 ms (-4.5%) | 7046 ms (-4.0%) | 28.2% (-0.5 pp) | 1.05 (+5.0%) | 13.4 req/s (+2.4%) |
| 20 req/s | kv_aware | 4812 ms (-40.5%) | 8479 ms (-45.0%) | 4329 ms (-41.0%) | 44.3% (+15.7 pp) | 1.10 (+10.0%) | 15.3 req/s (+16.5%) |
| 30 req/s | round_robin | 14552 ms | 26402 ms | 13203 ms | 24.5% | 1.00 | 12.7 req/s |
| 30 req/s | least_loaded | 14572 ms (+0.1%) | 29697 ms (+12.5%) | 13673 ms (+3.6%) | 21.5% (-3.0 pp) | 1.02 (+2.0%) | 12.0 req/s (-5.5%) |
| 30 req/s | kv_aware | 10229 ms (-29.7%) | 21961 ms (-16.8%) | 9596 ms (-27.3%) | 42.5% (+18.0 pp) | 1.05 (+5.5%) | 14.0 req/s (+10.0%) |
| 40 req/s | round_robin | 18551 ms | 35145 ms | 16165 ms | 18.5% | 1.00 | 11.8 req/s |
| 40 req/s | least_loaded | 18756 ms (+1.1%) | 34736 ms (-1.2%) | 16203 ms (+0.2%) | 18.5% (+0.0 pp) | 1.00 (+0.0%) | 11.9 req/s (+0.8%) |
| 40 req/s | kv_aware | 14282 ms (-23.0%) | 32992 ms (-6.1%) | 14232 ms (-12.0%) | 36.8% (+18.3 pp) | 1.08 (+8.5%) | 12.5 req/s (+5.9%) |
| 50 req/s | round_robin | 19806 ms | 39500 ms | 18136 ms | 18.3% | 1.00 | 11.3 req/s |
| 50 req/s | least_loaded | 19982 ms (+0.9%) | 39704 ms (+0.5%) | 18235 ms (+0.5%) | 17.8% (-0.5 pp) | 1.00 (+0.0%) | 11.2 req/s (-0.4%) |
| 50 req/s | kv_aware | 17155 ms (-13.4%) | 36177 ms (-8.4%) | 16084 ms (-11.3%) | 36.3% (+18.0 pp) | 1.01 (+1.5%) | 12.3 req/s (+9.4%) |

#### shared_prefix_heavy (600 requests, seed 42, 3 batched workers, max batch 16)

| Offered load | Policy | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Throughput |
|---|---|---|---|---|---|---|---|
| 10 req/s | round_robin | 57 ms | 402 ms | 20 ms | 58.5% | 1.00 | 9.5 req/s |
| 10 req/s | least_loaded | 77 ms (+34.1%) | 453 ms (+12.7%) | 30 ms (+49.8%) | 59.8% (+1.3 pp) | 1.08 (+8.0%) | 9.5 req/s (+0.0%) |
| 10 req/s | kv_aware | 35 ms (-38.7%) | 452 ms (+12.4%) | 21 ms (+5.2%) | 79.3% (+20.8 pp) | 1.33 (+33.0%) | 9.6 req/s (+0.3%) |
| 20 req/s | round_robin | 3090 ms | 6924 ms | 3013 ms | 51.3% | 1.00 | 16.0 req/s |
| 20 req/s | least_loaded | 3581 ms (+15.9%) | 5008 ms (-27.7%) | 3088 ms (+2.5%) | 51.8% (+0.5 pp) | 1.04 (+4.5%) | 17.1 req/s (+6.8%) |
| 20 req/s | kv_aware | 598 ms (-80.6%) | 3439 ms (-50.3%) | 896 ms (-70.3%) | 68.0% (+16.7 pp) | 1.09 (+9.0%) | 18.9 req/s (+18.3%) |
| 30 req/s | round_robin | 9951 ms | 16221 ms | 8856 ms | 47.0% | 1.00 | 16.1 req/s |
| 30 req/s | least_loaded | 10014 ms (+0.6%) | 16317 ms (+0.6%) | 8994 ms (+1.6%) | 44.2% (-2.8 pp) | 1.01 (+1.5%) | 15.7 req/s (-2.5%) |
| 30 req/s | kv_aware | 7757 ms (-22.0%) | 8846 ms (-45.5%) | 6618 ms (-25.3%) | 62.8% (+15.8 pp) | 1.04 (+4.0%) | 20.1 req/s (+24.7%) |
| 40 req/s | round_robin | 14078 ms | 22578 ms | 12231 ms | 46.0% | 1.00 | 15.4 req/s |
| 40 req/s | least_loaded | 13748 ms (-2.3%) | 21841 ms (-3.3%) | 12050 ms (-1.5%) | 46.8% (+0.8 pp) | 1.00 (+0.0%) | 15.7 req/s (+2.1%) |
| 40 req/s | kv_aware | 11838 ms (-15.9%) | 17081 ms (-24.3%) | 10324 ms (-15.6%) | 59.5% (+13.5 pp) | 1.03 (+3.5%) | 18.2 req/s (+18.4%) |
| 50 req/s | round_robin | 17483 ms | 26010 ms | 15134 ms | 39.5% | 1.00 | 15.3 req/s |
| 50 req/s | least_loaded | 16328 ms (-6.6%) | 24835 ms (-4.5%) | 14391 ms (-4.9%) | 43.0% (+3.5 pp) | 1.00 (+0.0%) | 15.7 req/s (+2.4%) |
| 50 req/s | kv_aware | 15447 ms (-11.6%) | 20739 ms (-20.3%) | 12871 ms (-15.0%) | 55.0% (+15.5 pp) | 1.00 (+0.0%) | 17.7 req/s (+15.6%) |

#### hotspot (600 requests, seed 42, 3 batched workers, max batch 16)

| Offered load | Policy | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Throughput |
|---|---|---|---|---|---|---|---|
| 10 req/s | round_robin | 47 ms | 613 ms | 23 ms | 59.7% | 1.00 | 9.5 req/s |
| 10 req/s | least_loaded | 47 ms (-0.4%) | 450 ms (-26.6%) | 26 ms (+15.4%) | 61.5% (+1.8 pp) | 1.05 (+5.0%) | 9.5 req/s (+0.1%) |
| 10 req/s | kv_aware | 44 ms (-6.5%) | 613 ms (+0.1%) | 25 ms (+8.7%) | 61.7% (+2.0 pp) | 1.19 (+18.5%) | 9.6 req/s (+0.2%) |
| 20 req/s | round_robin | 2799 ms | 7252 ms | 3218 ms | 52.5% | 1.00 | 16.1 req/s |
| 20 req/s | least_loaded | 3384 ms (+20.9%) | 5264 ms (-27.4%) | 3012 ms (-6.4%) | 53.5% (+1.0 pp) | 1.02 (+2.5%) | 17.5 req/s (+8.8%) |
| 20 req/s | kv_aware | 3082 ms (+10.1%) | 4822 ms (-33.5%) | 2768 ms (-14.0%) | 54.3% (+1.8 pp) | 1.05 (+5.5%) | 17.5 req/s (+9.1%) |
| 30 req/s | round_robin | 9771 ms | 15096 ms | 8938 ms | 49.8% | 1.00 | 16.5 req/s |
| 30 req/s | least_loaded | 9806 ms (+0.4%) | 14777 ms (-2.1%) | 8941 ms (+0.0%) | 49.8% (+0.0 pp) | 1.01 (+1.5%) | 16.1 req/s (-2.4%) |
| 30 req/s | kv_aware | 9929 ms (+1.6%) | 13748 ms (-8.9%) | 8762 ms (-2.0%) | 50.3% (+0.5 pp) | 1.05 (+5.5%) | 17.0 req/s (+3.1%) |
| 40 req/s | round_robin | 14339 ms | 21523 ms | 12684 ms | 45.3% | 1.00 | 15.8 req/s |
| 40 req/s | least_loaded | 14236 ms (-0.7%) | 20786 ms (-3.4%) | 12544 ms (-1.1%) | 45.8% (+0.5 pp) | 1.00 (+0.0%) | 16.2 req/s (+2.2%) |
| 40 req/s | kv_aware | 14032 ms (-2.1%) | 20352 ms (-5.4%) | 12358 ms (-2.6%) | 46.5% (+1.2 pp) | 1.01 (+1.5%) | 16.5 req/s (+4.0%) |
| 50 req/s | round_robin | 17338 ms | 25175 ms | 14981 ms | 41.8% | 1.00 | 15.6 req/s |
| 50 req/s | least_loaded | 17221 ms (-0.7%) | 24686 ms (-1.9%) | 14901 ms (-0.5%) | 42.2% (+0.3 pp) | 1.00 (+0.0%) | 15.6 req/s (-0.0%) |
| 50 req/s | kv_aware | 17194 ms (-0.8%) | 24670 ms (-2.0%) | 14813 ms (-1.1%) | 42.5% (+0.7 pp) | 1.00 (+0.0%) | 15.6 req/s (-0.4%) |

#### independent (600 requests, seed 42, 3 batched workers, max batch 16)

| Offered load | Policy | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Throughput |
|---|---|---|---|---|---|---|---|
| 10 req/s | round_robin | 135 ms | 236 ms | 10 ms | 0.0% | 1.00 | 9.7 req/s |
| 10 req/s | least_loaded | 137 ms (+1.3%) | 324 ms (+37.6%) | 22 ms (+112.8%) | 0.0% (+0.0 pp) | 1.04 (+4.5%) | 9.7 req/s (-0.2%) |
| 10 req/s | kv_aware | 137 ms (+1.8%) | 340 ms (+44.5%) | 23 ms (+125.5%) | 0.0% (+0.0 pp) | 1.04 (+4.5%) | 9.7 req/s (-0.1%) |
| 20 req/s | round_robin | 2727 ms | 5481 ms | 2506 ms | 0.0% | 1.00 | 16.4 req/s |
| 20 req/s | least_loaded | 2720 ms (-0.3%) | 5450 ms (-0.6%) | 2479 ms (-1.1%) | 0.0% (+0.0 pp) | 1.00 (+0.5%) | 16.4 req/s (+0.0%) |
| 20 req/s | kv_aware | 2750 ms (+0.8%) | 5498 ms (+0.3%) | 2506 ms (-0.0%) | 0.0% (+0.0 pp) | 1.00 (+0.5%) | 16.3 req/s (-0.3%) |
| 30 req/s | round_robin | 7398 ms | 14757 ms | 7032 ms | 0.0% | 1.00 | 16.7 req/s |
| 30 req/s | least_loaded | 7403 ms (+0.1%) | 14784 ms (+0.2%) | 7032 ms (+0.0%) | 0.0% (+0.0 pp) | 1.00 (+0.5%) | 16.7 req/s (+0.0%) |
| 30 req/s | kv_aware | 7403 ms (+0.1%) | 14784 ms (+0.2%) | 7032 ms (+0.0%) | 0.0% (+0.0 pp) | 1.00 (+0.5%) | 16.7 req/s (+0.0%) |
| 40 req/s | round_robin | 9629 ms | 19072 ms | 9099 ms | 0.0% | 1.00 | 17.1 req/s |
| 40 req/s | least_loaded | 9629 ms (+0.0%) | 19123 ms (+0.3%) | 9099 ms (-0.0%) | 0.0% (+0.0 pp) | 1.01 (+1.0%) | 17.0 req/s (-0.3%) |
| 40 req/s | kv_aware | 9629 ms (+0.0%) | 19123 ms (+0.3%) | 9099 ms (-0.0%) | 0.0% (+0.0 pp) | 1.01 (+1.0%) | 17.0 req/s (-0.3%) |
| 50 req/s | round_robin | 11018 ms | 21733 ms | 10383 ms | 0.0% | 1.00 | 17.3 req/s |
| 50 req/s | least_loaded | 11003 ms (-0.1%) | 21737 ms (+0.0%) | 10384 ms (+0.0%) | 0.0% (+0.0 pp) | 1.01 (+1.5%) | 17.2 req/s (-0.6%) |
| 50 req/s | kv_aware | 11003 ms (-0.1%) | 21737 ms (+0.0%) | 10384 ms (+0.0%) | 0.0% (+0.0 pp) | 1.01 (+1.5%) | 17.2 req/s (-0.6%) |

### Weights near the knee — hotspot_tight at 8, 10, 12, 15 req/s

The same one-at-a-time sweep as above plus two *combined* points: `beta=0 gamma=0 delta=0` is a pure-affinity router (the score is α·overlap and nothing else), and `alpha=8 beta=0 gamma=0 delta=0` is the same with a louder α.

#### hotspot_tight (600 requests, seed 42, 3 workers, batched workers (max batch 16) at 8 req/s)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 238 ms (+9.6%) | 784 ms (+47.8%) | 47 ms (+50.9%) | 24.2% (-27.5 pp) | 1.02 (-38.2%) | 13344 |
| round_robin (reference) | 235 ms (+8.2%) | 837 ms (+57.7%) | 41 ms (+32.5%) | 27.0% (-24.7 pp) | 1.00 (-39.4%) | 13376 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 217 ms | 531 ms | 31 ms | 51.7% | 1.65 | 10496 |
| alpha=0 | 230 ms (+6.2%) | 669 ms (+26.2%) | 40 ms (+27.5%) | 39.5% (-12.2 pp) | 1.02 (-38.2%) | 12768 |
| alpha=8 | 217 ms (+0.0%) | 531 ms (+0.0%) | 31 ms (+0.0%) | 51.7% (+0.0 pp) | 1.65 (+0.0%) | 10496 |
| beta=0 | 202 ms (-6.8%) | 642 ms (+21.0%) | 27 ms (-12.0%) | 52.7% (+1.0 pp) | 1.71 (+3.6%) | 9856 |
| beta=4 | 224 ms (+3.5%) | 595 ms (+12.1%) | 34 ms (+9.3%) | 44.8% (-6.8 pp) | 1.04 (-37.0%) | 12416 |
| delta=0 | 217 ms (+0.0%) | 531 ms (+0.0%) | 31 ms (+0.0%) | 51.7% (+0.0 pp) | 1.65 (+0.0%) | 10496 |
| delta=0.01 | 215 ms (-0.7%) | 459 ms (-13.4%) | 30 ms (-4.2%) | 52.7% (+1.0 pp) | 1.30 (-21.2%) | 10656 |
| alpha=8 beta=0 gamma=0 delta=0 | 27606 ms (+12635.5%) | 53225 ms (+9932.1%) | 27053 ms (+86713.4%) | 31.5% (-20.2 pp) | 3.00 (+81.8%) | 13312 |
| beta=0 gamma=0 delta=0 | 27606 ms (+12635.5%) | 53225 ms (+9932.1%) | 27053 ms (+86713.4%) | 31.5% (-20.2 pp) | 3.00 (+81.8%) | 13312 |

#### hotspot_tight (600 requests, seed 42, 3 workers, batched workers (max batch 16) at 10 req/s)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 244 ms (+11.5%) | 994 ms (+64.6%) | 85 ms (+160.2%) | 25.2% (-25.0 pp) | 1.02 (-35.5%) | 13120 |
| round_robin (reference) | 240 ms (+9.6%) | 902 ms (+49.4%) | 65 ms (+99.6%) | 28.2% (-22.0 pp) | 1.00 (-37.1%) | 13376 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 219 ms | 604 ms | 33 ms | 50.2% | 1.59 | 10144 |
| alpha=0 | 235 ms (+7.3%) | 807 ms (+33.7%) | 58 ms (+77.8%) | 41.2% (-9.0 pp) | 1.15 (-27.7%) | 12416 |
| alpha=8 | 219 ms (+0.0%) | 604 ms (+0.0%) | 33 ms (+0.0%) | 50.2% (+0.0 pp) | 1.59 (+0.0%) | 10144 |
| beta=0 | 219 ms (+0.0%) | 604 ms (+0.0%) | 33 ms (+0.0%) | 50.2% (+0.0 pp) | 1.59 (+0.0%) | 10144 |
| beta=4 | 235 ms (+7.3%) | 834 ms (+38.1%) | 66 ms (+100.7%) | 41.8% (-8.3 pp) | 1.11 (-30.2%) | 12192 |
| delta=0 | 219 ms (+0.0%) | 604 ms (+0.0%) | 33 ms (+0.0%) | 50.2% (+0.0 pp) | 1.59 (+0.0%) | 10144 |
| delta=0.01 | 219 ms (+0.0%) | 604 ms (+0.0%) | 33 ms (+0.0%) | 50.2% (+0.0 pp) | 1.59 (+0.0%) | 10144 |
| alpha=8 beta=0 gamma=0 delta=0 | 35424 ms (+16103.5%) | 68807 ms (+11294.6%) | 33141 ms (+101169.7%) | 30.2% (-20.0 pp) | 3.00 (+88.7%) | 13312 |
| beta=0 gamma=0 delta=0 | 35424 ms (+16103.5%) | 68807 ms (+11294.6%) | 33141 ms (+101169.7%) | 30.2% (-20.0 pp) | 3.00 (+88.7%) | 13312 |

#### hotspot_tight (600 requests, seed 42, 3 workers, batched workers (max batch 16) at 12 req/s)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 455 ms (+95.8%) | 1711 ms (+49.9%) | 249 ms (+177.9%) | 23.2% (-25.7 pp) | 1.05 (-24.2%) | 13024 |
| round_robin (reference) | 452 ms (+94.5%) | 1518 ms (+33.0%) | 312 ms (+248.2%) | 24.2% (-24.7 pp) | 1.00 (-27.8%) | 13376 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 232 ms | 1141 ms | 89 ms | 48.8% | 1.39 | 10400 |
| alpha=0 | 252 ms (+8.3%) | 1907 ms (+67.1%) | 154 ms (+72.3%) | 35.0% (-13.8 pp) | 1.11 (-19.5%) | 12160 |
| alpha=8 | 232 ms (+0.0%) | 1141 ms (+0.0%) | 89 ms (+0.0%) | 48.8% (+0.0 pp) | 1.39 (+0.0%) | 10400 |
| beta=0 | 232 ms (+0.0%) | 1141 ms (+0.0%) | 89 ms (+0.0%) | 48.8% (+0.0 pp) | 1.39 (+0.0%) | 10400 |
| beta=4 | 238 ms (+2.5%) | 1141 ms (+0.0%) | 97 ms (+8.1%) | 43.0% (-5.8 pp) | 1.05 (-23.8%) | 12320 |
| delta=0 | 232 ms (+0.0%) | 1141 ms (+0.0%) | 89 ms (+0.0%) | 48.8% (+0.0 pp) | 1.39 (+0.0%) | 10400 |
| delta=0.01 | 232 ms (+0.0%) | 1141 ms (+0.0%) | 89 ms (+0.0%) | 48.8% (+0.0 pp) | 1.39 (+0.0%) | 10400 |
| alpha=8 beta=0 gamma=0 delta=0 | 37509 ms (+16039.1%) | 76448 ms (+6599.3%) | 38278 ms (+42676.6%) | 31.5% (-17.3 pp) | 3.00 (+116.6%) | 13312 |
| beta=0 gamma=0 delta=0 | 37509 ms (+16039.1%) | 76448 ms (+6599.3%) | 38278 ms (+42676.6%) | 31.5% (-17.3 pp) | 3.00 (+116.6%) | 13312 |

#### hotspot_tight (600 requests, seed 42, 3 workers, batched workers (max batch 16) at 15 req/s)

| Setting | TTFT p50 | TTFT p99 | Queue mean | Hit rate | Imbalance | Evictions |
|---|---|---|---|---|---|---|
| least_loaded (reference) | 3300 ms (+713.9%) | 6742 ms (+211.4%) | 2909 ms (+690.5%) | 24.0% (-24.7 pp) | 1.03 (-13.8%) | 13088 |
| round_robin (reference) | 3341 ms (+724.0%) | 6661 ms (+207.6%) | 2864 ms (+678.2%) | 25.5% (-23.2 pp) | 1.00 (-16.3%) | 13376 |
| **default** (α=2 β=0.8 γ=0.4 δ=0.002) | 405 ms | 2165 ms | 368 ms | 48.7% | 1.20 | 11424 |
| alpha=0 | 1066 ms (+162.9%) | 2742 ms (+26.6%) | 694 ms (+88.5%) | 37.5% (-11.2 pp) | 1.18 (-1.3%) | 11712 |
| alpha=8 | 405 ms (+0.0%) | 2165 ms (+0.0%) | 368 ms (+0.0%) | 48.7% (+0.0 pp) | 1.20 (+0.0%) | 11424 |
| beta=0 | 405 ms (+0.0%) | 2165 ms (+0.0%) | 368 ms (+0.0%) | 48.7% (+0.0 pp) | 1.20 (+0.0%) | 11424 |
| beta=4 | 519 ms (+27.9%) | 2165 ms (+0.0%) | 394 ms (+7.0%) | 45.8% (-2.8 pp) | 1.14 (-5.0%) | 11712 |
| delta=0 | 405 ms (+0.0%) | 2165 ms (+0.0%) | 368 ms (+0.0%) | 48.7% (+0.0 pp) | 1.20 (+0.0%) | 11424 |
| delta=0.01 | 405 ms (+0.0%) | 2165 ms (+0.0%) | 368 ms (+0.0%) | 48.7% (+0.0 pp) | 1.20 (+0.0%) | 11424 |
| alpha=8 beta=0 gamma=0 delta=0 | 51135 ms (+12510.4%) | 101150 ms (+4571.3%) | 49151 ms (+13255.2%) | 19.7% (-29.0 pp) | 3.00 (+151.0%) | 13312 |
| beta=0 gamma=0 delta=0 | 51135 ms (+12510.4%) | 101150 ms (+4571.3%) | 49151 ms (+13255.2%) | 19.7% (-29.0 pp) | 3.00 (+151.0%) | 13312 |


### What saturation changes

- **Locality is a throughput lever once workers are busy.** Below the knee (10 req/s) `kv_aware` buys the same kind of TTFT gain as under the linear model (−9% p50, −33% p99 on `hotspot_tight`; −39% p50 on `shared_prefix_heavy`). At and above the knee the gain becomes *capacity*: +16% completed requests per second on `hotspot_tight` at 20 req/s, +18–25% on `shared_prefix_heavy` at 20–40 req/s, with TTFT p50 40–80% lower because the queues are shorter. `independent` shows no difference at any load, as it must.
- **The blended router does not invert.** The linear model's 1.60 imbalance on `hotspot_tight` appears here too (1.59–1.65 at 8–10 req/s), but only while the hot worker has headroom; as load rises the queue-depth and predicted-TTFT terms take over and imbalance falls to 1.20 at 15 req/s and ≈1.05 above 20. The default weights are self-limiting because δ prices the queue in milliseconds.
- **A pure-affinity router does invert, catastrophically.** With β=γ=δ=0 the tie-break sends the first request to `worker-0`, every prefix then lives there, and every later request follows: imbalance 3.00, TTFT p50 of 27–51 s against 217–405 ms for the default at the same loads, and a *lower* hit rate (31% vs 50%) because the single worker's 2 GiB cannot hold the working set. This is the failure hypothesis from the spec, reproduced. It needs both saturation and the absence of a load term; either one alone is survivable.
- **β=4 is no longer free.** Under the linear model it capped concentration at no TTFT cost. Here it costs 7–28% TTFT p50 and 3–8 points of hit rate at 10–15 req/s, because spreading a hot prefix across workers means each of them must prefill it cold and stall its batch. The right β depends on whether the hot worker is under or over its knee — which is what the router should be measuring, and what the predicted-TTFT term measures badly with a linear queue coefficient.
- **The linear TTFT estimator does not transfer across load levels.** Fitting it to the batched model's own output (`python -m benchmark.fit --csv benchmark/results/batched/batched16_hotspot_tight_r{10,20}_kv_aware_seed42.csv`) gives a queue coefficient of 123 ms per pending request instead of 12 and an in-sample R² of 0.91 — but fitted on 10 req/s it under-predicts 20 req/s by 4.3 s, and fitted on 20 it over-predicts 10 by 0.6 s. Queue wait in a batched engine depends on utilisation, not on a count of pending requests, so the honest fix is a feature the estimator does not have (batch occupancy, or the worker's measured iteration time), not a better coefficient.
- **Absolute numbers are still the simulator's.** The knee at ~5.5 req/s per worker comes from `decode_step_base_ms=7.5`, `decode_step_per_seq_ms=0.5`, `max_batch_size=16`, and no chunked prefill. Change any of them and the knee moves; the shape of the conclusions above should not.

## Reading the hotspot pair

`hotspot` and `hotspot_tight` share the same request stream (60% of a 40 rps stream on one 4K-token prefix). They differ only in per-worker KV capacity: 8 GiB vs 2 GiB.

| | hotspot (8 GiB/worker) | hotspot_tight (2 GiB/worker) |
| --- | --- | --- |
| kv_aware per-worker requests | 336 / 331 / 333 | 247 / 219 / **534** |
| kv_aware imbalance | 1.01 | **1.60** |
| kv_aware queue wait mean vs round_robin | −3.5% | **+17.5%** |
| kv_aware queue wait p99 vs round_robin | ≈ | **+53%** (312 vs 204 ms) |
| kv_aware TTFT p50 vs round_robin | −3.6% | −18.3% |
| kv_aware hit rate vs round_robin | 61.0% vs 60.7% | 55.2% vs 33.6% |

With ample capacity the hot prefix replicates onto every worker within a few requests, overlap becomes equal everywhere, and the load term takes over — no hotspot forms. With tight capacity the prefix can live on only one worker at a time, affinity concentrates 53% of traffic there, and queueing cost rises sharply. TTFT still improves overall because, in this model, one queued request costs 12 ms while a cold 4K prefill costs 205 ms. Under the batched model (previous section) the same concentration is self-limiting for the blended router and fatal for a pure-affinity one; measuring it on real hardware is the remaining roadmap item.
