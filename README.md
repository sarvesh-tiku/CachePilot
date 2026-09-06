# CachePilot

A local-first, KV-cache-aware LLM inference gateway for studying how request locality, worker load, and cache pressure interact in LLM serving.

**Status:** All ten spec milestones have a first version: simulated workers, OpenAI-style gateway with SSE streaming, three routing policies (round robin, least loaded, KV-aware), prefix fingerprinting with a logical KV directory, every routing decision persisted and explainable, a reproducible benchmark harness with seven workloads, an operator console that shows each decision's candidate scores and reason, observability (per-request event trail, JSON logs, Prometheus metrics), an adapter that puts any OpenAI-compatible server behind the same gateway, a live benchmark measured against a real model on this laptop, a TTFT estimator fitted to those measurements, a one-at-a-time sensitivity sweep of the four routing weights, and a second, continuous-batching worker model that saturates — under which locality-aware routing raises throughput 16–25% at the knee and a pure-affinity router collapses. Multi-worker real benchmarks need GPU hardware this project has not had yet.

**Read first:** [`docs/kv-cache-design.md`](docs/kv-cache-design.md) explains the mechanism (KV sizing, prefix identity, residency, locality-vs-load routing, saturation) and maps each design choice here onto vLLM, SGLang, Dynamo, and llm-d, including what is deliberately simplified.

## Benchmark results (simulated)

1000 requests per run, seed 42, three simulated workers, default KV-aware weights. Each policy sees the *identical* request stream. Full tables with all metrics and deltas: [`docs/benchmarking.md`](docs/benchmarking.md). These are simulator numbers — see the cost model below — not GPU measurements.

| Workload | TTFT p50: round_robin → kv_aware | TTFT p99 Δ | Cache hit rate: rr → kv | Evictions Δ | Imbalance (kv) |
| --- | --- | --- | --- | --- | --- |
| independent | 167 → 164 ms (−1.5%) | −0.2% | 0% → 0% | 0% | 1.06 |
| shared_prefix_small | 164 → 160 ms (−2.2%) | −1.6% | 23.2% → 24.8% | −1% | 1.05 |
| **shared_prefix_heavy** | **79 → 66 ms (−16.1%)** | −4.7% | **59.4% → 79.2%** | **−42%** | 1.13 |
| **bursty** | **258 → 179 ms (−30.7%)** | **−14.2%** | **39.3% → 57.7%** | −20% | 1.15 |
| cache_pressure | 168 → 155 ms (−7.7%) | −1.8% | **10.7% → 29.1%** | −13% | 1.05 |
| hotspot | 140 → 135 ms (−3.6%) | −2.0% | 60.7% → 61.0% | 0% | 1.01 |
| **hotspot_tight** | 321 → 262 ms (−18.3%) | −14.4% | 33.6% → 55.2% | −32% | **1.60**, queue wait **+17.5%** |

`least_loaded` tracks `round_robin` within ±3% on every metric except queue wait, because neither is locality-aware.

**What the numbers say, against the spec's hypotheses:**

- **Primary hypothesis holds where prefixes are large and concentrated.** On `shared_prefix_heavy` (four 4K-token prefixes, 80% of traffic) KV-aware routing lifts the hit rate by 20 points, cuts TTFT p50 by 16%, and — the part I did not predict — cuts LRU evictions by 42%, because each worker ends up specializing in a subset of prefixes instead of every worker churning through all of them. The cost is a 1.13 imbalance.
- **Bursty traffic is where locality matters most.** The first request of a burst warms one worker; routing the rest of the burst there yields −31% TTFT p50 and −14% p99, for +4.6% mean queue wait and a 1.15 imbalance. That is the tradeoff the project exists to expose.
- **Secondary hypothesis holds.** On `independent` traffic, KV-aware is indistinguishable from least-loaded (164 vs 164 ms p50): with zero overlap everywhere, the score degenerates to the load and predicted-TTFT terms.
- **A small shared fraction barely helps — and the reason is instructive.** With one shared prefix and ample capacity, *every* policy ends up with that prefix cached on all three workers within a few requests, so round robin already gets a 23% hit rate. Locality routing only pays when there are more hot prefixes than a single worker can comfortably hold.
- **Under cache pressure, locality nearly triples the hit rate** (10.7% → 29.1%) by partitioning the 40-prefix working set across workers rather than thrashing all three caches identically.
- **The failure hypothesis needs tight capacity to appear.** `hotspot` (8 GiB per worker) did *not* overload anyone (imbalance 1.01): the hot prefix replicated onto all three workers within a few requests, overlap became equal everywhere, and the load term decided. `hotspot_tight` (2 GiB per worker, room for one 4K prefix) is the real stress test: KV-aware sent **534 of 1000 requests to one worker**, mean queue wait rose 17.5% and queue-wait p99 rose 53%. TTFT p50 still improved 18%, because in this simulator a queued request costs 12 ms while a cold 4K prefill costs 205 ms. The concentration is real; whether it *hurts* depends on how steeply queueing cost grows — which is exactly what this simulator cannot tell you.
- **Throughput is unchanged across policies by construction.** The simulator's queue model penalizes latency per pending request but does not cap concurrency, so it cannot show the point where a hotspot inverts the result. That is a simulator limitation, not a finding, and the strongest argument for Milestone 9.
- **Weight sensitivity is below.** The runner accepts `--alpha … --delta`; `benchmark.sweep` varies them one at a time.

### Weight sensitivity (simulated)

`python -m benchmark.sweep --all-workloads` replays the identical seed-42 streams under 15 `kv_aware` settings: the defaults, then each weight in turn set to each of α ∈ {0, 0.5, 1, 4, 8}, β ∈ {0, 0.2, 2, 4}, γ ∈ {0, 2}, δ ∈ {0, 0.01, 0.05} with the other three at their defaults. Seed-42 highlights below; two of them shrink on other seeds, as the bullets say. Full tables: [`docs/benchmarking.md`](docs/benchmarking.md#weight-sensitivity-αβγδ-one-at-a-time).

| Workload | Setting | TTFT p50 vs default | Hit rate | Imbalance | Queue mean |
| --- | --- | --- | --- | --- | --- |
| hotspot_tight | **default** | 262 ms | 55.2% | 1.60 | 140 ms |
| hotspot_tight | **β=4** | **193 ms (−26%)** | 53.9% | **1.12** | 109 ms (−22%) |
| hotspot_tight | α=0.5 | 214 ms (−18%) | 54.1% | 1.22 | 114 ms (−19%) |
| hotspot_tight | α=0 | 275 ms (+5%) | 47.8% | 1.01 | 109 ms |
| bursty | **default** | 179 ms | 57.7% | 1.15 | 75 ms |
| bursty | β=0 | 166 ms (−7.5%) | 59.5% | 1.21 | 75 ms |
| bursty | β=4 | 226 ms (+26%) | 49.1% | 1.04 | 69 ms |
| bursty | δ=0 | 214 ms (+19%) | 54.2% | 1.21 | 75 ms |
| bursty | δ=0.01 | 173 ms (−3.3%) | 62.5% | 1.22 | 81 ms |
| shared_prefix_heavy | any α ≥ 0.5, any δ | 66 ms (±0.3%) | 78–79% | 1.10–1.13 | 42–44 ms |

- **α is a threshold, not a dial.** Every α ≥ 1 routes byte-identically to α=2 on all seven workloads and on every seed tried. Overlap is nearly binary here and the penalty terms are bounded, so once α clears their sum the cached worker always wins. Only α=0 costs anything (4–7 hit-rate points on every seed and 0–20% TTFT on `hotspot_tight`), and even it keeps most of the locality because the δ·predicted-TTFT term already prices uncached prefill.
- **β=4 caps hotspot concentration safely.** On `hotspot_tight` it holds imbalance at ≈1.11 on all three seeds, costs ≤1.3 points of hit rate, and never raises TTFT. The −26% TTFT above is seed 42's, where the default happened to concentrate at 1.60; on seeds 7 and 123 the default sits at 1.13–1.18 and β=4 saves ≈0%. On `bursty` β=4 is never better, and β=0's −7% comes with a 1.63 imbalance and +25% queue wait on one seed. No single β wins both — the α/β tradeoff the spec asks about, with numbers and error bars.
- **δ and γ effects do not survive replication.** δ=0's +19% and γ=2's −6.9% on `bursty` are seed-42 only (+1.6% and 0% elsewhere). Harmless at the defaults; not worth tuning on simulated data.
- **Four workloads are weight-insensitive** (`independent`, `hotspot` at 8 GiB, `cache_pressure`, `shared_prefix_small`: ≤2.4% TTFT p50 movement across all 15 settings). Where the policy did not matter, the weights do not either.
- **Seed variance is the first-order term.** Routing is discrete and path-dependent (on `bursty` seed 42, β = 0 / 0.2 / 0.8 / 2 / 4 gives 166 / 226 / 179 / 187 / 226 ms), and the default's own spread across seeds (bursty 166–225 ms, hotspot_tight 152–262 ms) exceeds every weight effect except α=0. The replication table is in [`docs/benchmarking.md`](docs/benchmarking.md#replication-on-two-more-seeds).

### Under saturation (simulated, continuous-batching workers)

The tables above use a worker model that cannot saturate. `--queue-model batched` swaps in a continuous-batching engine (bounded batch, KV budget, prefill stalls the batch, decode step grows with batch size; [design doc §5](docs/kv-cache-design.md#5-saturation-why-a-second-simulator-was-necessary)) and `python -m benchmark.loadscan` sweeps offered load through the knee (~16 req/s for three workers). Full tables: [`docs/benchmarking.md`](docs/benchmarking.md#under-saturation-continuous-batching-workers).

| Workload · load | Policy | TTFT p50 | TTFT p99 | Hit rate | Imbalance | Throughput |
| --- | --- | --- | --- | --- | --- | --- |
| hotspot_tight · 10 req/s | round_robin | 240 ms | 902 ms | 28% | 1.00 | 10.0 req/s |
| hotspot_tight · 10 req/s | kv_aware | **219 ms** | **604 ms** | 50% | 1.59 | 10.1 req/s |
| hotspot_tight · 10 req/s | α only (β=γ=δ=0) | **35 400 ms** | 68 800 ms | 30% | **3.00** | — |
| hotspot_tight · 20 req/s | round_robin | 8 093 ms | 15 424 ms | 29% | 1.00 | 13.1 req/s |
| hotspot_tight · 20 req/s | kv_aware | **4 812 ms (−41%)** | 8 479 ms (−45%) | 44% | 1.10 | **15.3 req/s (+16%)** |
| shared_prefix_heavy · 20 req/s | round_robin | 3 090 ms | 6 924 ms | 51% | 1.00 | 16.0 req/s |
| shared_prefix_heavy · 20 req/s | kv_aware | **598 ms (−81%)** | 3 439 ms (−50%) | 68% | 1.09 | **18.9 req/s (+18%)** |
| shared_prefix_heavy · 30 req/s | kv_aware vs rr | −22% | −46% | +16 pp | 1.04 | **+25%** |
| independent · any load | all three | ±1% | ±1% | 0% | ≈1.00 | ±1% |

- **Past the knee, locality is a throughput lever.** A cold 4K prefill stalls fifteen other sequences' decode for ~200 ms, so every hit is compute the batch gets back. That is why `kv_aware` completes 16–25% more requests per second on the shared-prefix workloads at 20–40 req/s while also cutting TTFT 40–80%.
- **The blended router is self-limiting; a pure-affinity one is not.** With the load terms removed the tie-break sends the first request to one worker, every prefix then lives there, and everything follows: imbalance 3.00 and TTFT in the tens of seconds. The default weights keep imbalance at 1.59 only while the hot worker has headroom, then fall back to ≈1.05 as queues form, because δ prices the queue in milliseconds. This is the spec's failure hypothesis, reproduced, and the condition for it: saturation *and* no load term.
- **β=4 is no longer free**: spreading a hot prefix means each worker prefills it cold, and at 10–15 req/s that costs 7–28% TTFT p50. The right β depends on which side of the knee the hot worker is on.
- **The linear TTFT estimator does not transfer across load.** Fitted to the batched model's output its queue coefficient is 123 ms per pending request, not 12, and a fit at 10 req/s under-predicts 20 req/s by 4 s. Queue wait depends on utilisation, not on a count; the estimator needs a batch-occupancy feature, not a better constant.

### Real-worker measurement (one real model, one worker)

`python -m benchmark.live` replays the same seeded workloads against a *running* gateway over HTTP and measures what a client sees: TTFT from send to first SSE content delta. Run here against **Qwen2.5-0.5B-Instruct (4-bit) served by `mlx_lm.server` on an Apple M1 Pro**, 512-token prefixes, 24 output tokens, 24 requests at 0.5 req/s, one warm-up request excluded. With a single real worker there is nothing to route *between*, so this measures whether the gateway's **estimated** prefix residency predicts real prefix-cache savings — not policy quality.

| Workload | Estimated hit rate | TTFT p50 all | TTFT p50 **hits** | TTFT p50 **misses** | TTFT p99 |
| --- | --- | --- | --- | --- | --- |
| independent (unique prefixes) | 0% | 288 ms | — | 288 ms | 923 ms |
| 4 rotating shared prefixes | 83% | 214 ms | **204 ms** | **400 ms** | 515 ms |
| 1 hot prefix | 100% | 192 ms | 192 ms | — | 489 ms |

Three honest takeaways:

- **The estimate is predictive.** In the same run, requests the gateway believed were cache hits saw half the TTFT of the ones it believed were misses (204 vs 400 ms). The server's own prompt cache and the gateway's directory agree often enough for KV-aware routing to be grounded in something real.
- **The simulator's absolute numbers are not transferable.** The TTFT estimator (simulator coefficients: 0.05 ms/token prefill, 15 ms overhead) predicted 20–40 ms for these requests; a small model on a laptop delivers 200–300 ms. Directional conclusions from the simulated tables (which policy wins, where hotspots form) survive; the milliseconds do not.
- **Fitting the estimator to these 72 requests fixes the scale.** `python -m benchmark.fit` (non-negative least squares on the per-request CSVs) gives **0.315 ms per uncached token and 198 ms fixed overhead** for this model on this laptop, cutting the estimator's mean absolute error from 246 ms to 70 ms in-sample and to 41–44 ms on held-out runs. The queue coefficient stays at the simulator's 12 ms because no request ever queued — the fitter reports that instead of inventing a number. Three env vars (`CACHEPILOT_TTFT_*`) put the fitted coefficients into routing; details and caveats in [`docs/real-workers.md`](docs/real-workers.md#fitting-the-ttft-estimator).
- **n=24 on a laptop is noisy.** p95/p99 tails (900 ms on the miss baseline) include thermal and scheduling noise unrelated to caching. Treat the medians as the signal.

A multi-worker real comparison (the spec's Milestone 10 in full) needs a machine that can host several vLLM instances with `--enable-prefix-caching`; `docker-compose.vllm.yml` and `docs/real-workers.md` describe the setup.

## Why this exists

LLM workloads are full of repeated prefixes: shared system prompts, repeated documents, multi-turn conversations, tool schemas. If requests with overlapping prefixes are routed without regard to where useful KV state already lives, prefill compute is wasted and time-to-first-token goes up.

CachePilot asks one question:

> Can KV-cache-aware routing and scheduling reduce latency and wasted compute for repeated-prefix LLM workloads compared with standard routing policies?

It is a systems research layer around inference workers, not a replacement for vLLM/SGLang/TensorRT-LLM. The point is to make routing decisions **observable and benchmarkable**.

## Architecture

```text
Clients -> POST /v1/chat/completions -> prefix fingerprint -> Scheduler (policy)
                                                                 |  round_robin
                                          KV Directory --------> |  least_loaded
                                     (per-worker prefix LRU)     |  kv_aware (next)
                                                                 v
                 Worker Registry (state) + Worker Pool (instances)
                        /            |            \
                  worker-0       worker-1       worker-2      <- SimulatedWorker (seeded)
                        \            |            /
                    RoutingDecision + RequestResult + cache admit
                                       |
                              SQL store (SQLite / Postgres)
                                       |
                 GET /api/decisions, GET /api/kv, operator console
```

Development proceeds in layers: deterministic simulation first, then real local workers, then a benchmark harness, then an operator dashboard.

### What exists today

| Layer | Path | Notes |
| --- | --- | --- |
| Domain models | `backend/cachepilot/core/models.py` | `InferenceRequest`, `WorkerState`, `KVCacheEntry`, `RoutingDecision`, `CandidateScore`, `RequestResult`, `SystemEvent` |
| Clock | `backend/cachepilot/core/clock.py` | `RealClock` for serving, `VirtualClock` for tests/benchmarks; injected into the simulator |
| Token estimate | `backend/cachepilot/core/tokens.py` | canonical prompt text + deterministic chars/4 approximation, no tokenizer |
| Prefix fingerprint | `backend/cachepilot/kv/fingerprint.py` | cumulative SHA-256 over fixed-size prefix chunks |
| KV directory | `backend/cachepilot/kv/directory.py` | per-worker prefix residency with LRU eviction, hit/miss/eviction counters |
| Worker registry | `backend/cachepilot/workers/registry.py` | register / heartbeat / mark unhealthy / list healthy |
| Simulated worker | `backend/cachepilot/workers/simulated.py` | seeded cost model; streams `TokenEvent`s then one `DoneEvent`; real queue/active accounting |
| Worker interface | `backend/cachepilot/workers/base.py` | `InferenceWorker` protocol shared by simulated and real workers |
| Real worker | `backend/cachepilot/workers/vllm.py` | adapter for any OpenAI-compatible streaming server; observed TTFT, vLLM metrics scraping |
| Schedulers | `backend/cachepilot/scheduler/` | `Scheduler` protocol, `round_robin`, `least_loaded`, `kv_aware`, shared `scoring` (candidate facts + TTFT estimate) |
| Benchmark | `benchmark/` | seeded workload generator, six workload specs, discrete-event runner, JSON/CSV export, comparison tables |
| Gateway | `backend/cachepilot/gateway/service.py` | fingerprint → schedule → persist decision → touch prefix → stream → admit prefix → persist result |
| Persistence | `backend/cachepilot/persistence/` | SQLAlchemy async; `requests`, `routing_decisions`, `request_results`, `events` |
| Telemetry | `backend/cachepilot/telemetry/` | event emitter, JSON log formatter, Prometheus collectors, metrics summary |
| API | `backend/cachepilot/api/` | chat completions, workers (+drain/restore), decisions, requests, kv, policies, events, metrics, benchmark runs |
| Operator console | `frontend/` | Vite + React + TS + Recharts; overview, requests, request detail, benchmarks |

## API

### Chat completions

```bash
curl localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'X-CachePilot-Policy: least_loaded' \
  -d '{
    "model": "sim-model",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Explain KV caching."}
    ],
    "max_tokens": 64,
    "stream": true
  }'
```

- OpenAI-shaped request and response; `stream: true` returns Server-Sent Events ending in `data: [DONE]`.
- `X-CachePilot-Policy` overrides the default policy (`CACHEPILOT_SCHEDULER_POLICY`) per request, so one workload can be replayed under each policy without a restart.
- Every response carries `X-CachePilot-Request-Id`, `X-CachePilot-Decision-Id`, `X-CachePilot-Policy`, `X-CachePilot-Worker`. Non-streaming bodies (and the first SSE chunk) include the same under a `cachepilot` key.
- Output is deterministic placeholder text from a simulated worker. Nothing here is a language model.

### Routing decisions

```bash
curl 'localhost:8000/api/decisions?policy=least_loaded&limit=20'
curl localhost:8000/api/decisions/<decision_id>      # decision + candidates + result
```

Each decision records every healthy candidate's queue depth, cache overlap, KV pressure, estimated TTFT, and the policy's score, plus a human-readable reason:

```text
least_loaded -> worker-1 | lowest load 0 (queue 0 + active 0); runner-up worker-0 at load 1
```

Prompt text is never persisted; only message count, token estimate, and (later) the prefix fingerprint.

### Workers and KV residency

```bash
curl localhost:8000/api/workers                  # WorkerState incl. kv_used_bytes
curl -X POST localhost:8000/api/workers/worker-0/drain     # stop routing to it; in-flight work finishes
curl -X POST localhost:8000/api/workers/worker-0/restore
curl localhost:8000/api/kv                       # per-worker entries / used / hits / misses / evictions
curl localhost:8000/api/kv/worker-0/entries      # resident prefix chunks, most recent first
```

### Observability

Every request leaves an append-only event trail, queryable by request or type:

```bash
curl 'localhost:8000/api/events?request_id=<id>&ascending=true'
# REQUEST_RECEIVED → REQUEST_FINGERPRINTED → ROUTING_DECISION → CACHE_HIT|CACHE_MISS
#   → FIRST_TOKEN → REQUEST_COMPLETED → CACHE_ADMITTED [→ CACHE_EVICTED]
curl 'localhost:8000/api/events?event_type=CACHE_EVICTED&limit=50'
```

Worker status transitions emit `WORKER_UNHEALTHY`, `WORKER_RECOVERED`, `WORKER_DRAINING`. The same events are written as JSON log lines (`ROUTING_DECISION`, `REQUEST_COMPLETED`, evictions and worker transitions at INFO; the rest at DEBUG), for example:

```json
{"ts": "…", "level": "INFO", "logger": "cachepilot", "event": "routing_decision", "request_id": "…", "worker": "worker-0", "policy": "kv_aware", "cache_overlap": 0.85, "queue_depth": 4, "predicted_ttft_ms": 82.3, "reason": "…"}
```

`GET /metrics` serves Prometheus text with `cachepilot_requests_total{policy,worker,cache_hit}`, `cachepilot_request_duration_ms`, `cachepilot_ttft_ms`, `cachepilot_queue_wait_ms` (histograms by policy), `cachepilot_cache_{hits,misses,evictions}_total{worker}`, and per-worker gauges for queue depth, active requests, KV utilization, tokens/s, and health. `GET /api/metrics/summary?window_s=60` returns the same picture as p50/p95/p99 JSON for the console.

## Scheduling policies

| Policy | Chooses | Ignores |
| --- | --- | --- |
| `round_robin` | next healthy worker in id order | load, cache |
| `least_loaded` | min `queue_depth + active_requests`, ties by id | cache |
| `kv_aware` | max of `α·overlap − β·load/max_load − γ·kv_pressure − δ·predicted_ttft_ms` | — |

All policies implement one interface and receive the same inputs (`InferenceRequest`, `list[WorkerState]`, `KVDirectory`), and every decision records each candidate's overlap, queue depth, KV pressure, predicted TTFT, and final score. `kv_aware` weights come from `CACHEPILOT_KV_AWARE_{ALPHA,BETA,GAMMA,DELTA}` (defaults 2.0 / 0.8 / 0.4 / 0.002) and are echoed in every reason string and at `GET /api/policies`. Load is normalized to the busiest candidate so β expresses *relative* load; absolute queueing cost enters through δ. The predicted TTFT behind δ (and behind every candidate's `estimated_ttft_ms`) is `queue_wait × pending + prefill × uncached_tokens + overhead` with coefficients from `CACHEPILOT_TTFT_{QUEUE_WAIT_MS_PER_PENDING,PREFILL_MS_PER_TOKEN,FIXED_OVERHEAD_MS}` (defaults 12 / 0.05 / 15, the simulator's cost model; refit for real hardware with `python -m benchmark.fit`). A typical reason:

```text
cache reuse on worker-0 (overlap 0.85, load 4, est TTFT 82ms) outweighed lower load on
worker-1 (overlap 0.00, load 1, est TTFT 232ms); score 1.24 (alpha=2.0, beta=0.8, gamma=0.4, delta=0.002)
```

## How KV locality is modeled

1. **Normalize** the prompt to canonical text: `"{role}:{content}\n"` per message.
2. **Chunk** it into fixed-size pieces of `kv_chunk_tokens` (default 128, i.e. 512 chars). Only *full* chunks count; a partial trailing chunk is never reusable, as with block-based KV caches.
3. **Hash cumulatively**: `h[i] = sha256(h[i-1] ‖ chunk[i])`, so two prompts share exactly the leading hashes their shared prefix covers. `prefix_fingerprint` is the last hash.
4. **Directory** keeps, per worker, an LRU-ordered set of resident chunk hashes, each costing `kv_chunk_tokens × kv_bytes_per_token` (default 512 KiB/token, Llama-2-7B fp16), bounded by the worker's `kv_capacity_bytes`.
5. **Overlap** for a (request, worker) pair = length of the leading run of resident hashes × chunk tokens ÷ prompt tokens, in `[0, 1]`. A missing middle chunk stops the run, exactly like a real prefix cache.
6. **Lifecycle:** when a request starts on a worker its matched prefix is *touched* (recency + `hit_count`); when it completes, all its chunks are *admitted* and LRU eviction runs until the worker is under capacity. Requests that start before an identical-prefix request finishes see no overlap — that is real behavior, not a bug.

This is a **logical directory**. For simulated workers it is authoritative and drives the simulated prefill cost. For real workers it is gateway-side estimated residency and is labeled as such. `kv_bytes_per_token` can also be derived from a model shape (`CACHEPILOT_KV_MODEL_PRESET=llama-3-8b` → 128 KiB/token; `cachepilot/kv/sizing.py` has the arithmetic and presets). How all of this maps onto vLLM's block hashing, SGLang's radix tree, and the event-driven routers: [`docs/kv-cache-design.md`](docs/kv-cache-design.md).

## How the simulated worker models time

```text
effective_prefill_tokens = prompt_tokens * (1 - cache_overlap)
prefill_ms   = effective_prefill_tokens * PREFILL_COST_PER_TOKEN / service_rate   # 0.05 ms/token
decode_step  = DECODE_COST_PER_TOKEN / service_rate                                # 8 ms
queue_wait   = (queue_depth + active_requests) * QUEUE_WAIT_PER_PENDING            # 12 ms each
ttft_ms      = queue_wait + prefill_ms + fixed_overhead(15 ms) + seeded_gaussian_noise(σ=5 ms)
total_ms     = ttft_ms + (output_tokens - 1) * decode_step
```

The worker actually sleeps these durations on the injected clock while streaming, so `stream: true` paces tokens realistically under `RealClock`. Reported timings come from the cost model, not from measuring wall time, so `VirtualClock` (tests) and `EventClock` (benchmarks) produce byte-identical results without waiting. These are **three deterministic simulated inference workers**, not GPUs. This *linear* model adds latency per pending request but does not cap concurrency, so it cannot express throughput saturation.

The **batched** model (`CACHEPILOT_SIMULATED_QUEUE_MODEL=batched`, `--queue-model batched` in the harness) is a continuous-batching engine on the same clock:

```text
admit      FIFO while running < MAX_BATCH_SIZE (16) and reserved tokens (prompt + max output) fit KV_BUDGET_TOKENS (64K)
iteration  = [overhead(15 ms) + uncached_prefill_tokens(new) × 0.05 ms]   # prefill stalls the batch
           + [7.5 ms + 0.5 ms × running_sequences]                         # one decode step; KV of every sequence is read
first token at the end of the admitting iteration; one token per iteration after; arrivals wait for the boundary
```

The engine runs as a task spawned on the injected clock, and consumers block on futures the clock knows about, so `EventClock` still orders everything in simulated time. Capacity with 64-token outputs is ≈5.5 req/s per worker; a cold 4K prefill stalls the batch for ≈205 ms, which is what makes cache hits worth throughput, not only latency.

## Operator console

Four screens at `http://localhost:5173`, all backed by the public API and refreshed by polling:

| Screen | Shows | API |
| --- | --- | --- |
| **Overview** | healthy workers, req/s, tok/s, cache hit rate, TTFT p50/p99, active requests; per-worker cards with queue, KV %, tokens/s, cache entries; the active policy and KV-aware weights | `GET /api/metrics/summary`, `GET /api/policies` |
| **Requests** | newest-first table (policy, worker, prompt tokens, overlap, TTFT, latency, status) with policy/worker filters, plus a **traffic generator** that sends a burst of shared-prefix requests under a chosen policy and reports how they spread across workers | `GET /api/requests`, `POST /v1/chat/completions` |
| **Request detail** | the routing reason, every candidate's overlap / queue / KV pressure / estimated TTFT / score with the winner marked, predicted-vs-actual TTFT, a queue → prefill → decode waterfall, and the request's event timeline | `GET /api/requests/{id}`, `GET /api/events` |
| **Benchmarks** | grouped runs (workload · seed · n); TTFT p50/p99, cache hit rate & prefill saved, queue wait, and worker imbalance per policy; a form that runs all three policies in-process (~1 s per 300 requests) | `GET/POST /api/benchmark-runs` |

The spec's first demo is two clicks: send a 12-request burst under `round_robin` (scatters), send it again under `kv_aware` (clusters on the warm worker), open one of the clustered requests and read the reason.

## Benchmark harness

`benchmark/` replays a seeded workload through the *same* gateway and schedulers the API uses, on an `EventClock`: a discrete-event clock that advances simulated time only when every in-flight request is parked on it, so thousands of concurrent requests interleave in exact simulated order and a 60-second workload runs in about a second. Workloads are small JSON specs (`benchmark/workloads/*.json`) materialized deterministically from a seed: prefix pool size, shared fraction, skew, prefix/question/output sizes, Poisson or bursty arrivals, optional per-worker KV capacity.

```bash
make bench WORKLOAD=bursty REQUESTS=1000 SEED=42   # all three policies + comparison table
python -m benchmark.runner --workload hotspot --policy kv_aware --requests 5000 --seed 7 --alpha 3.0
python -m benchmark.analysis benchmark/results/*.json
python -m benchmark.sweep --all-workloads                  # vary α/β/γ/δ one at a time, tabulate
python -m benchmark.loadscan --workload hotspot_tight --rates 10,20,30   # batched workers, offered load through the knee
python -m benchmark.runner --workload hotspot --policy kv_aware --queue-model batched --max-batch-size 8 --rate 12
python -m benchmark.fit --csv benchmark/results/live/*.csv # refit the TTFT estimator from observations
# against a running gateway (real clock, client-observed TTFT, real or simulated workers):
python -m benchmark.live --gateway http://localhost:8000 --workload shared_prefix_heavy \
  --policy kv_aware --requests 24 --rate 0.5 --prefix-tokens 512 --output-tokens 24
```

## Reproduce locally

One command (backend, frontend, PostgreSQL):

```bash
make dev            # docker compose up --build
```

Then:

```bash
curl localhost:8000/api/workers
open http://localhost:5173
```

Without Docker (uses SQLite at `./cachepilot.db`):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # or python3.12 -m venv
make install        # pip install -e ".[dev]" && npm ci
make backend        # uvicorn on :8000
make frontend       # vite on :5173 (proxies /api to :8000)
```

macOS + uv + Python ≥ 3.12 gotcha: uv marks `.venv` hidden, files inside inherit the flag (and on synced folders it can be re-applied), and Python 3.12 silently skips hidden `.pth` files — so the editable install "succeeds" but `import cachepilot` fails from a bare interpreter. `chflags -R nohidden .venv` clears it; every `make` target and `scripts/run_benchmark.sh` also set `PYTHONPATH` explicitly so they work either way.

## Real worker mode

```bash
export CACHEPILOT_VLLM_URL=http://localhost:8001            # vLLM, llama-server, mlx_lm.server, Ollama…
export CACHEPILOT_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct     # the name the server serves
export CACHEPILOT_SIMULATED_WORKER_COUNT=2                  # mixed cluster; 0 for real-only
make backend
# GPU host: docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build
```

The real worker appears as `backend: "vllm"` with `source: "estimated"` in `/api/kv`: the gateway tracks which prefixes it *sent* there and routes on that estimate, while queue depth and KV utilization come from the server's own metrics when it exposes them. Observed TTFT is measured on the wall clock; server-side queue wait is not observable through the OpenAI API and is reported as 0. Details and caveats: [`docs/real-workers.md`](docs/real-workers.md).

## Tests

```bash
make test           # pytest (unit + integration)
make lint           # ruff + mypy
```

Unit tests cover registry transitions, simulator determinism and queue accounting, scheduler invariants (healthy-only, even round-robin spread, deterministic tie-breaks, identical state → identical decision, KV-aware tradeoffs flipping with weights), fingerprint determinism and prefix consistency, LRU capacity and eviction order, the event clock's ordering semantics, workload generation, and store round-trips. Hypothesis property tests assert `used_bytes ≤ capacity` after arbitrary admit/touch sequences and `overlap ∈ [0, 1]`. Integration tests boot the app against in-memory SQLite with a `VirtualClock` and exercise every endpoint end to end, including the spec's "9 requests over 3 workers" round-robin check and KV-aware clustering a shared prefix that round robin scatters. Benchmark regression tests run a tiny workload under each policy, assert seed reproducibility, and check that simulated wall time agrees with the cost model to the microsecond — they deliberately do not assert that KV-aware "wins". The estimator fitter is tested on synthetic data (exact recovery, noise, non-negativity, unidentifiable features held) and end to end: fitting it to the simulator's own benchmark output recovers the simulator's coefficients. The sweep's default point is asserted to reproduce a plain `kv_aware` run byte for byte. The batched worker's tests work out TTFT and latency by hand for single requests, batches at the cap, KV-budget head-of-line blocking, and a burst that overflows the batch; the event clock's `wait`/`spawn` are tested for time-ordering across a producer task and for deadlock detection.

## Roadmap

1. ~~Bootstrap~~
2. ~~Worker simulator + registry~~
3. ~~Gateway: `POST /v1/chat/completions` with SSE streaming~~
4. ~~Baselines: round robin, least loaded, persisted routing decisions~~
5. ~~Prefix fingerprinting + logical KV directory with LRU eviction~~
6. ~~KV-aware scheduler with configurable weights and human-readable reasons~~
7. ~~Benchmark harness (independent, shared-prefix, bursty, cache-pressure, hotspot workloads)~~
8. ~~Operator dashboard: overview, requests, request detail, benchmark comparison~~
9. ~~Structured events + metrics~~
10. ~~Real local worker adapter~~
11. ~~Live benchmark against one real worker~~
12. ~~Fit the TTFT estimator from observed data~~
13. ~~α/β/γ/δ sensitivity sweep~~
14. ~~Continuous-batching worker model; load scan through saturation; failure hypothesis reproduced~~
15. Multi-worker real comparison on GPU hardware (needs several vLLM instances with prefix caching); then refit the queue coefficient — with a batch-occupancy feature — from multi-worker data

## Design tradeoffs so far

- **Timings come from the model, not the wall clock.** This keeps results reproducible and lets tests fast-forward, at the cost of not capturing gateway overhead. Real workers report observed timings instead.
- **Two worker models, not one richer one.** The linear model is the development and regression baseline: trivially explainable, and every routing test's expected numbers can be worked out by hand. The batched model exists only to answer the saturation question, and is kept to the few mechanisms that question needs (bounded batch, KV reservation, prefill stalls, batch-size-dependent decode). Chunked prefill, preemption, and a shared prefix/active memory pool are documented omissions, not oversights.
- **TTFT estimates default to the simulator's coefficients and are refit, not learned online.** A linear model with three coefficients is easy to fit, easy to read on the request-detail screen, and easy to be wrong about in a visible way; the fitter refuses to fit a coefficient whose feature never varied rather than silently extrapolating. Online adaptation would hide that.
- **Schema via `create_all`, no migrations yet.** Fine while tables are additive; Alembic when they stop being.
- **Worker instances and worker state are separate.** `WorkerRegistry` holds `WorkerState` (what schedulers see); the gateway holds `InferenceWorker` instances. Simulated workers push queue changes immediately; KV usage is written by the gateway from the directory after each admit.
- **Fixed-size chunks, not the spec's doubling sizes (512/1024/2048/4096).** Uniform chunks give uniform overlap granularity and map directly onto block-based caches; the cost is more hashes per long prompt, which is negligible.
- **Overlap is read before the request starts and cache admission happens after it finishes.** Concurrent same-prefix requests therefore do not "see" each other's cache — the honest behavior, and the reason bursty workloads are interesting.

- **Benchmarks bypass HTTP and SQL.** The runner drives `InferenceGateway` directly with an in-memory store so the only blocking primitive is the event clock. Gateway overhead is therefore not measured; the routing logic is byte-for-byte the same.

## Limitations

- Every multi-worker number is simulated with the cost model above; the only real measurements are single-worker and on a laptop.
- The TTFT estimator has been fitted once, to 72 laptop samples with one prompt size and no queueing: the prefill slope is a line through two points and the queue coefficient is still the simulator's. Routing still defaults to the simulator's coefficients unless the `CACHEPILOT_TTFT_*` variables are set.
- The batched model's knee (~5.5 req/s per worker) is set by four coefficients and no chunked prefill; the *shape* of the saturation results is the claim, the numbers are not.
- The batched model reserves KV for running sequences from a budget separate from the prefix cache; in a real engine they share one pool and decode-time growth evicts prefixes. The KV directory also ignores block fragmentation.
- The weight sweep is one-at-a-time; interactions between weights are not characterized, and only two workloads were replicated across seeds. Seed variance in this simulator is larger than most weight effects, so tuning weights on simulated data is not recommended.
- `VirtualClock` is not an event-ordered simulation; concurrent virtual-time requests do not interleave in simulated order.
- Every published number is reproducible from a committed JSON, but the simulated ones are only as real as the cost model above.

## License

MIT
