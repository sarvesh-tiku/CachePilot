# Real workers

CachePilot's simulated workers exist so the routing logic can be developed and benchmarked without GPUs. A real worker plugs into the *same* gateway through the `InferenceWorker` protocol; only the timing source changes.

## What `VLLMWorker` is

`backend/cachepilot/workers/vllm.py` is an adapter for **any OpenAI-compatible streaming server**: vLLM, SGLang, llama.cpp's `llama-server`, Ollama, `mlx_lm.server`, LM Studio. It

- POSTs `/v1/chat/completions` with `stream: true` and `stream_options.include_usage`, and yields each content delta as a `TokenEvent`;
- measures **TTFT** as the wall-clock arrival of the first content delta and **total latency** as the end of the stream, from the gateway's side;
- reports `queue_wait_ms = 0` because the OpenAI API does not expose server-side queueing;
- always sends the **served model name** it was configured with (clients may say `sim-model`);
- always sends the **full prompt** — the server does its own prefix caching;
- health-checks `GET /health` (falling back to `GET /v1/models`) and, if the server exposes vLLM's Prometheus endpoint, scrapes `vllm:num_requests_running`, `vllm:num_requests_waiting`, and `vllm:gpu_cache_usage_perc` so queue depth and KV utilization come from the backend rather than from guesses.

## Simulated vs. estimated vs. observed

Per spec §28, the API distinguishes three kinds of KV state:

| Source | Where | Meaning |
| --- | --- | --- |
| **simulated** | `GET /api/kv` → `source: "simulated"` | Authoritative. The directory *is* the simulated worker's KV memory. |
| **estimated** | `GET /api/kv` → `source: "estimated"` | Gateway-side residency guess for a real server: "we sent this prefix here recently and it probably still has it". Drives routing; may be wrong after the server evicts. |
| **observed** | `GET /api/workers` → `kv_used_bytes`, `queue_depth`, `active_requests` for `backend: "vllm"` | Scraped from the server's own metrics when available; overrides the estimate for load and KV utilization. |

Routing for a real worker uses the *estimated* overlap. Whether the estimate is right shows up as observed TTFT: a true hit skips prefill on the server, a stale estimate does not. Comparing predicted TTFT (`candidates[].estimated_ttft_ms`) with observed TTFT (`result.ttft_ms`) on the request-detail screen is the honest measurement of how good the estimate is.

## Running with a real server

Set two variables and start the backend as usual; a mixed cluster (simulated + real) is the default so routing still has choices:

```bash
export CACHEPILOT_VLLM_URL=http://localhost:8001
export CACHEPILOT_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct   # the name the server serves
export CACHEPILOT_SIMULATED_WORKER_COUNT=2                # or 0 for a pure real-worker gateway
make backend
```

Optional: `CACHEPILOT_VLLM_WORKER_ID` (default `vllm-0`), `CACHEPILOT_VLLM_API_KEY`, `CACHEPILOT_VLLM_KV_CAPACITY_BYTES` (the capacity the gateway assumes for the estimated directory; default 8 GiB), `CACHEPILOT_VLLM_TIMEOUT_S`.

### vLLM (NVIDIA GPU)

```bash
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up --build
# or, natively:
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching --port 8001
```

`--enable-prefix-caching` is what makes the KV-aware policy meaningful against a real server.

### Apple silicon / CPU (no vLLM)

Any OpenAI-compatible server works for exercising the flow; only vLLM exposes the extra metrics.

```bash
uvx --from mlx-lm mlx_lm.server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8080
# or: llama-server -m model.gguf --port 8080 ; or: ollama serve (port 11434)
export CACHEPILOT_VLLM_URL=http://localhost:8080 CACHEPILOT_VLLM_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

## Fitting the TTFT estimator

Every policy's candidate scores — and `kv_aware`'s δ term — use one linear estimator:

```text
predicted_ttft_ms = queue_wait_ms_per_pending × (queue_depth + active_requests)
                  + prefill_ms_per_token      × prompt_tokens × (1 − cache_overlap)
                  + fixed_overhead_ms
```

Its defaults are the simulator's cost model (12 ms / 0.05 ms / 15 ms) and are wrong for any real server. `python -m benchmark.fit` refits the three coefficients by non-negative least squares from observed timings, taken either from per-request CSVs written by `benchmark.runner` / `benchmark.live` or straight from a running gateway's request history (`GET /api/requests`, which persists the queue depth the scheduler saw for the worker it chose):

```bash
python -m benchmark.fit --csv benchmark/results/live/live_*_seed11.csv --write fit.json
python -m benchmark.fit --gateway http://localhost:8000 --worker mlx-0 --limit 1000
```

The tool reports current-vs-fitted MAE, bias, and R², a leave-one-source-out check when it has more than one file, and the three environment variables to export:

```bash
export CACHEPILOT_TTFT_QUEUE_WAIT_MS_PER_PENDING=12
export CACHEPILOT_TTFT_PREFILL_MS_PER_TOKEN=0.315
export CACHEPILOT_TTFT_FIXED_OVERHEAD_MS=198
```

`GET /api/policies` echoes the coefficients in use, and the overview screen prints the formula.

A coefficient can only be fitted if its feature varies in the data. When it does not — a single worker that never queued has `pending = 0` throughout — the fitter keeps that coefficient at its current value and says so, rather than reporting a meaningless number.

### Fitted to the laptop runs

Fitting to the 72 requests behind the README's live table (Qwen2.5-0.5B-Instruct 4-bit under `mlx_lm.server`, Apple M1 Pro) gives, from `benchmark/results/live/ttft_fit_mlx_qwen2.5-0.5b_m1pro.json`:

| Coefficient | Simulator default | Fitted (M1 Pro, 0.5B) | Identifiable? |
| --- | --- | --- | --- |
| queue wait per pending request | 12 ms | 12 ms (kept) | no — pending was 0 for all 72 samples |
| prefill per uncached token | 0.05 ms | **0.315 ms** | two distinct sizes only (549 and 37 tokens) |
| fixed overhead | 15 ms | **198 ms** | yes |

| | Simulator default | Fitted |
| --- | --- | --- |
| MAE on the 72 samples | 246 ms | **70 ms** |
| Bias (predicted − observed) | −246 ms | 0 ms |
| Leave-one-run-out MAE | — | hotspot 41 ms · shared_prefix_heavy 44 ms · independent 135 ms |

The fit is honest about what it is: two clusters of prompt size, so the prefill slope is a line through two points, and no queueing, so the queue coefficient is still the simulator's guess. In-sample R² is 0.31 because most within-cluster variance is scheduling noise the model has no feature for (the `independent` run's two 900 ms tails are what push its hold-out MAE to 135 ms). It should be refitted for every model + hardware pair, and refitted again with multi-worker data before the queue coefficient is trusted.

## Caveats

- Numbers from a real worker and from simulated workers in the same cluster are not comparable; mix them to exercise routing, not to benchmark.
- Prefix-cache hits on the server require its own prefix caching to be enabled and the prefix to still be resident; the gateway cannot verify this except through observed TTFT.
- The `hotspot`/`cache_pressure` conclusions in `docs/benchmarking.md` are about the simulator's cost model and must be re-measured against a real server (Milestone 10) before being believed.
