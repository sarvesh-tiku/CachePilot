# KV-cache optimisation: what this project models, and why

This is the background document for CachePilot. It explains the mechanism the whole project is built around — reuse of transformer key/value state across requests — and maps every design choice in the code onto what production engines (vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, llm-d) actually do, including where this project deliberately simplifies. Results are in [`benchmarking.md`](benchmarking.md); the real-server path is in [`real-workers.md`](real-workers.md).

## 1. The problem in numbers

A decoder-only transformer computes, for every token and every layer, a key and a value vector per attention head. Because attention is causal, token *t*'s K and V depend only on tokens ≤ *t*. Two consequences drive everything here:

1. **Decoding needs the K/V of every previous token, every step.** Keeping them (the *KV cache*) turns an O(n²) recomputation per generated token into O(n) reads. This is not an optimisation you can skip; it is how autoregressive serving works at all.
2. **A prefix's KV is identical across requests that share the prefix.** The system prompt, the tool schema, the document, the earlier turns of a conversation — if the leading tokens match exactly, their KV can be reused instead of recomputed.

How much memory is at stake:

```text
bytes_per_token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
```

| Model | Layers × KV heads × head_dim | KV / token (fp16) | 4K-token prompt | Notes |
| --- | --- | --- | --- | --- |
| Llama-2-7B | 32 × 32 × 128 | 512 KiB | 2 GiB | multi-head attention; the project's default |
| Llama-3-8B | 32 × 8 × 128 | 128 KiB | 512 MiB | grouped-query attention: 8 KV heads serve 32 query heads |
| Mistral-7B | 32 × 8 × 128 | 128 KiB | 512 MiB | GQA + sliding-window attention (see §6) |
| Qwen2.5-0.5B | 24 × 2 × 64 | 12 KiB | 48 MiB | the model behind the live measurements |
| Llama-3-70B | 80 × 8 × 128 | 320 KiB | 1.25 GiB | |
| DeepSeek-V3 | 61 × (512 + 64 latent) | ≈ 69 KiB | 275 MiB | multi-head latent attention stores a compressed latent, not K and V |

`cachepilot.kv.sizing` has these as presets (`CACHEPILOT_KV_MODEL_PRESET=llama-3-8b`), and the arithmetic explains two facts the benchmarks depend on: on an 80 GB GPU running a 7B fp16 model (14 GB of weights) the KV pool is tens of GB, i.e. tens of thousands of tokens — *a few dozen* 4K prompts, not thousands; and GQA/MLA models cache 4–40× more prompts in the same memory, which is why cache pressure workloads are parameterised by bytes-per-token rather than hard-coded.

**Why TTFT is the metric.** A request's latency splits into *prefill* (one forward pass over all prompt tokens, compute-bound, producing the first token) and *decode* (one forward pass per output token, memory-bandwidth-bound, reading the whole KV cache each step). Time-to-first-token is essentially queueing + prefill. Prefix reuse removes the prefill of the matched tokens, so its benefit shows up in TTFT and in freed compute; it does nothing for decode speed. The simulator's cost model (`0.05 ms/token` prefill, `8 ms` decode step) has that shape by construction; the fitted real numbers on a laptop (`0.315 ms/token`, `198 ms` overhead) have it too.

## 2. Finding reusable KV: prefix identity

Reuse is only valid for an *exact leading match*: because of causality, the KV at position *t* is a function of tokens 0…*t*, so one differing token anywhere in the prefix invalidates everything after it. Every real system therefore keys the cache on the token sequence itself, in one of two shapes:

- **Hashed blocks** (vLLM automatic prefix caching, TensorRT-LLM block reuse). KV lives in fixed-size blocks (16 tokens in vLLM by default). A block's identity is `hash(parent_block_hash, token_ids_in_block)`, so the hash of block *i* commits to the entire prefix up to it. Lookup is one dictionary probe per block, walking from the first block until the first miss.
- **Radix tree** (SGLang RadixAttention). Token sequences are inserted into a radix tree whose edges are token runs and whose nodes point at KV blocks; matching a new prompt is a longest-prefix walk. Same semantics, but it also gives the scheduler an ordering: SGLang sorts waiting requests by matched-prefix length so hits are served before the blocks are evicted.

CachePilot does the block-hash version at the gateway, on the *canonical text* of the messages rather than on token ids: chunks of 128 tokens' worth of characters, `h[i] = sha256(h[i-1] ‖ chunk[i])`. The choices and their reasons:

- **Cumulative hashing** so that chunk *i*'s hash is unique to its entire prefix. Two prompts share exactly the leading run of hashes their common prefix covers, and a missing hash anywhere stops the run — which is what a block-based cache would do.
- **Only full chunks count.** A partial trailing chunk can never be reused by anyone, exactly as a partially-filled last block cannot be shared in vLLM. The waste is bounded by one chunk per prompt.
- **Fixed-size chunks, not the doubling sizes the original spec suggested**, so overlap granularity is uniform and the directory maps one-to-one onto a paged allocator's block table. The cost is more hashes per long prompt, which is negligible next to a forward pass.
- **Text rather than tokens** is the simplification. Real engines hash token ids because the model sees tokens; the same text can tokenise differently at a chunk boundary depending on what follows it (BPE merges across the cut). At the gateway the tokenizer may not be available, and the cost of being wrong is a false *hit* estimate on a boundary chunk, which shows up as a worse-than-predicted TTFT, never as a wrong answer. The chat template matters for the same reason: a real engine's prefix includes the template tokens around each message, so two clients with different templates never share KV even with identical text. The canonical `role:content\n` form here is the gateway's stand-in for that.

`cache_overlap` for a (request, worker) pair is the matched leading run × chunk tokens ÷ prompt tokens, in [0, 1]. That is the number vLLM reports as "prefix cache hit tokens" and the number the router needs.

## 3. Residency and eviction: what is actually on the worker

A prefix cache is not a separate store. In vLLM it *is* the free-block pool: blocks released by finished sequences keep their hash and content until the allocator needs them, and a new request that hashes to a still-resident block takes it back. This has three consequences that the directory models, approximates, or deliberately drops:

| Reality | In CachePilot |
| --- | --- |
| Blocks held by running sequences are pinned (reference-counted) and cannot be evicted. | Not modelled: the directory's LRU ignores in-flight use. Admission happens only when a request *completes*, so an in-flight prefix is never in the directory to be evicted in the first place. |
| Free blocks are evicted LRU when a running sequence needs memory for its next token — decode-time growth evicts prefixes. | The directory has a fixed byte budget per worker and evicts LRU on *admission* only. The batched worker reserves prompt + max output tokens for running sequences (§5), but from a separate budget. |
| Eviction is per block; a long prefix can be half-resident. | Same: chunks are evicted individually, and overlap stops at the first missing chunk. |
| Hit counts and recency exist per block. | Same (`hit_count`, `last_access`, touched when a request starts on the worker). |

The "admit on completion, not on start" rule is worth dwelling on because it is the most common surprise in the benchmark results. Two requests with the same new prefix that arrive 5 ms apart both miss: the first has not finished, so its KV is not yet admitted. Real engines behave the same way (vLLM cannot hand a block to request B while request A is still computing it) unless the scheduler explicitly batches the two so that A's prefill fills the blocks B then reads — SGLang's radix-sorted scheduling does something like this within one engine. Across *workers* nothing can help: the second request either waits or recomputes. This is why the `bursty` workload is where locality-aware routing gains most and also why its results are noisiest.

**Alternatives to LRU** that a cache-aware router could exploit: LFU or frequency-weighted recency (protects a hot system prompt from a scan of one-off documents), cost-aware eviction (evict what is cheapest to recompute per byte — short prefixes first), and admission control (do not admit a prefix seen once). The directory keeps LRU because it is what the engines it stands in for use, so the router's estimate of "what is resident" stays honest.

## 4. Routing on locality vs load

A gateway in front of *N* workers has one lever the engine does not: which worker sees the request. Sending same-prefix requests to the same worker is the whole idea; the failure mode is obvious — a hot prefix concentrates load. Every production router is a way of blending those two forces:

- **SGLang router (cache-aware load balancing).** The router keeps an *approximate* radix tree per worker of what it has sent there. If the worker with the longest prefix match has load within a threshold ratio of the least-loaded worker, route for cache; otherwise route to the shortest queue.
- **vLLM production stack.** Prefix-aware routing by hashing the prompt and remembering which endpoint saw which hash, falling back to load-based choice.
- **NVIDIA Dynamo KV router.** Workers publish KV block *events* (block stored/evicted), so the router has true residency rather than a guess, and scores each worker by matched blocks against its active load with a tunable cost function.
- **llm-d / Gateway API Inference Extension.** A scheduler plug-in chain: a prefix-cache scorer (either gateway-estimated or fed by KV events), a queue-depth scorer, a KV-utilisation scorer, weighted and summed.

CachePilot's `kv_aware` is the weighted-sum family:

```text
score = α·overlap − β·load/max_load − γ·kv_pressure − δ·predicted_ttft_ms
```

Each term is there for a specific reason. `overlap` is the locality signal. `load/max_load` is *relative* load, so β cannot see the difference between "everyone idle" and "everyone swamped"; that is deliberate, and it is what δ is for: `predicted_ttft` is queue wait + uncached prefill + overhead, an absolute cost in milliseconds that already prices *both* the queue and the missing prefix — which is why the sweep found that even α=0 keeps most of the locality benefit. `kv_pressure` is the fraction of the worker's KV budget in use: a worker at 95% will evict whatever it admits next, so a hit there is worth less than a hit on an empty worker. Every candidate's four inputs and score are persisted with the decision so any choice can be reproduced and explained (`reason` strings, `/api/decisions`).

The directory the score reads is **gateway-estimated residency**, the SGLang-router style: "we sent this prefix here recently and the worker probably still has it". It drifts from truth whenever the worker evicts on its own schedule. The event-driven designs (Dynamo, llm-d with KV events) remove that drift at the cost of a worker-to-router event stream; the real-worker adapter here closes the loop the cheap way, by comparing predicted with observed TTFT per request so the drift is at least *visible*.

**What the measurements say about the blend** (details in `benchmarking.md`):

- Where prefixes are large and concentrated, locality routing raises hit rate by ~20 points and cuts TTFT p50 by 16–31% against round robin, and — the part that is easy to miss — cuts *evictions* by 20–42%, because workers specialise in subsets of prefixes instead of all churning through all of them.
- α behaves as a threshold, not a dial: once it exceeds the bounded penalty terms the cached worker always wins, and further increases change nothing. β is the real lever, and no single β wins both bursty traffic and tight hotspots. Seed-to-seed variance is larger than most weight effects.
- Under a saturating worker model (§5), locality becomes a **throughput** lever: prefill compute is what stalls a continuous-batching engine, so a 15-point hit-rate gain is worth 16–25% more requests per second at the knee — and the load terms keep the blended router from over-concentrating, which a pure-affinity router does not manage.

## 5. Saturation: why a second simulator was necessary

The first simulator charges a fixed 12 ms per pending request and lets every request proceed concurrently. That makes routing results perfectly reproducible and lets the KV logic be developed without a GPU, but it cannot express the thing that makes hotspots dangerous: an engine has finite capacity, and past it latency grows without bound.

The batched worker (`workers/batched.py`) models a continuous-batching engine the way vLLM's scheduler behaves in its default prefill-priority mode:

- a FIFO waiting queue and a running batch bounded by `max_batch_size` and by a KV budget in tokens (each admitted sequence reserves prompt + max-output tokens, the way a paged allocator must be able to hold them; the head of the queue blocks until memory frees, so long prompts reduce concurrency);
- one engine iteration = prefill of the newly admitted sequences (uncached tokens × cost, plus fixed overhead), then one decode step for the whole batch whose cost is `base + per_seq × batch_size` — every running sequence's KV is read on every step, so batching raises throughput but lengthens each sequence's step;
- a request's first token comes from its prefill; every later token costs one iteration; a request that arrives mid-iteration waits for the boundary.

Two things follow immediately and both showed up in the load scan. First, a cold 4K-token prefill (≈205 ms) stalls fifteen other sequences' decode for that long, so cache hits raise the *capacity* of the worker, not only the latency of the hit. Second, once a worker is past its knee, queue wait dominates TTFT and grows with offered load, so the routing question becomes "which worker has headroom" — which is exactly what the load and predicted-TTFT terms measure, and why the blended router's imbalance collapses to ≈1.05 at high load while its throughput advantage stays.

What it still does not model, in decreasing order of importance for routing: chunked prefill (modern engines cap the tokens per iteration, so a long prefill is spread over several steps and decodes are not stalled for 200 ms — this shrinks the throughput effect of misses but does not remove it), preemption and recompute when KV runs out mid-decode, and the fact that the prefix cache and the running sequences share one memory pool.

## 6. Techniques beyond this project, and how each would change the router

- **Chunked prefill** bounds per-iteration latency; the prefill-stall term in §5 becomes a token budget shared with decode. Routing calculus unchanged; magnitudes shrink.
- **Prefill/decode disaggregation** splits the request into a prefill on one pool and decode on another, shipping KV between them (Mooncake, Dynamo, DistServe lineage). The router then makes *two* decisions: prefill placement by prefix locality, decode placement by free KV memory and batch headroom. The score in §4 splits accordingly, and "overlap" is only a prefill-side term.
- **KV offloading and tiered storage** (vLLM CPU offload, LMCache, engine-side disk/remote KV) turns a miss on the GPU into a *load* from a slower tier rather than a recompute. Overlap stops being binary per worker: a prefix can be resident on GPU (free), in host memory (cheap), or on another node (transfer cost). The directory's `estimated_bytes` per chunk would become a per-tier cost, and `predicted_ttft` would price transfer time against recompute time — the same trade a CPU cache hierarchy makes.
- **KV transfer between workers** (NIXL, RDMA) removes the "cold worker must recompute" assumption that makes hotspots inevitable: the router could pick the worker with headroom and *pull* the prefix. Transfer of a 4K-token Llama-2-7B prefix is 2 GiB; at 100 Gb/s that is ~170 ms, comparable to recomputing it. For GQA/MLA models it is 4–40× less, which is one reason those architectures make disaggregated serving practical.
- **Quantised KV (fp8)** halves bytes per token; capacity doubles, the eviction workloads in this project become half as tight, nothing else changes.
- **Sliding-window attention** (Mistral, Gemma) keeps only the last *W* tokens' KV per layer, so a prefix longer than *W* is not fully reusable and "overlap" has to be defined per layer type. Hybrid models (SWA + global layers) are the norm now, so a router that assumes full-prefix reuse over-estimates hits on them.
- **Multi-turn conversations** are the dominant real source of shared prefixes: turn *n*'s prompt is turns 1…*n−1* plus the new message, so session affinity and prefix affinity coincide, and the cache is refreshed every turn. This is also where decode-time KV growth matters most: the reusable prefix grows by the previous answer each turn.
- **API-level prompt caching** (Anthropic, OpenAI) is the same mechanism exposed as a billing primitive: cached prefix tokens are cheaper because the provider skips their prefill. Everything in §2 about exact leading matches applies verbatim to writing prompts that hit it.
- **Speculative decoding** changes decode throughput, not KV size or prefix reuse; it is orthogonal to routing.

## 7. Measurement discipline

Three labels run through the API and the docs, because conflating them is the easiest way to publish a number that means nothing:

- **simulated** — a timing produced by a cost model. Reproducible to the microsecond, and only as true as the model. Every multi-worker number in this repo is simulated.
- **estimated** — the gateway's belief about a real worker's residency or load. Drives routing; can be stale.
- **observed** — wall-clock measurements against a real model (24 requests per workload on a laptop here). The only numbers that can validate or refute the other two.

The rule the project follows is that a claim is stated at the weakest of the labels it rests on: "KV-aware routing cuts TTFT 16%" is a simulated claim; "the gateway's residency estimate predicts real prefix-cache savings" is an observed one (204 vs 400 ms hit vs miss on the laptop); "the TTFT estimator's queue coefficient is 12 ms" is a simulated number that the batched model already shows to be load-dependent (fitted to batched output it comes out at 123 ms and does not transfer across load levels), and the real one has not been measured because nothing has queued on a real worker yet. The roadmap item that closes that gap is a multi-worker GPU run, and the harness is ready for it.
