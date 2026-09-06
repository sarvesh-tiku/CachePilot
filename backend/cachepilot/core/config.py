from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CACHEPILOT_", env_file=".env", extra="ignore")

    app_name: str = "CachePilot"
    environment: str = "local"
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///./cachepilot.db"
    simulated_worker_count: int = 3
    # "linear" (fixed penalty per pending request) or "batched" (continuous batching that
    # saturates; see docs/kv-cache-design.md).
    simulated_queue_model: str = "linear"
    simulated_max_batch_size: int = 16
    worker_heartbeat_interval_s: float = 1.0
    scheduler_policy: str = "round_robin"
    max_tokens_limit: int = 4096
    kv_aware_alpha: float = 2.0
    kv_aware_beta: float = 0.8
    kv_aware_gamma: float = 0.4
    kv_aware_delta: float = 0.002
    # TTFT estimator used by every policy's candidate scores and by kv_aware's delta term.
    # Defaults are the simulator's cost model; refit for real workers: `python -m benchmark.fit`.
    ttft_queue_wait_ms_per_pending: float = 12.0
    ttft_prefill_ms_per_token: float = 0.05
    ttft_fixed_overhead_ms: float = 15.0
    benchmark_results_dir: str | None = None
    # Optional real worker: any OpenAI-compatible streaming server (vLLM, SGLang, llama.cpp, ...).
    vllm_url: str | None = None
    vllm_model: str | None = None
    vllm_worker_id: str = "vllm-0"
    vllm_api_key: str | None = None
    vllm_kv_capacity_bytes: int = 8 * 1024 * 1024 * 1024
    vllm_timeout_s: float = 120.0
    kv_chunk_tokens: int = 128
    # Llama-2-7B fp16: 32 layers * 2 (K,V) * 4096 hidden * 2 bytes = 512 KiB per token.
    kv_bytes_per_token: int = 524_288
    # Optional: derive kv_bytes_per_token from a model shape in cachepilot.kv.sizing.PRESETS
    # (e.g. "llama-3-8b" -> 128 KiB/token thanks to grouped-query attention).
    kv_model_preset: str | None = None

    def effective_kv_bytes_per_token(self) -> int:
        if self.kv_model_preset:
            from cachepilot.kv.sizing import bytes_per_token

            return bytes_per_token(self.kv_model_preset)
        return self.kv_bytes_per_token


def get_settings() -> Settings:
    return Settings()
