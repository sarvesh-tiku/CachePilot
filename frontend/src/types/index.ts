export type WorkerStatus = "healthy" | "unhealthy" | "draining";

export interface WorkerState {
  worker_id: string;
  model: string;
  status: WorkerStatus;
  queue_depth: number;
  active_requests: number;
  kv_capacity_bytes: number;
  kv_used_bytes: number;
  tokens_per_second_estimate: number;
  last_heartbeat: string;
}

export interface Percentiles {
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

export interface WorkerLoad {
  worker_id: string;
  backend: string;
  status: WorkerStatus;
  queue_depth: number;
  active_requests: number;
  kv_utilization: number;
  tokens_per_second_estimate: number;
  cache_entries: number;
  completed_in_window: number;
}

export interface MetricsSummary {
  window_s: number;
  completed_requests: number;
  requests_per_s: number;
  output_tokens_per_s: number;
  ttft_ms: Percentiles;
  total_latency_ms: Percentiles;
  queue_wait_ms: Percentiles;
  cache_hit_rate: number;
  mean_cache_overlap: number;
  healthy_workers: number;
  total_workers: number;
  active_requests: number;
  queued_requests: number;
  workers: WorkerLoad[];
}

export interface CandidateScore {
  worker_id: string;
  cache_overlap: number;
  queue_depth: number;
  active_requests?: number;
  estimated_ttft_ms: number;
  kv_pressure: number;
  final_score: number;
}

export interface RoutingDecision {
  decision_id: string;
  request_id: string;
  policy: string;
  selected_worker_id: string;
  candidates: CandidateScore[];
  reason: string;
  created_at: string;
}

export interface RequestResult {
  request_id: string;
  worker_id: string;
  queue_wait_ms: number;
  ttft_ms: number;
  total_latency_ms: number;
  output_tokens: number;
  cache_hit: boolean;
  cache_overlap: number;
  completed_at: string;
}

export interface RequestRecord {
  request_id: string;
  model: string;
  message_count: number;
  prompt_tokens_estimate: number;
  max_tokens: number;
  latency_slo_ms: number | null;
  prefix_fingerprint: string | null;
  created_at: string;
}

export interface RequestView {
  request: RequestRecord;
  decision: RoutingDecision | null;
  result: RequestResult | null;
  status: "received" | "in_flight" | "completed";
}

export interface PolicyInfo {
  name: string;
  default: boolean;
  parameters: Record<string, number>;
  estimator?: Record<string, number>;
}

export interface CacheSummary {
  worker_id: string;
  source: "simulated" | "estimated";
  capacity_bytes: number;
  used_bytes: number;
  entries: number;
  hits: number;
  misses: number;
  evictions: number;
}

export interface WorkloadSpec {
  name: string;
  description: string;
  prefix_tokens: number;
  question_tokens: number;
  output_tokens: number;
  shared_prefix_count: number;
  shared_fraction: number;
  shared_skew: number;
  arrival: { kind: "poisson" | "bursty"; rate_rps: number; burst_size: number };
  kv_capacity_bytes_per_worker: number | null;
}

export interface BenchmarkMetrics {
  requests: number;
  completed: number;
  makespan_ms: number;
  requests_per_s: number;
  output_tokens_per_s: number;
  ttft_ms: Percentiles;
  total_latency_ms: Percentiles;
  queue_wait_ms: Percentiles;
  cache_hit_rate: number;
  mean_cache_overlap: number;
  prefill_tokens_total: number;
  prefill_tokens_saved: number;
  prefill_saved_fraction: number;
  worker_request_counts: Record<string, number>;
  worker_imbalance: number;
  kv: CacheSummary[];
}

export interface BenchmarkRun {
  run_id: string;
  created_at: string;
  workload: WorkloadSpec;
  policy: string;
  policy_parameters: Record<string, number>;
  seed: number;
  requested: number;
  workers: number;
  metrics: BenchmarkMetrics;
}

export interface SystemEvent {
  event_id: string;
  event_type: string;
  request_id: string | null;
  worker_id: string | null;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface ChatSendResult {
  requestId: string;
  decisionId: string;
  policy: string;
  worker: string;
}
