import type {
  BenchmarkRun,
  CacheSummary,
  ChatSendResult,
  MetricsSummary,
  PolicyInfo,
  RequestView,
  SystemEvent,
  WorkerState,
  WorkloadSpec,
} from "../types";

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

async function postJson<T>(url: string, body: unknown, headers: Record<string, string> = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`POST ${url} failed: ${res.status} ${detail}`);
  }
  return { data: (await res.json()) as T, headers: res.headers };
}

export const fetchWorkers = () => getJson<WorkerState[]>("/api/workers");
export const fetchMetrics = (windowS: number) =>
  getJson<MetricsSummary>(`/api/metrics/summary?window_s=${windowS}`);
export const fetchPolicies = () => getJson<PolicyInfo[]>("/api/policies");
export const fetchCache = () => getJson<CacheSummary[]>("/api/kv");
export const fetchRequest = (id: string) => getJson<RequestView>(`/api/requests/${id}`);
export const fetchBenchmarkRuns = () => getJson<BenchmarkRun[]>("/api/benchmark-runs");
export const fetchEvents = (requestId: string) =>
  getJson<SystemEvent[]>(`/api/events?request_id=${requestId}&ascending=true`);
export const drainWorker = (id: string) => postJson<WorkerState>(`/api/workers/${id}/drain`, {});
export const restoreWorker = (id: string) =>
  postJson<WorkerState>(`/api/workers/${id}/restore`, {});
export const fetchWorkloads = () => getJson<WorkloadSpec[]>("/api/benchmark-runs/workloads");

export function fetchRequests(params: { policy?: string; worker_id?: string; limit?: number }) {
  const query = new URLSearchParams();
  if (params.policy) query.set("policy", params.policy);
  if (params.worker_id) query.set("worker_id", params.worker_id);
  query.set("limit", String(params.limit ?? 100));
  return getJson<RequestView[]>(`/api/requests?${query.toString()}`);
}

export async function startBenchmark(body: {
  workload: string;
  policies?: string[];
  requests: number;
  seed: number;
}): Promise<BenchmarkRun[]> {
  return (await postJson<BenchmarkRun[]>("/api/benchmark-runs", body)).data;
}

export async function sendChat(
  policy: string,
  body: { model: string; messages: { role: string; content: string }[]; max_tokens: number },
): Promise<ChatSendResult> {
  const { headers } = await postJson<unknown>("/v1/chat/completions", body, {
    "X-CachePilot-Policy": policy,
  });
  return {
    requestId: headers.get("X-CachePilot-Request-Id") ?? "",
    decisionId: headers.get("X-CachePilot-Decision-Id") ?? "",
    policy: headers.get("X-CachePilot-Policy") ?? policy,
    worker: headers.get("X-CachePilot-Worker") ?? "?",
  };
}
