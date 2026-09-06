import { drainWorker, fetchMetrics, fetchPolicies, restoreWorker } from "../api/client";
import { usePolling } from "../hooks/usePolling";
import { ms, num, pct } from "../format";
import { Bar, ErrorNote, Panel, PolicyBadge, StatTile, StatusDot } from "../components/ui";

const WINDOW_S = 60;

export default function Overview() {
  const { data: m, error, refresh } = usePolling(() => fetchMetrics(WINDOW_S), 2000);
  const { data: policies } = usePolling(fetchPolicies, 0);

  async function toggle(workerId: string, status: string) {
    await (status === "healthy" ? drainWorker(workerId) : restoreWorker(workerId));
    refresh();
  }

  return (
    <>
      <ErrorNote error={error} />
      <div className="tiles">
        <StatTile label="Healthy workers" value={m ? `${m.healthy_workers} / ${m.total_workers}` : "—"} />
        <StatTile label="Requests / s" value={num(m?.requests_per_s, 2)} sub={`last ${WINDOW_S}s`} />
        <StatTile label="Tokens / s" value={num(m?.output_tokens_per_s, 0)} sub="output tokens" />
        <StatTile label="Cache hit rate" value={pct(m?.cache_hit_rate)} sub={`mean overlap ${pct(m?.mean_cache_overlap)}`} />
        <StatTile label="TTFT p50" value={ms(m?.ttft_ms.p50)} />
        <StatTile label="TTFT p99" value={ms(m?.ttft_ms.p99)} sub={`p95 ${ms(m?.ttft_ms.p95)}`} />
        <StatTile label="Active requests" value={num(m?.active_requests)} sub={`${num(m?.queued_requests)} queued`} />
        <StatTile label="Completed" value={num(m?.completed_requests)} sub={`in window`} />
      </div>

      <Panel title="Workers">
        <div className="cards">
          {(m?.workers ?? []).map((w) => (
            <article key={w.worker_id} className="card">
              <h3 className="mono">
                <StatusDot status={w.status} /> {w.worker_id}
                <span className={`chip backend-${w.backend}`}>{w.backend}</span>
              </h3>
              <dl>
                <dt>status</dt>
                <dd>{w.status}</dd>
                <dt>queue</dt>
                <dd>{w.queue_depth}</dd>
                <dt>active</dt>
                <dd>{w.active_requests}</dd>
                <dt>KV</dt>
                <dd>
                  <Bar fraction={w.kv_utilization} color="var(--amber)" label={pct(w.kv_utilization)} />
                </dd>
                <dt>tokens/sec</dt>
                <dd>{num(w.tokens_per_second_estimate, 0)}</dd>
                <dt>cache entries</dt>
                <dd>{w.cache_entries}</dd>
                <dt>served ({WINDOW_S}s)</dt>
                <dd>{w.completed_in_window}</dd>
              </dl>
              <button className="ghost" onClick={() => void toggle(w.worker_id, w.status)}>
                {w.status === "healthy" ? "drain" : "restore"}
              </button>
            </article>
          ))}
        </div>
      </Panel>

      <Panel title="Routing policies">
        <table className="table">
          <thead>
            <tr>
              <th>Policy</th>
              <th>Default</th>
              <th>Parameters</th>
            </tr>
          </thead>
          <tbody>
            {(policies ?? []).map((p) => (
              <tr key={p.name}>
                <td>
                  <PolicyBadge policy={p.name} />
                </td>
                <td>{p.default ? "yes" : ""}</td>
                <td className="mono">
                  {Object.keys(p.parameters).length
                    ? Object.entries(p.parameters)
                        .map(([k, v]) => `${k}=${v}`)
                        .join("  ")
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">
          Override per request with the <code>X-CachePilot-Policy</code> header. KV-aware score = α·overlap −
          β·load/max_load − γ·kv_pressure − δ·predicted_ttft_ms.
        </p>
        {policies?.[0]?.estimator && (
          <p className="hint mono">
            predicted TTFT = {policies[0].estimator.queue_wait_ms_per_pending} ms × pending +{" "}
            {policies[0].estimator.prefill_ms_per_token} ms × uncached tokens +{" "}
            {policies[0].estimator.fixed_overhead_ms} ms
          </p>
        )}
      </Panel>
    </>
  );
}
