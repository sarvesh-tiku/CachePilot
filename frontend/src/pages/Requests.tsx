import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { fetchRequests, fetchWorkers } from "../api/client";
import { TrafficPanel } from "../components/TrafficPanel";
import { Bar, Empty, ErrorNote, Panel, PolicyBadge } from "../components/ui";
import { clock, ms, num, pct, POLICY_LABEL, POLICY_ORDER, shortId } from "../format";
import { usePolling } from "../hooks/usePolling";

export default function Requests() {
  const navigate = useNavigate();
  const [policy, setPolicy] = useState("");
  const [worker, setWorker] = useState("");
  const { data: rows, error, refresh } = usePolling(
    () => fetchRequests({ policy: policy || undefined, worker_id: worker || undefined, limit: 100 }),
    2000,
    [policy, worker],
  );
  const { data: workers } = usePolling(fetchWorkers, 0);

  return (
    <>
      <TrafficPanel onSent={refresh} />
      <Panel
        title="Requests"
        actions={
          <div className="form-row compact">
            <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
              <option value="">all policies</option>
              {POLICY_ORDER.map((p) => (
                <option key={p} value={p}>
                  {POLICY_LABEL[p]}
                </option>
              ))}
            </select>
            <select value={worker} onChange={(e) => setWorker(e.target.value)}>
              <option value="">all workers</option>
              {(workers ?? []).map((w) => (
                <option key={w.worker_id} value={w.worker_id}>
                  {w.worker_id}
                </option>
              ))}
            </select>
          </div>
        }
      >
        <ErrorNote error={error} />
        {rows && rows.length === 0 && <Empty>No requests yet. Send a burst above or POST to /v1/chat/completions.</Empty>}
        {rows && rows.length > 0 && (
          <table className="table clickable">
            <thead>
              <tr>
                <th>Time</th>
                <th>Request</th>
                <th>Policy</th>
                <th>Worker</th>
                <th>Prompt tokens</th>
                <th>Cache overlap</th>
                <th>TTFT</th>
                <th>Total latency</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.request.request_id} onClick={() => navigate(`/requests/${r.request.request_id}`)}>
                  <td className="mono">{clock(r.request.created_at)}</td>
                  <td className="mono">
                    <Link to={`/requests/${r.request.request_id}`}>{shortId(r.request.request_id)}</Link>
                  </td>
                  <td>
                    <PolicyBadge policy={r.decision?.policy} />
                  </td>
                  <td className="mono">{r.decision?.selected_worker_id ?? "—"}</td>
                  <td className="num">{num(r.request.prompt_tokens_estimate)}</td>
                  <td>
                    {r.result ? (
                      <Bar fraction={r.result.cache_overlap} color="var(--green)" label={pct(r.result.cache_overlap)} />
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="num">{ms(r.result?.ttft_ms)}</td>
                  <td className="num">{ms(r.result?.total_latency_ms)}</td>
                  <td>
                    <span className={`status status-${r.status}`}>{r.status.replace("_", " ")}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </>
  );
}
