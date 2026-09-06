import { Link, useParams } from "react-router-dom";
import { fetchEvents, fetchRequest } from "../api/client";
import { CandidateTable } from "../components/CandidateTable";
import { EventTimeline } from "../components/EventTimeline";
import { Waterfall } from "../components/Waterfall";
import { Empty, ErrorNote, Panel, PolicyBadge, StatTile } from "../components/ui";
import { ms, num, pct } from "../format";
import { usePolling } from "../hooks/usePolling";

export default function RequestDetail() {
  const { id = "" } = useParams();
  const { data: view, error } = usePolling(() => fetchRequest(id), 1000, [id]);
  const { data: events } = usePolling(() => fetchEvents(id), 1000, [id]);

  if (error) return <ErrorNote error={error} />;
  if (!view) return <Empty>Loading…</Empty>;

  const { request, decision, result } = view;
  const selected = decision?.candidates.find((c) => c.worker_id === decision.selected_worker_id);

  return (
    <>
      <p className="crumbs">
        <Link to="/requests">← requests</Link>
      </p>
      <header className="detail-head">
        <h1 className="mono">{request.request_id}</h1>
        <PolicyBadge policy={decision?.policy} />
        <span className={`status status-${view.status}`}>{view.status.replace("_", " ")}</span>
      </header>

      <div className="tiles">
        <StatTile label="Selected worker" value={<span className="mono">{decision?.selected_worker_id ?? "—"}</span>} />
        <StatTile label="Prompt tokens" value={num(request.prompt_tokens_estimate)} sub={`${request.message_count} messages · max_tokens ${request.max_tokens}`} />
        <StatTile label="Cache overlap" value={pct(result?.cache_overlap ?? selected?.cache_overlap)} sub={result ? (result.cache_hit ? "cache hit" : "cache miss") : "pending"} />
        <StatTile label="TTFT" value={ms(result?.ttft_ms, 1)} sub={selected ? `predicted ${ms(selected.estimated_ttft_ms, 1)}` : undefined} />
        <StatTile label="Total latency" value={ms(result?.total_latency_ms, 1)} sub={result ? `${result.output_tokens} output tokens` : undefined} />
        <StatTile label="Queue wait" value={ms(result?.queue_wait_ms, 1)} sub={selected ? `queue depth ${selected.queue_depth} at routing` : undefined} />
      </div>

      {decision ? (
        <Panel title="Routing decision">
          <blockquote className="reason">{decision.reason}</blockquote>
          <CandidateTable decision={decision} />
          <p className="hint">
            Candidates are every healthy worker at the moment of routing, ranked by the policy's final score.
            Estimated TTFT = queue wait + uncached prefill + fixed overhead.
          </p>
        </Panel>
      ) : (
        <Panel title="Routing decision">
          <Empty>No routing decision recorded.</Empty>
        </Panel>
      )}

      <Panel title="Timing">
        {result ? (
          <Waterfall result={result} predictedTtft={selected?.estimated_ttft_ms} />
        ) : (
          <Empty>Request still in flight.</Empty>
        )}
      </Panel>

      <Panel title="Event timeline">
        <EventTimeline events={events ?? []} />
      </Panel>

      <Panel title="Request metadata">
        <dl className="meta">
          <dt>model</dt>
          <dd className="mono">{request.model}</dd>
          <dt>created</dt>
          <dd className="mono">{new Date(request.created_at).toISOString()}</dd>
          <dt>prefix fingerprint</dt>
          <dd className="mono">{request.prefix_fingerprint ?? "— (prompt shorter than one chunk)"}</dd>
          <dt>latency SLO</dt>
          <dd>{request.latency_slo_ms ? ms(request.latency_slo_ms) : "none"}</dd>
          <dt>decision id</dt>
          <dd className="mono">{decision?.decision_id ?? "—"}</dd>
        </dl>
        <p className="hint">Prompt text is never persisted; only counts, estimates, and the prefix fingerprint.</p>
      </Panel>
    </>
  );
}
