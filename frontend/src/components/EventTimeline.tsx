import type { SystemEvent } from "../types";

const HIDDEN_KEYS = new Set(["reason", "decision_id", "prefix_fingerprint"]);

function summarize(payload: Record<string, unknown>): string {
  return Object.entries(payload)
    .filter(([k]) => !HIDDEN_KEYS.has(k))
    .map(([k, v]) => `${k}=${typeof v === "number" ? Number(v.toFixed(3)) : String(v)}`)
    .join("  ");
}

export function EventTimeline({ events }: { events: SystemEvent[] }) {
  if (!events.length) return <p className="empty">No events recorded.</p>;
  const t0 = new Date(events[0].timestamp).getTime();
  return (
    <table className="table events">
      <thead>
        <tr>
          <th>+ms</th>
          <th>Event</th>
          <th>Worker</th>
          <th>Details</th>
        </tr>
      </thead>
      <tbody>
        {events.map((e) => (
          <tr key={e.event_id}>
            <td className="num mono">{(new Date(e.timestamp).getTime() - t0).toFixed(0)}</td>
            <td className="mono">{e.event_type.toLowerCase()}</td>
            <td className="mono">{e.worker_id ?? ""}</td>
            <td className="mono details">{summarize(e.payload)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
