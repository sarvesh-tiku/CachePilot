import { ms, num, pct } from "../format";
import type { RoutingDecision } from "../types";
import { Bar } from "./ui";

export function CandidateTable({ decision }: { decision: RoutingDecision }) {
  const ranked = [...decision.candidates].sort((a, b) => b.final_score - a.final_score);
  const maxTtft = Math.max(...ranked.map((c) => c.estimated_ttft_ms), 1);
  const scores = ranked.map((c) => c.final_score);
  const minScore = Math.min(...scores);
  const span = Math.max(...scores) - minScore || 1;

  return (
    <table className="table candidates">
      <thead>
        <tr>
          <th>Worker</th>
          <th>Prefix overlap</th>
          <th>Queue</th>
          <th>KV pressure</th>
          <th>Est. TTFT</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        {ranked.map((c) => {
          const selected = c.worker_id === decision.selected_worker_id;
          return (
            <tr key={c.worker_id} className={selected ? "selected" : undefined}>
              <td className="mono">
                {c.worker_id}
                {selected && <span className="chip">selected</span>}
              </td>
              <td>
                <Bar fraction={c.cache_overlap} color="var(--green)" label={pct(c.cache_overlap)} />
              </td>
              <td className="num">{c.queue_depth}</td>
              <td>
                <Bar fraction={c.kv_pressure} color="var(--amber)" label={pct(c.kv_pressure)} />
              </td>
              <td>
                <Bar fraction={c.estimated_ttft_ms / maxTtft} color="var(--blue)" label={ms(c.estimated_ttft_ms)} />
              </td>
              <td>
                <Bar
                  fraction={(c.final_score - minScore) / span}
                  color={selected ? "var(--accent)" : "var(--muted)"}
                  label={num(c.final_score, 3)}
                />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
