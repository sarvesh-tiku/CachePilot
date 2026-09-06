import { ms } from "../format";
import type { RequestResult } from "../types";

export function Waterfall({ result, predictedTtft }: { result: RequestResult; predictedTtft?: number }) {
  const queue = result.queue_wait_ms;
  const prefill = Math.max(0, result.ttft_ms - result.queue_wait_ms);
  const decode = Math.max(0, result.total_latency_ms - result.ttft_ms);
  const total = Math.max(result.total_latency_ms, 1);
  const segments = [
    { label: "queue wait", value: queue, color: "var(--amber)" },
    { label: "prefill + overhead", value: prefill, color: "var(--blue)" },
    { label: "decode", value: decode, color: "var(--green)" },
  ];

  return (
    <div className="waterfall">
      <div className="waterfall-track">
        {segments.map((s) => (
          <div
            key={s.label}
            className="waterfall-seg"
            style={{ width: `${(s.value / total) * 100}%`, background: s.color }}
            title={`${s.label}: ${ms(s.value, 1)}`}
          />
        ))}
        {predictedTtft !== undefined && (
          <div
            className="waterfall-marker"
            style={{ left: `${Math.min(100, (predictedTtft / total) * 100)}%` }}
            title={`predicted TTFT ${ms(predictedTtft, 1)}`}
          />
        )}
      </div>
      <ul className="legend">
        {segments.map((s) => (
          <li key={s.label}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label} <b>{ms(s.value, 1)}</b>
          </li>
        ))}
        {predictedTtft !== undefined && (
          <li>
            <span className="swatch marker" /> predicted TTFT <b>{ms(predictedTtft, 1)}</b> vs actual{" "}
            <b>{ms(result.ttft_ms, 1)}</b>
          </li>
        )}
      </ul>
    </div>
  );
}
