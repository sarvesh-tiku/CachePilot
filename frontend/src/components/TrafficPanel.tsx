import { useState } from "react";
import { fetchPolicies, sendChat } from "../api/client";
import { usePolling } from "../hooks/usePolling";
import { POLICY_LABEL } from "../format";
import { Panel } from "./ui";

const SENTENCE =
  "Company travel policy, section 7: employees must file receipts within thirty days of return. ";

function sharedPrefix(tokens: number): string {
  const chars = tokens * 4;
  return SENTENCE.repeat(Math.ceil(chars / SENTENCE.length)).slice(0, chars);
}

export function TrafficPanel({ onSent }: { onSent: () => void }) {
  const { data: policies } = usePolling(fetchPolicies, 0);
  const [policy, setPolicy] = useState("kv_aware");
  const [count, setCount] = useState(12);
  const [prefixTokens, setPrefixTokens] = useState(4096);
  const [busy, setBusy] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);

  async function send() {
    setBusy(true);
    setSummary(null);
    const prefix = sharedPrefix(prefixTokens);
    try {
      const results = await Promise.all(
        Array.from({ length: count }, (_, i) =>
          sendChat(policy, {
            model: "sim-model",
            messages: [
              { role: "system", content: prefix },
              { role: "user", content: `Question ${i} asked at ${Date.now()}` },
            ],
            max_tokens: 32,
          }),
        ),
      );
      const byWorker: Record<string, number> = {};
      for (const r of results) byWorker[r.worker] = (byWorker[r.worker] ?? 0) + 1;
      const spread = Object.entries(byWorker)
        .sort()
        .map(([w, n]) => `${w}: ${n}`)
        .join("  ·  ");
      setSummary(`${count} requests under ${POLICY_LABEL[policy] ?? policy} → ${spread}`);
      onSent();
    } catch (err) {
      setSummary(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel title="Generate traffic">
      <div className="form-row">
        <label>
          Policy
          <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
            {(policies ?? []).map((p) => (
              <option key={p.name} value={p.name}>
                {POLICY_LABEL[p.name] ?? p.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Requests
          <input type="number" min={1} max={64} value={count} onChange={(e) => setCount(Number(e.target.value))} />
        </label>
        <label>
          Shared prefix tokens
          <input
            type="number"
            min={128}
            max={16384}
            step={128}
            value={prefixTokens}
            onChange={(e) => setPrefixTokens(Number(e.target.value))}
          />
        </label>
        <button onClick={() => void send()} disabled={busy}>
          {busy ? "Sending…" : "Send burst"}
        </button>
      </div>
      <p className="hint">
        All requests share one system prompt and differ only in the question. Send the same burst
        twice: the first warms whichever workers serve it, the second shows where each policy
        routes reusable prefixes.
      </p>
      {summary && <p className="summary mono">{summary}</p>}
    </Panel>
  );
}
