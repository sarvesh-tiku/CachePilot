import { useEffect, useMemo, useState } from "react";
import {
  Bar as ChartBar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchBenchmarkRuns, fetchWorkloads, startBenchmark } from "../api/client";
import { Empty, ErrorNote, Panel } from "../components/ui";
import { CHART, ms, num, pct, POLICY_COLOR, POLICY_LABEL, POLICY_ORDER } from "../format";
import { usePolling } from "../hooks/usePolling";
import type { BenchmarkRun } from "../types";

interface Group {
  key: string;
  label: string;
  runs: BenchmarkRun[];
  latest: string;
}

function groupRuns(runs: BenchmarkRun[]): Group[] {
  const groups = new Map<string, Group>();
  for (const run of runs) {
    const key = `${run.workload.name}|${run.seed}|${run.requested}|${run.workers}`;
    const group = groups.get(key) ?? {
      key,
      label: `${run.workload.name} · seed ${run.seed} · ${run.requested} requests · ${run.workers} workers`,
      runs: [],
      latest: run.created_at,
    };
    group.runs.push(run);
    if (run.created_at > group.latest) group.latest = run.created_at;
    groups.set(key, group);
  }
  return [...groups.values()].sort((a, b) => (a.latest < b.latest ? 1 : -1));
}

function ChartCard({
  title,
  data,
  series,
  formatter,
}: {
  title: string;
  data: Record<string, number | string>[];
  series: { key: string; label: string; color: string }[];
  formatter: (v: number) => string;
}) {
  return (
    <div className="chart">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
          <XAxis dataKey="policy" tick={{ fontSize: 11 }} interval={0} />
          <YAxis tick={{ fontSize: 12 }} width={56} tickFormatter={(v: number) => formatter(v)} />
          <Tooltip formatter={(v: number) => formatter(v)} contentStyle={{ fontSize: 12 }} />
          {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
          {series.map((s) => (
            <ChartBar
              key={s.key}
              dataKey={s.key}
              name={s.label}
              fill={s.color}
              radius={[3, 3, 0, 0]}
              isAnimationActive={false}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Benchmarks() {
  const { data: runs, error, refresh } = usePolling(fetchBenchmarkRuns, 0);
  const { data: workloads } = usePolling(fetchWorkloads, 0);
  const groups = useMemo(() => groupRuns(runs ?? []), [runs]);
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [workload, setWorkload] = useState("shared_prefix_heavy");
  const [requests, setRequests] = useState(300);
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedKey && groups.length) setSelectedKey(groups[0].key);
  }, [groups, selectedKey]);

  const group = groups.find((g) => g.key === selectedKey);
  const ordered = (group?.runs ?? [])
    .slice()
    .sort((a, b) => POLICY_ORDER.indexOf(a.policy) - POLICY_ORDER.indexOf(b.policy));
  const chartData = ordered.map((r) => ({
    policy: POLICY_LABEL[r.policy] ?? r.policy,
    ttft_p50: r.metrics.ttft_ms.p50,
    ttft_p99: r.metrics.ttft_ms.p99,
    latency_p99: r.metrics.total_latency_ms.p99,
    queue_mean: r.metrics.queue_wait_ms.mean,
    hit_rate: r.metrics.cache_hit_rate * 100,
    prefill_saved: r.metrics.prefill_saved_fraction * 100,
    imbalance: r.metrics.worker_imbalance,
    evictions: r.metrics.kv.reduce((n, k) => n + k.evictions, 0),
  }));

  async function run() {
    setRunning(true);
    setRunError(null);
    try {
      const created = await startBenchmark({ workload, requests, seed });
      refresh();
      setSelectedKey(`${created[0].workload.name}|${seed}|${requests}|${created[0].workers}`);
    } catch (err) {
      setRunError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <Panel title="Run a benchmark">
        <div className="form-row">
          <label>
            Workload
            <select value={workload} onChange={(e) => setWorkload(e.target.value)}>
              {(workloads ?? []).map((w) => (
                <option key={w.name} value={w.name}>
                  {w.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Requests
            <input type="number" min={10} max={5000} step={10} value={requests} onChange={(e) => setRequests(Number(e.target.value))} />
          </label>
          <label>
            Seed
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
          <button onClick={() => void run()} disabled={running}>
            {running ? "Running all three policies…" : "Run all three policies"}
          </button>
        </div>
        <p className="hint">{workloads?.find((w) => w.name === workload)?.description}</p>
        <ErrorNote error={runError} />
      </Panel>

      <Panel
        title="Compare policies"
        actions={
          <select value={selectedKey} onChange={(e) => setSelectedKey(e.target.value)}>
            {groups.map((g) => (
              <option key={g.key} value={g.key}>
                {g.label}
              </option>
            ))}
          </select>
        }
      >
        <ErrorNote error={error} />
        {!group && <Empty>No benchmark runs yet. Run one above or use `make bench`.</Empty>}
        {group && (
          <>
            <div className="charts">
              <ChartCard
                title="TTFT"
                data={chartData}
                series={[
                  { key: "ttft_p50", label: "p50", color: CHART.blue },
                  { key: "ttft_p99", label: "p99", color: CHART.muted },
                ]}
                formatter={(v) => ms(v)}
              />
              <ChartCard
                title="Cache"
                data={chartData}
                series={[
                  { key: "hit_rate", label: "hit rate", color: CHART.green },
                  { key: "prefill_saved", label: "prefill saved", color: CHART.amber },
                ]}
                formatter={(v) => `${v.toFixed(0)}%`}
              />
              <ChartCard
                title="Queue wait (mean)"
                data={chartData}
                series={[{ key: "queue_mean", label: "queue wait", color: CHART.amber }]}
                formatter={(v) => ms(v)}
              />
              <ChartCard
                title="Worker imbalance"
                data={chartData}
                series={[{ key: "imbalance", label: "max / mean", color: CHART.red }]}
                formatter={(v) => v.toFixed(2)}
              />
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>TTFT p50</th>
                  <th>TTFT p99</th>
                  <th>Latency p99</th>
                  <th>Queue mean</th>
                  <th>Hit rate</th>
                  <th>Prefill saved</th>
                  <th>Imbalance</th>
                  <th>Evictions</th>
                  <th>Per worker</th>
                </tr>
              </thead>
              <tbody>
                {ordered.map((r) => (
                  <tr key={r.run_id}>
                    <td style={{ color: POLICY_COLOR[r.policy] }}>{POLICY_LABEL[r.policy] ?? r.policy}</td>
                    <td className="num">{ms(r.metrics.ttft_ms.p50)}</td>
                    <td className="num">{ms(r.metrics.ttft_ms.p99)}</td>
                    <td className="num">{ms(r.metrics.total_latency_ms.p99)}</td>
                    <td className="num">{ms(r.metrics.queue_wait_ms.mean)}</td>
                    <td className="num">{pct(r.metrics.cache_hit_rate, 1)}</td>
                    <td className="num">{pct(r.metrics.prefill_saved_fraction, 1)}</td>
                    <td className="num">{num(r.metrics.worker_imbalance, 2)}</td>
                    <td className="num">{num(r.metrics.kv.reduce((n, k) => n + k.evictions, 0))}</td>
                    <td className="mono">
                      {Object.entries(r.metrics.worker_request_counts)
                        .map(([w, n]) => `${w.replace("worker-", "w")}:${n}`)
                        .join(" ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="hint">
              {group.runs[0].workload.description} Simulated workers; see docs/benchmarking.md for the cost model.
            </p>
          </>
        )}
      </Panel>
    </>
  );
}
