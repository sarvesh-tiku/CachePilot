export const ms = (v: number | null | undefined, digits = 0) =>
  v == null ? "—" : `${v.toFixed(digits)} ms`;

export const pct = (v: number | null | undefined, digits = 0) =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 0) =>
  v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: digits });

export const gib = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(1)} GiB`;

export const shortId = (id: string) => id.slice(0, 8);

export const clock = (iso: string) =>
  new Date(iso).toLocaleTimeString(undefined, { hour12: false });

export const POLICY_LABEL: Record<string, string> = {
  round_robin: "round robin",
  least_loaded: "least loaded",
  kv_aware: "KV-aware",
};

export const POLICY_ORDER = ["round_robin", "least_loaded", "kv_aware"];

export const POLICY_COLOR: Record<string, string> = {
  round_robin: "#7a8699",
  least_loaded: "#d29922",
  kv_aware: "#2ea043",
};

// SVG presentation attributes cannot resolve CSS variables, so charts use literal colors.
export const CHART = {
  blue: "#3b82f6",
  green: "#2ea043",
  amber: "#d29922",
  red: "#d1242f",
  muted: "#7a8699",
};
