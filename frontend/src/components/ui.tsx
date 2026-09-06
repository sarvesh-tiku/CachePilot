import type { ReactNode } from "react";
import { POLICY_COLOR, POLICY_LABEL } from "../format";

export function StatTile({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

export function PolicyBadge({ policy }: { policy: string | null | undefined }) {
  if (!policy) return <span className="badge muted">—</span>;
  return (
    <span className="badge" style={{ borderColor: POLICY_COLOR[policy], color: POLICY_COLOR[policy] }}>
      {POLICY_LABEL[policy] ?? policy}
    </span>
  );
}

export function StatusDot({ status }: { status: string }) {
  return <span className={`dot dot-${status}`} title={status} />;
}

export function Bar({
  fraction,
  color = "var(--accent)",
  label,
}: {
  fraction: number;
  color?: string;
  label?: ReactNode;
}) {
  const width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  return (
    <div className="bar-wrap">
      <div className="bar-track">
        <div className="bar-fill" style={{ width, background: color }} />
      </div>
      {label !== undefined && <span className="bar-label">{label}</span>}
    </div>
  );
}

export function Panel({ title, children, actions }: { title: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{title}</h2>
        {actions}
      </header>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}

export function ErrorNote({ error }: { error: string | null }) {
  return error ? <p className="error">{error}</p> : null;
}
