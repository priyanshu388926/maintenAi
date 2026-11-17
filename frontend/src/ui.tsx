import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { Risk } from "./api";

export const C = {
  accent: "#0f766e", blue: "#2563eb", orange: "#ea580c", gray: "#94a3b8", ink: "#0f172a",
  low: "#16a34a", medium: "#d97706", high: "#dc2626", uncertain: "#7c3aed",
  series: ["#2563eb", "#0f766e", "#ea580c", "#d97706", "#db2777", "#7c3aed", "#0891b2", "#65a30d", "#475569", "#b91c1c"],
};
export const RISK_COLOR: Record<Risk, string> = { Low: C.low, Medium: C.medium, High: C.high, Uncertain: C.uncertain };
export const MODE_COLOR: Record<string, string> = { TWF: "#d97706", HDF: "#dc2626", PWF: "#2563eb", OSF: "#7c3aed", RNF: "#0891b2", "": "#64748b" };

export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`;
export const num = (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : "–");
export const int = (x: number) => x.toLocaleString("en-US");
export const when = (iso: string) => new Date(iso).toLocaleString();

export function Page({ title, subtitle, actions, children }: { title: string; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>{title}</h1>
          {subtitle && <p className="sub">{subtitle}</p>}
        </div>
        {actions && <div className="actions">{actions}</div>}
      </header>
      {children}
    </div>
  );
}

export function Card({ title, extra, children, className = "" }: { title?: ReactNode; extra?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || extra) && <div className="card-head"><h2>{title}</h2>{extra}</div>}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={tone ? { color: tone } : undefined}>{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function RiskBadge({ risk, big }: { risk: Risk; big?: boolean }) {
  return <span className={`badge ${big ? "badge-big" : ""}`} style={{ background: RISK_COLOR[risk] }}>{risk}</span>;
}

export function Loading({ text = "Loading…" }: { text?: string }) {
  return <div className="loading"><span className="spinner" />{text}</div>;
}

export function ErrorBox({ error, children }: { error: string; children?: ReactNode }) {
  return <div className="alert alert-error"><strong>Something went wrong.</strong> {error}{children}</div>;
}

export function Notice({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" }) {
  return <div className={`alert alert-${tone}`}>{children}</div>;
}

export function NeedsModels({ error }: { error: string }) {
  return (
    <div className="alert alert-warn">
      <strong>Models are not ready.</strong> {error} <Link to="/experiments">Open Experiments to train them</Link> (about 2 minutes).
    </div>
  );
}

export function ProbBar({ value, threshold, color }: { value: number; threshold?: number; color?: string }) {
  return (
    <div className="probbar" title={pct(value, 2)}>
      <div className="probbar-fill" style={{ width: `${Math.max(0.5, value * 100)}%`, background: color ?? C.blue }} />
      {threshold !== undefined && <div className="probbar-thr" style={{ left: `${threshold * 100}%` }} />}
    </div>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: string }[]; value: T; onChange: (t: T) => void }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map(t => (
        <button key={t.id} role="tab" aria-selected={value === t.id} className={value === t.id ? "tab active" : "tab"}
          onClick={() => onChange(t.id)}>{t.label}</button>
      ))}
    </div>
  );
}

export function downloadText(name: string, text: string, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return "";
  const cols = Object.keys(rows[0]);
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(","), ...rows.map(r => cols.map(c => esc(r[c])).join(","))].join("\n");
}
