import { useState } from "react";
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { useLoad } from "../state";
import { C, Card, ErrorBox, Loading, MODE_COLOR, Notice, Page, int, num, pct } from "../ui";

export default function DataExplorer() {
  const summary = useLoad(api.summary, []);
  const feats = useLoad(api.features, []);
  const [hf, setHf] = useState("Torque [Nm]");
  const [sx, setSx] = useState("Rotational speed [rpm]");
  const [sy, setSy] = useState("Torque [Nm]");
  const hist = useLoad(() => api.histogram(hf, 30), [hf]);
  const sc = useLoad(() => api.scatter(sx, sy), [sx, sy]);
  const [page, setPage] = useState(0);
  const [failOnly, setFailOnly] = useState(false);
  const rows = useLoad(() => api.rows(page * 25, 25, failOnly), [page, failOnly]);
  const s = summary.data;
  const opts = feats.data ?? [];
  const sel = (v: string, set: (x: string) => void, label: string) => (
    <select value={v} onChange={e => set(e.target.value)} aria-label={label}>{opts.map(o => <option key={o.key} value={o.key}>{o.label}</option>)}</select>);

  const hdata = hist.data ? hist.data.normal.map((n, i) => ({
    x: (hist.data!.edges[i] + hist.data!.edges[i + 1]) / 2, normal: n, failure: hist.data!.failure[i], rate: hist.data!.failure_rate[i] * 100,
  })) : [];
  const modes = sc.data ? Array.from(new Set(sc.data.failure.map(p => p.mode))) : [];

  return (
    <Page title="Data Explorer" subtitle="The AI4I 2020 predictive-maintenance dataset: 10,000 machine records with five operating variables and labelled failures.">
      {summary.error && <ErrorBox error={summary.error} />}
      {s && <Notice>{s.synthetic_note} Data quirks kept as published: {s.quirks.failures_without_mode} failures with no mode flag,
        {" "}{s.quirks.rnf_not_failure} random failures labelled as non-failures, {s.quirks.multi_mode} rows with several modes.
        Failure-mode columns are never used as model inputs (leakage).</Notice>}
      <Card title="Feature statistics">
        {!s ? <Loading /> : (
          <div className="tablewrap">
            <table className="t">
              <thead><tr><th>Feature</th><th className="n">Mean</th><th className="n">Std</th><th className="n">Min</th><th className="n">Median</th>
                <th className="n">Max</th><th className="n">Mean (normal)</th><th className="n">Mean (failure)</th></tr></thead>
              <tbody>{s.stats.map(f => (
                <tr key={f.key}><td>{f.feature}</td><td className="n">{num(f.mean, 1)}</td><td className="n">{num(f.std, 1)}</td><td className="n">{num(f.min, 1)}</td>
                  <td className="n">{num(f.median, 1)}</td><td className="n">{num(f.max, 1)}</td><td className="n">{num(f.mean_normal, 1)}</td>
                  <td className="n" style={{ color: C.high }}>{num(f.mean_failure, 1)}</td></tr>))}</tbody>
            </table>
          </div>)}
      </Card>
      <div className="grid g2 mt">
        <Card title="Distribution by outcome" extra={sel(hf, setHf, "Histogram feature")}>
          {!hist.data ? <Loading /> : (
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={hdata}>
                <CartesianGrid stroke="#eef2f6" />
                <XAxis dataKey="x" tickFormatter={v => Number(v).toFixed(0)} tick={{ fontSize: 11 }} />
                <YAxis yAxisId="c" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="r" orientation="right" tickFormatter={v => `${v}%`} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number, n: string) => (n === "failure rate" ? `${v.toFixed(1)}%` : v)} labelFormatter={l => `${hist.data!.feature} ≈ ${Number(l).toFixed(1)}`} />
                <Legend verticalAlign="top" />
                <Bar yAxisId="c" dataKey="normal" stackId="a" fill="#cbd5e1" name="normal" />
                <Bar yAxisId="c" dataKey="failure" stackId="a" fill={C.high} name="failure" />
                <Line yAxisId="r" dataKey="rate" stroke={C.orange} dot={false} strokeWidth={2} name="failure rate" />
              </ComposedChart>
            </ResponsiveContainer>)}
        </Card>
        <Card title="Failure regions" extra={<div className="row">{sel(sx, setSx, "X axis")}{sel(sy, setSy, "Y axis")}</div>}>
          {!sc.data ? <Loading /> : (
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ left: -6 }}>
                <CartesianGrid stroke="#eef2f6" />
                <XAxis type="number" dataKey="x" name={sc.data.x} domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
                <YAxis type="number" dataKey="y" name={sc.data.y} domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend verticalAlign="top" />
                <Scatter name="normal (sample)" data={sc.data.normal} fill="#cbd5e1" shape="circle" isAnimationActive={false} />
                {modes.map(m => <Scatter key={m || "none"} name={m || "no mode"} data={sc.data!.failure.filter(p => p.mode === m)}
                  fill={MODE_COLOR[m] ?? C.ink} isAnimationActive={false} />)}
              </ScatterChart>
            </ResponsiveContainer>)}
          <p className="small muted">Try power or tool wear × torque against speed to see the documented failure thresholds.</p>
        </Card>
      </div>
      <Card className="mt" title={rows.data ? `Records (${int(rows.data.total)})` : "Records"} extra={
        <div className="row">
          <label className="small"><input type="checkbox" checked={failOnly} onChange={e => { setFailOnly(e.target.checked); setPage(0); }} /> failures only</label>
          <button className="btn btn-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Previous</button>
          <span className="small">page {page + 1}{rows.data ? ` of ${Math.ceil(rows.data.total / 25)}` : ""}</span>
          <button className="btn btn-sm" disabled={!rows.data || (page + 1) * 25 >= rows.data.total} onClick={() => setPage(p => p + 1)}>Next</button>
        </div>}>
        {!rows.data ? <Loading /> : (
          <div className="tablewrap">
            <table className="t">
              <thead><tr>{Object.keys(rows.data.rows[0] ?? {}).map(k => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{rows.data.rows.map((r, i) => (
                <tr key={i} style={r["Machine failure"] === 1 ? { background: "#fef2f2" } : undefined}>
                  {Object.values(r).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>))}</tbody>
            </table>
          </div>)}
        {s && <p className="small muted">Overall failure rate {pct(s.failure_rate, 2)}.</p>}
      </Card>
    </Page>
  );
}
