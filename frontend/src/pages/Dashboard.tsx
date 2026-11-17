import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Risk } from "../api";
import { useApp, useLoad } from "../state";
import { C, Card, ErrorBox, Loading, MODE_COLOR, Page, RISK_COLOR, RiskBadge, Stat, int, pct, when } from "../ui";

const ORDER = ["IEMAF", "Stacking", "Soft Voting", "BMA", "XGBoost", "Gradient Boosting", "Random Forest", "AdaBoost", "CART", "MARS", "GAM"];

export default function Dashboard() {
  const { fs, health } = useApp();
  const summary = useLoad(api.summary, []);
  const stats = useLoad(api.historyStats, [health?.history_count]);
  const ready = !!health?.models?.[fs]?.ready;
  const models = useLoad(() => (ready ? api.models(fs) : Promise.resolve(null)), [fs, ready]);

  const s = summary.data;
  const m = models.data;
  const iemaf = m?.metrics_test?.IEMAF;
  const perf = m ? ORDER.filter(n => m.metrics_test[n]).map(n => ({
    name: n === "IEMAF" ? "IEMAF (cal. BMA)" : n, pr: m.metrics_test[n].pr_auc, recall: m.metrics_test[n].recall,
  })) : [];
  const riskData = (["Low", "Medium", "High", "Uncertain"] as Risk[]).map(r => ({ name: r, value: stats.data?.risk_counts?.[r] ?? 0 }))
    .filter(d => d.value > 0);

  return (
    <Page title="Dashboard" subtitle="Machine-failure risk on the AI4I 2020 data: dataset health, model quality and recent predictions."
      actions={<><Link className="btn btn-primary" to="/predict">New prediction</Link><Link className="btn" to="/batch">Score a CSV</Link></>}>
      {summary.error && <ErrorBox error={summary.error} />}
      <div className="grid g4">
        <Stat label="Machine records" value={s ? int(s.rows) : "–"} hint={s ? `${s.types.length} product types (L / M / H)` : ""} />
        <Stat label="Failure rate" value={s ? pct(s.failure_rate, 2) : "–"} hint={s ? `${int(s.failures)} failures` : ""} tone={C.high} />
        <Stat label={`Test PR-AUC (${fs})`} value={iemaf ? iemaf.pr_auc.toFixed(3) : ready ? "…" : "not trained"}
          hint={iemaf ? `recall ${pct(iemaf.recall)} · precision ${pct(iemaf.precision)}` : "calibrated BMA on 1,500 held-out rows"} tone={C.accent} />
        <Stat label="Predictions logged" value={stats.data ? int(stats.data.total) : "–"}
          hint={health ? `stored in ${health.store === "mongodb" ? "MongoDB" : "a local file"}` : ""} />
      </div>

      <div className="grid g3 mt">
        <Card title="Failure modes">
          {!s ? <Loading /> : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={s.modes} margin={{ left: -10 }}>
                <CartesianGrid vertical={false} stroke="#eef2f6" />
                <XAxis dataKey="mode" tickLine={false} />
                <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
                <Tooltip formatter={(v: number, _n, p) => [v, p.payload.name]} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>{s.modes.map(d => <Cell key={d.mode} fill={MODE_COLOR[d.mode]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>)}
          <p className="small muted">A row can have several modes; 9 failures have none and 18 random failures are not labelled as failures.</p>
        </Card>
        <Card title="Failure rate by product type">
          {!s ? <Loading /> : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={s.types} margin={{ left: -6 }}>
                <CartesianGrid vertical={false} stroke="#eef2f6" />
                <XAxis dataKey="type" tickLine={false} />
                <YAxis tickFormatter={v => pct(v, 0)} tickLine={false} axisLine={false} />
                <Tooltip formatter={(v: number, _n, p) => [`${pct(v, 2)} (${p.payload.failures} of ${int(p.payload.rows)})`, "failure rate"]} />
                <Bar dataKey="failure_rate" fill={C.orange} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>)}
          <p className="small muted">L = low, M = medium, H = high quality variant.</p>
        </Card>
        <Card title="Logged risk levels">
          {riskData.length === 0 ? <p className="muted">No predictions yet. <Link to="/predict">Make the first one.</Link></p> : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={riskData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2}>
                  {riskData.map(d => <Cell key={d.name} fill={RISK_COLOR[d.name as Risk]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>)}
          <div className="legend">{riskData.map(d => <span key={d.name}><i style={{ background: RISK_COLOR[d.name as Risk] }} />{d.name} {d.value}</span>)}</div>
        </Card>
      </div>

      <div className="grid g2 mt">
        <Card title={`Model quality on the test split (${fs} features)`} extra={<Link to="/models" className="small">Model Lab →</Link>}>
          {!ready ? <p className="muted">Train the models on the <Link to="/experiments">Experiments</Link> page to see live metrics.</p> :
            !m ? <Loading /> : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={perf} layout="vertical" margin={{ left: 0, right: 16 }}>
                  <CartesianGrid horizontal={false} stroke="#eef2f6" />
                  <XAxis type="number" domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
                  <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v: number) => v.toFixed(3)} />
                  <Bar dataKey="pr" name="PR-AUC" radius={[0, 4, 4, 0]}>
                    {perf.map((d, i) => <Cell key={d.name} fill={i === 0 ? C.accent : "#94a3b8"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>)}
        </Card>
        <Card title="Recent predictions" extra={<Link to="/history" className="small">All history →</Link>}>
          {!stats.data ? <Loading /> : stats.data.recent.length === 0 ? <p className="muted">Nothing logged yet.</p> : (
            <table className="t">
              <thead><tr><th>When</th><th>Machine</th><th>Type</th><th className="n">Probability</th><th>Risk</th></tr></thead>
              <tbody>{stats.data.recent.map(h => (
                <tr key={h.id}><td>{when(h.created_at)}</td><td>{h.machine_id || "–"}</td><td>{h.input.type}</td>
                  <td className="n">{pct(h.probability)}</td><td><RiskBadge risk={h.risk} /></td></tr>))}</tbody>
            </table>)}
        </Card>
      </div>
    </Page>
  );
}
