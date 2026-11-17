import { useState } from "react";
import { api, type Risk } from "../api";
import { useApp, useLoad } from "../state";
import { Card, ErrorBox, Loading, Page, RiskBadge, downloadText, pct, toCsv, when } from "../ui";

export default function History() {
  const { refreshHealth } = useApp();
  const [risk, setRisk] = useState<Risk | "">("");
  const [q, setQ] = useState("");
  const h = useLoad(() => api.history(2000, risk || undefined), [risk]);
  const items = (h.data ?? []).filter(x => !q || (x.machine_id ?? "").toLowerCase().includes(q.toLowerCase()) ||
    (x.note ?? "").toLowerCase().includes(q.toLowerCase()));

  const del = async (id: string) => { await api.deleteHistory(id); h.reload(); refreshHealth(); };
  const clear = async () => {
    if (!window.confirm("Delete all saved predictions? This cannot be undone.")) return;
    await api.clearHistory(); h.reload(); refreshHealth();
  };
  const exportCsv = () => downloadText("maintenai_history.csv", toCsv(items.map(x => ({
    created_at: x.created_at, machine_id: x.machine_id ?? "", note: x.note ?? "", feature_set: x.feature_set, ...x.input,
    probability: x.probability, risk: x.risk, predicted_failure: x.predicted_failure, review_recommended: x.review_recommended,
    entropy: x.entropy, disagreement: x.disagreement, triggered_rules: x.triggered_rules.join(" "),
  }))));

  return (
    <Page title="Prediction History" subtitle="Every prediction made in the Prediction Centre, newest first."
      actions={<><button className="btn" onClick={exportCsv} disabled={!items.length}>Export CSV</button>
        <button className="btn btn-danger" onClick={clear} disabled={!h.data?.length}>Clear all</button></>}>
      <Card title={`${items.length} predictions`} extra={
        <div className="row">
          <input type="text" placeholder="Search machine or note" value={q} onChange={e => setQ(e.target.value)}
            style={{ border: "1px solid var(--line)", borderRadius: 8, padding: "6px 10px" }} aria-label="Search" />
          <select value={risk} onChange={e => setRisk(e.target.value as Risk | "")} aria-label="Risk filter">
            <option value="">All risk levels</option><option>High</option><option>Medium</option><option>Uncertain</option><option>Low</option>
          </select>
        </div>}>
        {h.error && <ErrorBox error={h.error} />}
        {h.loading && !h.data ? <Loading /> : items.length === 0 ? <p className="muted">No predictions match.</p> : (
          <div className="tablewrap">
            <table className="t">
              <thead><tr><th>When</th><th>Machine</th><th>Set</th><th>Type</th><th className="n">Air / Proc K</th><th className="n">rpm</th>
                <th className="n">Torque</th><th className="n">Wear</th><th className="n">Probability</th><th>Risk</th><th>Rules met</th><th>Note</th><th /></tr></thead>
              <tbody>{items.map(x => (
                <tr key={x.id}>
                  <td>{when(x.created_at)}</td><td>{x.machine_id || "–"}</td><td>{x.feature_set}</td><td>{x.input.type}</td>
                  <td className="n">{x.input.air_temperature.toFixed(1)} / {x.input.process_temperature.toFixed(1)}</td>
                  <td className="n">{x.input.rotational_speed}</td><td className="n">{x.input.torque.toFixed(1)}</td><td className="n">{x.input.tool_wear}</td>
                  <td className="n">{pct(x.probability)}</td>
                  <td><RiskBadge risk={x.risk} />{x.review_recommended && <span className="tag" style={{ marginLeft: 4 }}>review</span>}</td>
                  <td>{x.triggered_rules.map(r => <span key={r} className="tag tag-on">{r}</span>)}</td>
                  <td className="small">{x.note}</td>
                  <td><button className="btn btn-sm btn-danger" onClick={() => del(x.id)} aria-label="Delete">Delete</button></td>
                </tr>))}</tbody>
            </table>
          </div>)}
      </Card>
    </Page>
  );
}
