import { useRef, useState } from "react";
import { api, type BatchResult, type Risk } from "../api";
import { useApp } from "../state";
import { Card, ErrorBox, NeedsModels, Page, RISK_COLOR, RiskBadge, Stat, downloadText, int, num, pct, toCsv } from "../ui";

export default function Batch() {
  const { fs, health } = useApp();
  const [file, setFile] = useState<File | null>(null);
  const [res, setRes] = useState<BatchResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const [filter, setFilter] = useState<Risk | "all" | "review">("all");
  const input = useRef<HTMLInputElement>(null);
  const ready = health?.models?.[fs]?.ready;

  const run = async (f: File) => {
    setFile(f); setBusy(true); setErr(null); setRes(null);
    try { setRes(await api.batch(f, fs)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  const template = async () => {
    const r = await fetch("/api/batch/template");
    downloadText("maintenai_batch_template.csv", await r.text());
  };
  const rows = res?.rows.filter(r => filter === "all" ? true : filter === "review" ? r.review_recommended : r.risk === filter) ?? [];
  const sorted = [...rows].sort((a, b) => b.probability - a.probability);

  return (
    <Page title="Batch Scoring"
      subtitle="Upload a CSV of machine readings to score many machines at once. Results can be filtered, sorted by risk and downloaded."
      actions={<button className="btn" onClick={template}>Download CSV template</button>}>
      {health && !ready && <NeedsModels error={health.models[fs].error ?? ""} />}
      <Card>
        <div className={`drop ${over ? "over" : ""}`} onClick={() => input.current?.click()}
          onDragOver={e => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={e => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && ready) run(f); }}>
          <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={e => { const f = e.target.files?.[0]; if (f) run(f); e.target.value = ""; }} />
          <strong>{busy ? "Scoring…" : "Drop a CSV here or click to choose a file"}</strong>
          <div className="small muted" style={{ marginTop: 6 }}>
            Columns: type, air_temperature, process_temperature, rotational_speed, torque, tool_wear (the original AI4I column names also work).
            An optional <code>Machine failure</code> column adds accuracy metrics. Up to 20,000 rows. Feature set: <strong>{fs}</strong>.
          </div>
          {file && <div className="small" style={{ marginTop: 6 }}>Last file: {file.name}</div>}
        </div>
      </Card>
      {err && <div className="mt"><ErrorBox error={err} /></div>}
      {res && (
        <>
          <div className="grid g4 mt">
            <Stat label="Machines scored" value={int(res.summary.rows)} />
            <Stat label="Predicted failures" value={int(res.summary.predicted_failures)} tone={RISK_COLOR.High}
              hint={`threshold ${pct(res.summary.threshold)}`} />
            <Stat label="Review recommended" value={int(res.summary.review_recommended)} tone={RISK_COLOR.Uncertain} />
            <Stat label="Risk mix" value={<span style={{ fontSize: 15 }}>{(["High", "Medium", "Uncertain", "Low"] as Risk[]).map(r =>
              <span key={r} style={{ color: RISK_COLOR[r], marginRight: 10 }}>{r[0]} {res.summary.risk_counts[r] ?? 0}</span>)}</span>} />
          </div>
          {res.summary.evaluation && (
            <Card title="Accuracy against the labels in your file" className="mt">
              <div className="grid g4">
                {["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"].map(k =>
                  <Stat key={k} label={k.replace("_", "-").toUpperCase()} value={num(res.summary.evaluation![k], 3)} />)}
                <Stat label="Confusion" value={<span style={{ fontSize: 14 }}>TP {res.summary.evaluation.tp} · FP {res.summary.evaluation.fp} · FN {res.summary.evaluation.fn} · TN {res.summary.evaluation.tn}</span>} />
              </div>
              <p className="small muted">{res.summary.evaluation_note}</p>
            </Card>)}
          <Card className="mt" title={`Results (${rows.length} shown, highest risk first)`} extra={
            <div className="row">
              <select value={filter} onChange={e => setFilter(e.target.value as typeof filter)} aria-label="Filter">
                <option value="all">All risk levels</option><option value="High">High</option><option value="Medium">Medium</option>
                <option value="Uncertain">Uncertain</option><option value="Low">Low</option><option value="review">Review recommended</option>
              </select>
              <button className="btn btn-sm" onClick={() => downloadText(`maintenai_scored_${fs}.csv`, toCsv(sorted as unknown as Record<string, unknown>[]))}>Download results</button>
            </div>}>
            <div className="tablewrap">
              <table className="t">
                <thead><tr><th>ID</th><th>Type</th><th className="n">Air K</th><th className="n">Process K</th><th className="n">rpm</th>
                  <th className="n">Torque</th><th className="n">Wear</th><th className="n">Probability</th><th>Risk</th><th>Review</th></tr></thead>
                <tbody>{sorted.slice(0, 1000).map(r => (
                  <tr key={r.row}><td>{r.id}</td><td>{r.type}</td><td className="n">{r.air_temperature.toFixed(1)}</td>
                    <td className="n">{r.process_temperature.toFixed(1)}</td><td className="n">{r.rotational_speed}</td>
                    <td className="n">{r.torque.toFixed(1)}</td><td className="n">{r.tool_wear}</td>
                    <td className="n">{pct(r.probability)}</td><td><RiskBadge risk={r.risk} /></td><td>{r.review_recommended ? "yes" : ""}</td></tr>))}</tbody>
              </table>
            </div>
            {sorted.length > 1000 && <p className="small muted">Showing the first 1,000 rows; the download contains all of them.</p>}
          </Card>
        </>)}
    </Page>
  );
}
