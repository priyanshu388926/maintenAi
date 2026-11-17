import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Prediction, type Reading } from "../api";
import { useApp } from "../state";
import { C, Card, ErrorBox, NeedsModels, Page, ProbBar, RISK_COLOR, RiskBadge, num, pct } from "../ui";

const FIELDS: { key: keyof Omit<Reading, "type">; label: string; unit: string; min: number; max: number; step: number }[] = [
  { key: "air_temperature", label: "Air temperature", unit: "K", min: 295, max: 305, step: 0.1 },
  { key: "process_temperature", label: "Process temperature", unit: "K", min: 305, max: 314, step: 0.1 },
  { key: "rotational_speed", label: "Rotational speed", unit: "rpm", min: 1150, max: 2900, step: 1 },
  { key: "torque", label: "Torque", unit: "Nm", min: 3, max: 77, step: 0.1 },
  { key: "tool_wear", label: "Tool wear", unit: "min", min: 0, max: 255, step: 1 },
];

const PRESETS: { name: string; v: Reading }[] = [
  { name: "Normal", v: { type: "M", air_temperature: 298.1, process_temperature: 308.6, rotational_speed: 1551, torque: 42.8, tool_wear: 0 } },
  { name: "Heat dissipation", v: { type: "L", air_temperature: 302.5, process_temperature: 310.5, rotational_speed: 1300, torque: 60, tool_wear: 120 } },
  { name: "Low power", v: { type: "M", air_temperature: 299.0, process_temperature: 309.5, rotational_speed: 2700, torque: 9, tool_wear: 50 } },
  { name: "High power", v: { type: "H", air_temperature: 300.0, process_temperature: 310.0, rotational_speed: 1250, torque: 72, tool_wear: 80 } },
  { name: "Overstrain", v: { type: "L", air_temperature: 300.0, process_temperature: 310.0, rotational_speed: 1500, torque: 50, tool_wear: 230 } },
  { name: "Worn tool", v: { type: "H", air_temperature: 299.0, process_temperature: 309.0, rotational_speed: 1600, torque: 38, tool_wear: 225 } },
];

export default function Predict() {
  const { fs, health } = useApp();
  const [v, setV] = useState<Reading>(PRESETS[0].v);
  const [machine, setMachine] = useState("");
  const [note, setNote] = useState("");
  const [res, setRes] = useState<Prediction | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ready = health?.models?.[fs]?.ready;

  const run = async () => {
    setBusy(true); setErr(null);
    try {
      setRes(await api.predict({ ...v, feature_set: fs, machine_id: machine || undefined, note: note || undefined }));
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };

  return (
    <Page title="Prediction Centre"
      subtitle="Enter one machine's operating readings. Seven models, three ensembles and the calibrated BMA estimate the failure probability, explain it and flag cases for engineer review.">
      {health && !ready && <NeedsModels error={health.models[fs].error ?? ""} />}
      <div className="grid g-side">
        <Card title="Machine readings">
          <div className="presets">{PRESETS.map(p => <button key={p.name} className="chip" onClick={() => setV(p.v)}>{p.name}</button>)}</div>
          <div className="field">
            <label>Product type <span>quality variant</span></label>
            <div className="typepick">{(["L", "M", "H"] as const).map(t =>
              <button key={t} className={v.type === t ? "on" : ""} onClick={() => setV({ ...v, type: t })}>{t}</button>)}</div>
          </div>
          {FIELDS.map(f => (
            <div className="field" key={f.key}>
              <label htmlFor={f.key}>{f.label} <span>{f.unit}</span></label>
              <div className="row">
                <input type="range" min={f.min} max={f.max} step={f.step} value={v[f.key]} style={{ flex: 1 }}
                  onChange={e => setV({ ...v, [f.key]: Number(e.target.value) })} aria-label={f.label} />
                <input id={f.key} type="number" step={f.step} value={v[f.key]} style={{ width: 92 }}
                  onChange={e => setV({ ...v, [f.key]: Number(e.target.value) })} />
              </div>
            </div>))}
          <div className="grid g2" style={{ gap: 10 }}>
            <div className="field"><label htmlFor="mid">Machine ID <span>optional</span></label>
              <input id="mid" type="text" value={machine} maxLength={60} onChange={e => setMachine(e.target.value)} placeholder="e.g. Mill-07" /></div>
            <div className="field"><label htmlFor="note">Note <span>optional</span></label>
              <input id="note" type="text" value={note} maxLength={300} onChange={e => setNote(e.target.value)} placeholder="shift, operator…" /></div>
          </div>
          <button className="btn btn-primary" style={{ width: "100%", justifyContent: "center" }} disabled={busy || !ready} onClick={run}>
            {busy ? "Predicting…" : "Predict failure risk"}</button>
          <p className="small muted" style={{ marginBottom: 0 }}>Model: {fs} feature set. Each prediction is saved to History.</p>
        </Card>

        <div>
          {err && <ErrorBox error={err} />}
          {!res ? <Card><p className="muted">Choose a preset or adjust the readings, then press <strong>Predict failure risk</strong>.</p></Card> : <Result r={res} />}
        </div>
      </div>
    </Page>
  );
}

function Result({ r }: { r: Prediction }) {
  const shap = r.shap.contributions.slice(0, 8).map(c => ({ ...c, label: `${c.feature} = ${Number.isInteger(c.value) ? c.value : c.value.toFixed(1)}` }));
  return (
    <>
      <Card>
        <div className="result-hero">
          <div>
            <div className="muted small">Calibrated failure probability</div>
            <div className="gauge-num" style={{ color: RISK_COLOR[r.risk] }}>{pct(r.probability, 1)}</div>
          </div>
          <div><div className="muted small">Risk level</div><RiskBadge risk={r.risk} big /></div>
          <div style={{ flex: 1, minWidth: 240 }}>
            <p style={{ margin: "0 0 6px", fontWeight: 600 }}>{r.advice}</p>
            <ProbBar value={r.probability} threshold={r.threshold} color={RISK_COLOR[r.risk]} />
            <div className="small muted" style={{ marginTop: 4 }}>Black line: decision threshold {pct(r.threshold)} (chosen on the validation split).</div>
          </div>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          {r.review_recommended && <span className="tag tag-on">Engineer review recommended</span>}
          {r.outside_training_range && <span className="tag tag-on">Outside the training data range</span>}
          <span className="tag">entropy {num(r.entropy, 2)} bits (review ≥ {num(r.tau_entropy, 2)})</span>
          <span className="tag">model disagreement {num(r.disagreement, 3)} (review ≥ {num(r.tau_disagree, 3)})</span>
          <span className="tag">power {Math.round(r.derived.power_w).toLocaleString()} W</span>
          <span className="tag">ΔT {r.derived.temperature_difference.toFixed(1)} K</span>
          <span className="tag">wear × torque {Math.round(r.derived.wear_torque).toLocaleString()}</span>
        </div>
      </Card>

      <div className="grid g2 mt">
        <Card title="Every model's probability">
          {r.models.map(m => (
            <div className="modelrow" key={m.model} title={`threshold ${pct(m.threshold)} · BMA weight ${m.weight_bma.toExponential(1)}`}>
              <span>{m.model}</span><ProbBar value={m.probability} threshold={m.threshold} color={m.probability >= m.threshold ? C.high : C.gray} />
              <span className="n">{pct(m.probability)}</span></div>))}
          <hr style={{ border: 0, borderTop: "1px solid #eef2f6" }} />
          {r.ensembles.map(m => (
            <div className="modelrow" key={m.model}>
              <strong style={{ fontSize: 12.5 }}>{m.model}</strong>
              <ProbBar value={m.probability} threshold={m.threshold} color={m.probability >= m.threshold ? C.high : C.accent} />
              <span className="n">{pct(m.probability)}</span></div>))}
          <p className="small muted">BMA weights come from cross-validated likelihood; on this data they concentrate on Gradient Boosting.</p>
        </Card>
        <Card title="Documented failure rules (dataset generator)">
          <ul className="rules">{r.rules.map(x => (
            <li key={x.mode}><span className="mark" style={{ background: x.triggered ? C.high : C.low }}>{x.triggered ? "!" : "✓"}</span>
              <div><strong>{x.name} ({x.mode})</strong> {x.triggered ? <span style={{ color: C.high }}>condition met</span> : <span className="muted">not met</span>}
                <div className="small muted">{x.detail}</div></div></li>))}</ul>
          <p className="small muted">These are the threshold rules used to generate the synthetic data, shown so you can check the model's reasoning.</p>
        </Card>
      </div>

      <div className="grid g2 mt">
        <Card title="Why? SHAP contributions (XGBoost)">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={shap} layout="vertical" margin={{ left: 0, right: 12 }}>
              <CartesianGrid horizontal={false} stroke="#eef2f6" />
              <XAxis type="number" tickFormatter={v => v.toFixed(1)} />
              <YAxis type="category" dataKey="label" width={200} tick={{ fontSize: 11 }} />
              <ReferenceLine x={0} stroke="#475569" />
              <Tooltip formatter={(x: number) => [x.toFixed(3), "log-odds contribution"]} />
              <Bar dataKey="shap">{shap.map(d => <Cell key={d.feature} fill={d.shap > 0 ? C.high : C.blue} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="small muted">Red pushes towards failure, blue away from it (log-odds). Base value {num(r.shap.base_logit, 2)}; XGBoost output {pct(r.shap.output_probability)}.</p>
        </Card>
        <Card title="Decision-tree path (CART)">
          <ol className="path">{r.cart_path.steps.map((s, i) => <li key={i}>{s}</li>)}</ol>
          <p className="small" style={{ marginBottom: 0 }}>Leaf: {r.cart_path.leaf_samples} training machines, of which <strong>{pct(r.cart_path.leaf_failure_share)}</strong> failed.</p>
        </Card>
      </div>
    </>
  );
}
