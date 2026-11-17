import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Metrics } from "../api";
import { useApp, useLoad } from "../state";
import { C, Card, ErrorBox, Loading, Notice, Page, Tabs, num, pct } from "../ui";

type T = "compare" | "live" | "bma" | "imbalance" | "calibration" | "uncertainty" | "ablation" | "general" | "figures";
const BASE = ["CART", "AdaBoost", "Gradient Boosting", "XGBoost", "MARS", "GAM", "Random Forest"];
const ENS = ["Soft Voting", "Stacking", "BMA"];

export default function ModelLab() {
  const { fs, health } = useApp();
  const res = useLoad(api.results, [health?.results]);
  const [tab, setTab] = useState<T>("compare");
  const r = res.data;
  return (
    <Page title="Model Lab" subtitle={<>Results of the eight research experiments (single stratified 70/15/15 split, seed 42; test set of 1,500 machines with 51 failures). Showing the <strong>{fs}</strong> feature set where relevant.</>}>
      <Tabs<T> value={tab} onChange={setTab} tabs={[
        { id: "compare", label: "Comparison" }, { id: "live", label: "Deployed models" }, { id: "bma", label: "Bayesian averaging" },
        { id: "imbalance", label: "Imbalance" }, { id: "calibration", label: "Calibration" }, { id: "uncertainty", label: "Uncertainty" },
        { id: "ablation", label: "Ablation" }, { id: "general", label: "Generalisation" }, { id: "figures", label: "Figures" }]} />
      {tab === "live" ? <Live /> : tab === "figures" ? <Figures /> :
        res.error ? <Notice tone="warn">{res.error}</Notice> : !r ? <Loading /> : (
          <>
            {tab === "compare" && <Compare r={r} fs={fs} />}
            {tab === "bma" && <Bma r={r} fs={fs} />}
            {tab === "imbalance" && <Imbalance r={r} />}
            {tab === "calibration" && <Calibration r={r} fs={fs} />}
            {tab === "uncertainty" && <Uncertainty r={r} fs={fs} />}
            {tab === "ablation" && <MetricTable title="Ablation: adding IEMAF components one at a time" rows={r.exp7[fs]} />}
            {tab === "general" && <General r={r} fs={fs} />}
          </>)}
    </Page>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type R = Record<string, any>;

function Compare({ r, fs }: { r: R; fs: string }) {
  const rows = [...BASE.map(n => ({ n, m: r.exp1[fs][n] as Metrics & { cv_pr_auc_mean: number; cv_pr_auc_sd: number }, ens: false })),
    ...ENS.map(n => ({ n, m: r.exp2[fs][n], ens: true }))];
  const best = rows.reduce((a, b) => (b.m.pr_auc > a.m.pr_auc ? b : a));
  const chart = rows.map(({ n, m }) => {
    const ci = r.ci?.[fs]?.[n]?.pr_auc_ci as [number, number] | undefined;
    return { n, pr: m.pr_auc, err: ci ? [m.pr_auc - ci[0], ci[1] - m.pr_auc] : [0, 0] };
  });
  return (
    <div className="grid">
      <Card title="Test PR-AUC with 95% bootstrap intervals">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chart} margin={{ bottom: 30 }}>
            <CartesianGrid vertical={false} stroke="#eef2f6" />
            <XAxis dataKey="n" angle={-25} textAnchor="end" interval={0} tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v: number | number[]) => (Array.isArray(v) ? "" : v.toFixed(3))} />
            <Bar dataKey="pr" name="PR-AUC" radius={[4, 4, 0, 0]}>
              {chart.map(d => <Cell key={d.n} fill={ENS.includes(d.n) ? C.accent : C.blue} />)}
              <ErrorBar dataKey="err" width={5} stroke="#0f172a" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="legend"><span><i style={{ background: C.blue }} />single model</span><span><i style={{ background: C.accent }} />ensemble</span>
          <span className="muted">Intervals that overlap mean the difference may be chance.</span></div>
      </Card>
      <Card title="All metrics on the test split (thresholds chosen on validation)">
        <div className="tablewrap">
          <table className="t">
            <thead><tr><th>Model</th><th className="n">CV PR-AUC</th><th className="n">PR-AUC</th><th className="n">ROC-AUC</th><th className="n">Recall</th>
              <th className="n">Precision</th><th className="n">F1</th><th className="n">Brier</th><th className="n">ECE</th><th className="n">TP / FP / FN</th></tr></thead>
            <tbody>{rows.map(({ n, m, ens }) => (
              <tr key={n} className={n === best.n ? "best" : ""}>
                <td>{ens ? <strong>{n}</strong> : n}</td>
                <td className="n">{"cv_pr_auc_mean" in m ? `${num(m.cv_pr_auc_mean)} ± ${num(m.cv_pr_auc_sd, 3)}` : "–"}</td>
                <td className="n">{num(m.pr_auc)}</td><td className="n">{num(m.roc_auc)}</td><td className="n">{num(m.recall)}</td>
                <td className="n">{num(m.precision)}</td><td className="n">{num(m.f1)}</td><td className="n">{num(m.brier, 4)}</td>
                <td className="n">{num(m.ece, 4)}</td><td className="n">{m.tp} / {m.fp} / {m.fn}</td></tr>))}</tbody>
          </table>
        </div>
        <p className="small muted">Highlighted: highest test PR-AUC. Best single model by cross-validation: {r.exp2[fs].best_individual_by_cv}.
          Tuned CART: {JSON.stringify(r.tuning[fs].cart_best)}; tuned XGBoost: {JSON.stringify(r.tuning[fs].xgb_best)}.</p>
      </Card>
    </div>
  );
}

function Live() {
  const { fs, health } = useApp();
  const ready = health?.models?.[fs]?.ready;
  const m = useLoad(() => (ready ? api.models(fs) : Promise.resolve(null)), [fs, ready, health?.models?.[fs]?.trained_at]);
  if (!ready) return <Notice tone="warn">The {fs} models are not trained yet. Use the Experiments page.</Notice>;
  if (m.error) return <ErrorBox error={m.error} />;
  if (!m.data) return <Loading />;
  const d = m.data;
  return (
    <div className="grid">
      <Card title={`Deployed ${fs} bundle`}>
        <dl className="kv">
          <dt>Trained</dt><dd>{new Date(d.trained_at).toLocaleString()} ({d.train_seconds} s)</dd>
          <dt>Libraries</dt><dd>{Object.entries(d.versions).map(([k, v]) => `${k} ${v}`).join(" · ")}</dd>
          <dt>Split</dt><dd>{Object.entries(d.split).map(([k, v]) => `${k} ${v.rows} (${v.failures} failures)`).join(" · ")}</dd>
          <dt>Tuning</dt><dd>{d.tuning.note}</dd>
          <dt>Decision threshold</dt><dd>{pct(d.thresholds.IEMAF, 1)} on calibrated BMA probability</dd>
          <dt>Review policy</dt><dd>entropy ≥ {num(d.tau_entropy, 3)} bits or disagreement ≥ {num(d.tau_disagree, 3)} (90th percentiles on validation)</dd>
          <dt>Review on test split</dt><dd>{pct(d.deferral_test.deferred_share)} of machines flagged, holding {d.deferral_test.failures_deferred} of {d.deferral_test.failures_total} failures;
            error rate on the rest {pct(d.deferral_test.error_rate_kept, 2)} vs {pct(d.deferral_test.error_rate_all, 2)} overall</dd>
        </dl>
      </Card>
      <MetricTable title="Test metrics of the deployed models" rows={d.metrics_test} />
    </div>
  );
}

function MetricTable({ title, rows }: { title: string; rows: Record<string, Metrics> }) {
  return (
    <Card title={title}>
      <div className="tablewrap">
        <table className="t">
          <thead><tr><th>Configuration</th><th className="n">PR-AUC</th><th className="n">ROC-AUC</th><th className="n">Recall</th><th className="n">Precision</th>
            <th className="n">F1</th><th className="n">ECE</th><th className="n">Threshold</th></tr></thead>
          <tbody>{Object.entries(rows).map(([k, m]) => (
            <tr key={k}><td>{k}</td><td className="n">{num(m.pr_auc)}</td><td className="n">{num(m.roc_auc)}</td><td className="n">{num(m.recall)}</td>
              <td className="n">{num(m.precision)}</td><td className="n">{num(m.f1)}</td><td className="n">{num(m.ece, 4)}</td><td className="n">{num(m.threshold)}</td></tr>))}</tbody>
        </table>
      </div>
    </Card>
  );
}

function Bma({ r, fs }: { r: R; fs: string }) {
  const b = r.bma[fs];
  const data = BASE.map(n => ({ n, w: b.weights[n], ll: b.oof_loglik[n], t10: b.tempered["10"].weights[n], t100: b.tempered["100"].weights[n] }));
  return (
    <div className="grid g2">
      <Card title="Posterior model probabilities">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data} margin={{ bottom: 30 }}>
            <CartesianGrid vertical={false} stroke="#eef2f6" />
            <XAxis dataKey="n" angle={-25} textAnchor="end" interval={0} tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v: number) => v.toExponential(2)} />
            <Legend verticalAlign="top" />
            <Bar dataKey="w" name="BMA (T = 1)" fill={C.accent} />
            <Bar dataKey="t10" name="tempered T = 10" fill={C.blue} />
            <Bar dataKey="t100" name="tempered T = 100" fill={C.gray} />
          </BarChart>
        </ResponsiveContainer>
        <p className="small muted">Weights ∝ exp(out-of-fold log-likelihood) with a uniform prior. With 7,000 training rows the posterior concentrates on one model,
          as Minka (2000) predicts. Tempered weights are a sensitivity check only, not BMA.</p>
      </Card>
      <Card title="Out-of-fold log-likelihood (evidence)">
        <table className="t">
          <thead><tr><th>Model</th><th className="n">log-likelihood</th><th className="n">BMA weight</th></tr></thead>
          <tbody>{[...data].sort((a, c) => c.ll - a.ll).map(d => (
            <tr key={d.n}><td>{d.n}</td><td className="n">{d.ll.toFixed(1)}</td><td className="n">{d.w > 1e-4 ? d.w.toFixed(4) : d.w.toExponential(1)}</td></tr>))}</tbody>
        </table>
        <p className="small muted">Test PR-AUC of tempered averaging: T = 10 → {num(b.tempered["10"].test.pr_auc)}, T = 100 → {num(b.tempered["100"].test.pr_auc)}.</p>
      </Card>
    </div>
  );
}

function Imbalance({ r }: { r: R }) {
  return (
    <div className="grid g2">
      {Object.entries(r.exp3 as R).map(([model, strat]) => (
        <Card key={model} title={`${model} (raw features)`}>
          <table className="t">
            <thead><tr><th>Strategy</th><th className="n">Recall @0.5</th><th className="n">Precision @0.5</th><th className="n">Recall @tuned</th>
              <th className="n">Precision @tuned</th><th className="n">PR-AUC</th></tr></thead>
            <tbody>{Object.entries(strat as R).map(([s, v]) => (
              <tr key={s}><td>{s}</td><td className="n">{num(v["at_0.5"].recall)}</td><td className="n">{num(v["at_0.5"].precision)}</td>
                <td className="n">{num(v.tuned_threshold.recall)}</td><td className="n">{num(v.tuned_threshold.precision)}</td><td className="n">{num(v.tuned_threshold.pr_auc)}</td></tr>))}</tbody>
          </table>
        </Card>))}
      <Card title="Finding" className="span2"><p className="small" style={{ margin: 0 }}>Resampling mainly moves the operating point. Choosing the threshold on validation data
        gives most of the benefit; SMOTE gives the lowest PR-AUC for XGBoost.</p></Card>
    </div>
  );
}

function Calibration({ r, fs }: { r: R; fs: string }) {
  const rows = Object.entries(r.exp5[fs] as R);
  return (
    <div className="grid g2">
      <Card title="Expected calibration error before and after Platt scaling">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={rows.map(([n, v]) => ({ n, before: v.ece, after: v.ece_cal }))} margin={{ bottom: 20 }}>
            <CartesianGrid vertical={false} stroke="#eef2f6" />
            <XAxis dataKey="n" angle={-20} textAnchor="end" interval={0} tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={v => Number(v).toFixed(3)} />
            <Tooltip formatter={(v: number) => v.toFixed(4)} /><Legend verticalAlign="top" />
            <Bar dataKey="before" name="uncalibrated" fill={C.blue} /><Bar dataKey="after" name="Platt (validation)" fill={C.orange} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Brier score and ECE">
        <table className="t"><thead><tr><th>Model</th><th className="n">Brier</th><th className="n">ECE</th><th className="n">Brier cal.</th><th className="n">ECE cal.</th></tr></thead>
          <tbody>{rows.map(([n, v]) => <tr key={n}><td>{n}</td><td className="n">{num(v.brier, 4)}</td><td className="n">{num(v.ece, 4)}</td>
            <td className="n">{num(v.brier_cal, 4)}</td><td className="n">{num(v.ece_cal, 4)}</td></tr>)}</tbody></table>
        <p className="small muted">The base models are already well calibrated; fitting Platt scaling on only 51 validation failures slightly worsened ECE.</p>
      </Card>
    </div>
  );
}

function Uncertainty({ r, fs }: { r: R; fs: string }) {
  const u = r.exp6[fs];
  const data = u.coverage.entropy.map((e: R, i: number) => ({ cov: e.coverage * 100, entropy: e.error_rate * 100,
    disagreement: u.coverage.disagreement[i].error_rate * 100, deferred: e.failures_deferred }));
  return (
    <div className="grid g2">
      <Card title="Error rate on kept cases vs coverage">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data}>
            <CartesianGrid stroke="#eef2f6" />
            <XAxis dataKey="cov" type="number" domain={[80, 100]} reversed ticks={[100, 95, 90, 85, 80]} tickFormatter={v => `${v}% kept`} /><YAxis tickFormatter={v => `${Number(v).toFixed(1)}%`} />
            <Tooltip formatter={(v: number) => `${v.toFixed(2)}%`} /><Legend verticalAlign="top" />
            <Line dataKey="entropy" name="defer by entropy" stroke={C.accent} strokeWidth={2} />
            <Line dataKey="disagreement" name="defer by disagreement" stroke={C.orange} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Deferral table (entropy)">
        <table className="t"><thead><tr><th>Kept</th><th className="n">Deferred</th><th className="n">Failures deferred</th><th className="n">Error on kept</th><th className="n">Missed kept</th></tr></thead>
          <tbody>{u.coverage.entropy.map((e: R) => <tr key={e.coverage}><td>{pct(e.coverage, 0)}</td><td className="n">{e.deferred}</td>
            <td className="n">{e.failures_deferred}</td><td className="n">{pct(e.error_rate, 2)}</td><td className="n">{e.missed_failures_kept}</td></tr>)}</tbody></table>
        <p className="small muted">{u.n_wrong} wrong predictions; mean entropy {num(u.mean_entropy_wrong, 3)} bits when wrong vs {num(u.mean_entropy_correct, 3)} when correct.</p>
      </Card>
    </div>
  );
}

function General({ r, fs }: { r: R; fs: string }) {
  const g = r.exp8[fs] as R;
  const models = Object.keys(g.L);
  const data = Object.entries(g).map(([t, v]) => ({ t: `${t} held out`, ...Object.fromEntries(models.map(m => [m, (v as R)[m].pr_auc])) }));
  return (
    <Card title="Leave-one-product-type-out: PR-AUC on the unseen type">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid vertical={false} stroke="#eef2f6" />
          <XAxis dataKey="t" /><YAxis domain={[0, 1]} /><Tooltip formatter={(v: number) => v.toFixed(3)} /><Legend verticalAlign="top" />
          {models.map((m, i) => <Bar key={m} dataKey={m} fill={C.series[i]} />)}
        </BarChart>
      </ResponsiveContainer>
      <p className="small muted">Product IDs are unique per row, so holding out a whole product type is the closest proxy for an unseen machine family.
        Scores fall well below the in-distribution test results.</p>
    </Card>
  );
}

function Figures() {
  const f = useLoad(api.figures, []);
  const [open, setOpen] = useState<string | null>(null);
  if (!f.data) return <Loading />;
  if (!f.data.length) return <Notice tone="warn">No figures yet; run the experiments.</Notice>;
  return (
    <>
      <div className="figs">{f.data.map(n => (
        <figure key={n}><img src={`/api/figures/${n}`} alt={n} loading="lazy" onClick={() => setOpen(n)} />
          <figcaption>{n.replace(".png", "").replace(/_/g, " ")}</figcaption></figure>))}</div>
      {open && <div className="lightbox" onClick={() => setOpen(null)}><img src={`/api/figures/${open}`} alt={open} /></div>}
    </>
  );
}
