import { useEffect, useRef, useState } from "react";
import { api, type FeatureSet, type Job } from "../api";
import { useApp, useLoad } from "../state";
import { Card, ErrorBox, Notice, Page, when } from "../ui";

export default function Experiments() {
  const { health, refreshHealth } = useApp();
  const jobs = useLoad(api.jobs, []);
  const [active, setActive] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sets, setSets] = useState<FeatureSet[]>(["engineered", "raw"]);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => { if (!active && jobs.data?.length) setActive(jobs.data[0]); }, [jobs.data, active]);
  useEffect(() => {
    if (!active || !["queued", "running"].includes(active.state)) return;
    const t = setInterval(async () => {
      const j = await api.job(active.id);
      setActive(j);
      if (!["queued", "running"].includes(j.state)) { jobs.reload(); refreshHealth(); }
    }, 1500);
    return () => clearInterval(t);
  }, [active, jobs, refreshHealth]);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [active?.log.length]);

  const start = async (fn: () => Promise<Job>) => {
    setErr(null);
    try { const j = await fn(); setActive(j); jobs.reload(); refreshHealth(); } catch (e) { setErr((e as Error).message); }
  };
  const running = !!health?.job || (active && ["queued", "running"].includes(active.state));
  const toggle = (f: FeatureSet) => setSets(s => (s.includes(f) ? s.filter(x => x !== f) : [...s, f]));

  return (
    <Page title="Experiments" subtitle="Retrain the deployed models or re-run all eight research experiments. Jobs run in the background on this computer.">
      {err && <ErrorBox error={err} />}
      <div className="grid g2">
        <Card title="Retrain deployed models">
          <p className="small">Fits the seven base models, voting, stacking and BMA, calibrates on the validation split and saves a verified bundle.
            About 1 minute per feature set. Predictions use the new models as soon as the job finishes.</p>
          <div className="row" style={{ marginBottom: 12 }}>
            {(["engineered", "raw"] as FeatureSet[]).map(f => (
              <label key={f} className="small"><input type="checkbox" checked={sets.includes(f)} onChange={() => toggle(f)} /> {f}
                {health?.models?.[f]?.ready ? <span className="muted"> (trained {when(health.models[f].trained_at!)})</span> : <span style={{ color: "#b45309" }}> (not trained)</span>}</label>))}
          </div>
          <button className="btn btn-primary" disabled={!!running || !sets.length} onClick={() => start(() => api.train(sets))}>Train models</button>
        </Card>
        <Card title="Re-run research experiments">
          <p className="small">Runs <code>python -m ml.experiments.run_all</code>: tuning, experiments 1–8, bootstrap intervals and all figures.
            Takes about 5 minutes and rewrites <code>artifacts/metrics/results.json</code> and <code>reports/figures/</code>.</p>
          <Notice>Results status: {health?.results ? "results.json present" : "no results yet"}.</Notice>
          <button className="btn" disabled={!!running} onClick={() => start(api.runExperiments)}>Run all experiments</button>
        </Card>
      </div>
      <Card className="mt" title={active ? `Job ${active.id}: ${active.kind} · ${active.state}` : "Job log"}
        extra={active?.started_at && <span className="small muted">started {when(active.started_at)}{active.finished_at ? ` · finished ${when(active.finished_at)}` : ""}</span>}>
        {active?.error && <ErrorBox error={active.error} />}
        <pre className="log" ref={logRef}>{active ? (active.log.join("\n") || "Starting…") : "No jobs yet in this session."}</pre>
      </Card>
      {jobs.data && jobs.data.length > 1 && (
        <Card className="mt" title="Earlier jobs">
          <table className="t"><thead><tr><th>Job</th><th>Kind</th><th>State</th><th>Started</th><th>Finished</th><th /></tr></thead>
            <tbody>{jobs.data.map(j => <tr key={j.id}><td>{j.id}</td><td>{j.kind}</td><td>{j.state}</td><td>{j.started_at ? when(j.started_at) : ""}</td>
              <td>{j.finished_at ? when(j.finished_at) : ""}</td><td><button className="btn btn-sm" onClick={() => setActive(j)}>View log</button></td></tr>)}</tbody></table>
        </Card>)}
    </Page>
  );
}
