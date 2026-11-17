import { NavLink, Route, Routes } from "react-router-dom";
import { useApp } from "./state";
import Dashboard from "./pages/Dashboard";
import DataExplorer from "./pages/DataExplorer";
import ModelLab from "./pages/ModelLab";
import Predict from "./pages/Predict";
import Explain from "./pages/Explain";
import Batch from "./pages/Batch";
import History from "./pages/History";
import Experiments from "./pages/Experiments";

const NAV = [
  { to: "/", label: "Dashboard", icon: "M3 12l9-8 9 8M5 10v10h14V10" },
  { to: "/predict", label: "Prediction Centre", icon: "M12 3v4M12 17v4M3 12h4M17 12h4M12 8a4 4 0 100 8 4 4 0 000-8z" },
  { to: "/batch", label: "Batch Scoring", icon: "M4 5h16M4 10h16M4 15h10M4 20h7" },
  { to: "/history", label: "History", icon: "M12 7v5l3 2M21 12a9 9 0 11-9-9 9 9 0 019 9z" },
  { to: "/explain", label: "Explainability", icon: "M9 18h6M10 21h4M12 3a6 6 0 00-3.5 10.9V16h7v-2.1A6 6 0 0012 3z" },
  { to: "/models", label: "Model Lab", icon: "M4 20V10M10 20V4M16 20v-7M22 20H2" },
  { to: "/data", label: "Data Explorer", icon: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" },
  { to: "/experiments", label: "Experiments", icon: "M9 3h6M10 3v6l-5 9a2 2 0 001.8 3h10.4a2 2 0 001.8-3l-5-9V3" },
];

function Icon({ d }: { d: string }) {
  return <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d={d} /></svg>;
}

export default function App() {
  const { fs, setFs, health, healthError } = useApp();
  const ready = health?.models?.[fs]?.ready;
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden><rect width="32" height="32" rx="7" fill="#0f766e" />
            <path d="M7 21l5-8 4 5 3-4 6 7" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
          <div><div className="brand-name">MaintenAI</div><div className="brand-sub">Predictive maintenance</div></div>
        </div>
        <nav>
          {NAV.map(n => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"} className={({ isActive }) => (isActive ? "nav active" : "nav")}>
              <Icon d={n.icon} /><span>{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="side-foot">
          <div className="fs-label">Feature set</div>
          <div className="seg" role="radiogroup" aria-label="Feature set">
            {(["engineered", "raw"] as const).map(f => (
              <button key={f} role="radio" aria-checked={fs === f} className={fs === f ? "seg-on" : ""} onClick={() => setFs(f)}>
                {f === "engineered" ? "Engineered" : "Raw"}
              </button>
            ))}
          </div>
          <div className="status">
            {healthError ? <><span className="dot dot-red" />API offline</> :
              !health ? <><span className="dot" />Connecting…</> :
                <><span className={`dot ${ready ? "dot-green" : "dot-amber"}`} />{ready ? "Models ready" : "Models not trained"}
                  <span className="muted"> · {health.store === "mongodb" ? "MongoDB" : "file storage"}</span></>}
          </div>
          {health?.job && <div className="status"><span className="dot dot-blue pulse" />Job running: {health.job.kind}</div>}
        </div>
      </aside>
      <main className="main">
        {healthError && <div className="alert alert-error">Cannot reach the API ({healthError}). Start the back end with <code>bash run_web.sh</code>.</div>}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/history" element={<History />} />
          <Route path="/explain" element={<Explain />} />
          <Route path="/models" element={<ModelLab />} />
          <Route path="/data" element={<DataExplorer />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
        <footer className="foot">
          MaintenAI is a decision-support research prototype trained on the synthetic AI4I 2020 dataset. It cannot guarantee
          that failures are prevented; final maintenance decisions rest with qualified engineers.
        </footer>
      </main>
    </div>
  );
}
