import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, type FeatureSet, type Health } from "./api";

interface AppState {
  fs: FeatureSet;
  setFs: (f: FeatureSet) => void;
  health: Health | null;
  healthError: string | null;
  refreshHealth: () => Promise<void>;
}

const Ctx = createContext<AppState | null>(null);

function initialFs(): FeatureSet {
  try {
    const v = localStorage.getItem("maintenai.fs");
    if (v === "raw" || v === "engineered") return v;
  } catch { /* storage unavailable */ }
  return "engineered";
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [fs, setFsState] = useState<FeatureSet>(initialFs);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (e) {
      setHealthError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const t = setInterval(refreshHealth, 15000);
    return () => clearInterval(t);
  }, [refreshHealth]);

  const setFs = (f: FeatureSet) => {
    setFsState(f);
    try { localStorage.setItem("maintenai.fs", f); } catch { /* ignore */ }
  };

  return <Ctx.Provider value={{ fs, setFs, health, healthError, refreshHealth }}>{children}</Ctx.Provider>;
}

export function useApp() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useApp outside provider");
  return c;
}

/** Small data-loading hook: re-runs when deps change; exposes loading, error and a reload function. */
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    fn().then(d => { if (alive) { setData(d); setError(null); } })
      .catch(e => { if (alive) setError((e as Error).message); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { data, error, loading, reload: () => setTick(t => t + 1) };
}
