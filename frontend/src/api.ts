// Typed client for the MaintenAI FastAPI back end. All URLs are relative (/api/...).

export type FeatureSet = "engineered" | "raw";
export type Risk = "Low" | "Medium" | "High" | "Uncertain";

export interface Metrics {
  threshold: number; accuracy: number; precision: number; recall: number; f1: number; macro_f1: number;
  roc_auc: number; pr_auc: number; brier: number; ece: number; fnr: number; fpr: number;
  tp: number; fp: number; fn: number; tn: number;
}

export interface Health {
  status: string; store: "file" | "mongodb"; store_warning: string | null;
  models: Record<FeatureSet, { ready: boolean; trained_at?: string; error?: string }>;
  dataset: boolean; dataset_error: string | null; results: boolean; history_count: number; job: Job | null;
}

export interface DatasetSummary {
  rows: number; columns: number; failures: number; failure_rate: number;
  types: { type: string; rows: number; failures: number; failure_rate: number }[];
  modes: { mode: string; name: string; count: number }[];
  stats: { feature: string; key: string; mean: number; std: number; min: number; median: number; max: number;
           mean_failure: number; mean_normal: number }[];
  quirks: { failures_without_mode: number; rnf_not_failure: number; multi_mode: number };
  synthetic_note: string;
}

export interface Histogram { feature: string; edges: number[]; normal: number[]; failure: number[]; failure_rate: number[] }
export interface ScatterPt { x: number; y: number; mode: string }
export interface Scatter { x: string; y: string; normal: ScatterPt[]; failure: ScatterPt[] }

export interface ModelInfo {
  feature_set: FeatureSet; trained_at: string; train_seconds: number; versions: Record<string, string>;
  thresholds: Record<string, number>; tau_entropy: number; tau_disagree: number;
  deferral_test: { deferred_share: number; failures_deferred: number; failures_total: number;
                   error_rate_kept: number; error_rate_all: number };
  metrics_test: Record<string, Metrics>; bma_weights: Record<string, number>; bma_loglik: Record<string, number>;
  split: Record<string, { rows: number; failures: number }>;
  tuning: { cart: Record<string, unknown> | null; xgb: Record<string, unknown> | null; note: string };
  input_ranges: Record<string, [number, number]>;
}

export interface Reading {
  type: "L" | "M" | "H"; air_temperature: number; process_temperature: number;
  rotational_speed: number; torque: number; tool_wear: number;
}

export interface Prediction {
  feature_set: FeatureSet; input: Reading;
  derived: { temperature_difference: number; power_w: number; wear_torque: number };
  probability: number; threshold: number; predicted_failure: boolean; risk: Risk; advice: string;
  entropy: number; disagreement: number; tau_entropy: number; tau_disagree: number; uncertain_bits: number;
  outside_training_range: boolean; review_recommended: boolean;
  models: { model: string; probability: number; threshold: number; weight_bma: number }[];
  ensembles: { model: string; probability: number; threshold: number }[];
  shap: { model: string; base_logit: number; output_logit: number; output_probability: number;
          contributions: { feature: string; value: number; shap: number }[] };
  cart_path: { steps: string[]; leaf_samples: number; leaf_failure_share: number };
  rules: { mode: string; name: string; triggered: boolean; detail: string }[];
  history_id?: string; created_at?: string;
}

export interface HistoryItem {
  id: string; created_at: string; machine_id: string | null; note: string | null; feature_set: FeatureSet;
  input: Reading; probability: number; risk: Risk; predicted_failure: boolean; entropy: number;
  disagreement: number; review_recommended: boolean; triggered_rules: string[];
}

export interface BatchRow extends Reading {
  row: number; id: string; probability: number; risk: Risk; predicted_failure: number;
  entropy: number; disagreement: number; review_recommended: boolean;
}

export interface BatchResult {
  feature_set: FeatureSet;
  summary: { rows: number; risk_counts: Partial<Record<Risk, number>>; predicted_failures: number;
             review_recommended: number; threshold: number;
             evaluation?: Record<string, number>; evaluation_note?: string };
  rows: BatchRow[];
}

export interface GlobalExplain {
  shap_mean_abs: Record<string, number>; cart_rules: string; cart_importance: Record<string, number>;
  cart_depth: number; cart_leaves: number;
  gam_shapes: Record<string, { x: number[]; y: number[]; lo: number[]; hi: number[] }>;
  mars_terms: string[] | string; stacking_coef: Record<string, number>;
}

export interface Job {
  id: string; kind: "train" | "experiments"; params: Record<string, unknown>;
  state: "queued" | "running" | "succeeded" | "failed"; started_at: string | null; finished_at: string | null;
  error: string | null; log: string[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Results = Record<string, any>;

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") msg = body.detail;
      else if (Array.isArray(body.detail)) msg = body.detail.map((d: { loc: string[]; msg: string }) =>
        `${d.loc.slice(1).join(".")}: ${d.msg}`).join("; ");
    } catch { /* not JSON */ }
    throw new ApiError(res.status, msg);
  }
  const type = res.headers.get("content-type") || "";
  return (type.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/api/health"),
  summary: () => req<DatasetSummary>("/api/dataset/summary"),
  features: () => req<{ key: string; label: string }[]>("/api/dataset/features"),
  histogram: (feature: string, bins = 30) =>
    req<Histogram>(`/api/dataset/histogram?feature=${encodeURIComponent(feature)}&bins=${bins}`),
  scatter: (x: string, y: string) =>
    req<Scatter>(`/api/dataset/scatter?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}`),
  rows: (offset: number, limit: number, failuresOnly: boolean) =>
    req<{ total: number; offset: number; rows: Record<string, string | number>[] }>(
      `/api/dataset/rows?offset=${offset}&limit=${limit}&failures_only=${failuresOnly}`),
  results: () => req<Results>("/api/results"),
  figures: () => req<string[]>("/api/figures"),
  models: (fs: FeatureSet) => req<ModelInfo>(`/api/models?feature_set=${fs}`),
  explainGlobal: (fs: FeatureSet) => req<GlobalExplain>(`/api/explain/global?feature_set=${fs}`),
  predict: (body: Reading & { feature_set: FeatureSet; machine_id?: string; note?: string; save?: boolean }) =>
    req<Prediction>("/api/predict", json(body)),
  batch: (file: File, fs: FeatureSet) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("feature_set", fs);
    return req<BatchResult>("/api/batch", { method: "POST", body: fd });
  },
  history: (limit = 500, risk?: string) =>
    req<HistoryItem[]>(`/api/history?limit=${limit}${risk ? `&risk=${risk}` : ""}`),
  historyStats: () => req<{ total: number; risk_counts: Partial<Record<Risk, number>>; recent: HistoryItem[] }>(
    "/api/history/stats"),
  deleteHistory: (id: string) => req<{ deleted: number }>(`/api/history/${id}`, { method: "DELETE" }),
  clearHistory: () => req<{ deleted: number }>("/api/history", { method: "DELETE" }),
  train: (feature_sets: FeatureSet[]) => req<Job>("/api/jobs/train", json({ feature_sets })),
  runExperiments: () => req<Job>("/api/jobs/experiments", { method: "POST" }),
  jobs: () => req<Job[]>("/api/jobs"),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
};
