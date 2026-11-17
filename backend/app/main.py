"""MaintenAI REST API (FastAPI). Serves the built React front end from frontend/dist when it exists.

    uvicorn app.main:app --app-dir backend --reload        (from the project root)
"""
from __future__ import annotations

import io
import json
import re
import sys
import threading
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config

sys.path.insert(0, str(config.ROOT))  # make the `ml` package importable

from ml.data import schema as S  # noqa: E402
from ml.evaluation.metrics import evaluate  # noqa: E402
from ml.inference.predictor import FIELDS, BundleError, Predictor, to_frame  # noqa: E402
from app import data_service as DS  # noqa: E402
from app.jobs import JobManager  # noqa: E402
from app.store import make_store  # noqa: E402

FeatureSet = Literal["engineered", "raw"]
app = FastAPI(title="MaintenAI API", version="1.0.0",
              description="Predictive maintenance on AI4I 2020: predictions, explanations, history and experiments.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

STORE, STORE_WARNING = make_store()
_PRED: dict[str, Predictor] = {}
_PRED_ERR: dict[str, str] = {}
_plock = threading.Lock()


def load_predictors():
    with _plock:
        for fs in ("engineered", "raw"):
            try:
                _PRED[fs] = Predictor(fs)
                _PRED_ERR.pop(fs, None)
            except BundleError as e:
                _PRED.pop(fs, None)
                _PRED_ERR[fs] = str(e)


JOBS = JobManager(on_trained=load_predictors)
load_predictors()


def predictor(fs: str) -> Predictor:
    p = _PRED.get(fs)
    if p is None:
        raise HTTPException(503, _PRED_ERR.get(fs, f"model '{fs}' not available"))
    return p


# ------------------------------------------------------------------ schemas
class Reading(BaseModel):
    type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temperature: float = Field(ge=280, le=320, description="Air temperature [K]")
    process_temperature: float = Field(ge=280, le=330, description="Process temperature [K]")
    rotational_speed: float = Field(ge=0, le=5000, description="Rotational speed [rpm]")
    torque: float = Field(ge=0, le=150, description="Torque [Nm]")
    tool_wear: float = Field(ge=0, le=400, description="Tool wear [min]")


class PredictIn(Reading):
    feature_set: FeatureSet = "engineered"
    machine_id: str | None = Field(default=None, max_length=60)
    note: str | None = Field(default=None, max_length=300)
    save: bool = True


class TrainIn(BaseModel):
    feature_sets: list[FeatureSet] = ["engineered", "raw"]


# ------------------------------------------------------------------ status and data
@app.get("/api/health")
def health():
    models = {fs: ({"ready": True, "trained_at": p.b["trained_at"]} if (p := _PRED.get(fs)) else
                   {"ready": False, "error": _PRED_ERR.get(fs)}) for fs in ("engineered", "raw")}
    try:
        DS.frame(); dataset = True; dataset_error = None
    except Exception as e:
        dataset, dataset_error = False, str(e)
    return {"status": "ok", "store": STORE.kind, "store_warning": STORE_WARNING, "models": models,
            "dataset": dataset, "dataset_error": dataset_error, "results": config.RESULTS_PATH.exists(),
            "history_count": STORE.count(), "job": (j.to_dict() if (j := JOBS.running()) else None)}


@app.get("/api/dataset/summary")
def dataset_summary():
    return DS.summary()


@app.get("/api/dataset/histogram")
def dataset_histogram(feature: str = S.TORQUE, bins: int = Query(30, ge=5, le=80)):
    try:
        return DS.histograms(feature, bins)
    except KeyError:
        raise HTTPException(400, f"unknown feature {feature}")


@app.get("/api/dataset/scatter")
def dataset_scatter(x: str = S.ROT_SPEED, y: str = S.TORQUE, n: int = Query(1500, ge=100, le=5000)):
    try:
        return DS.scatter(x, y, n)
    except KeyError:
        raise HTTPException(400, "unknown feature")


@app.get("/api/dataset/rows")
def dataset_rows(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500), failures_only: bool = False):
    return DS.rows(offset, limit, failures_only)


@app.get("/api/dataset/features")
def dataset_features():
    return [{"key": k, "label": DS.DERIVED.get(k, k)} for k in DS.NUMERIC + list(DS.DERIVED)]


# ------------------------------------------------------------------ results of the research experiments
@app.get("/api/results")
def results():
    if not config.RESULTS_PATH.exists():
        raise HTTPException(404, "No experiment results yet. Run the experiments from the Experiments page.")
    return json.loads(config.RESULTS_PATH.read_text())


@app.get("/api/figures")
def figures():
    if not config.FIGURES_DIR.exists():
        return []
    return sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))


@app.get("/api/figures/{name}")
def figure(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_\-]+\.png", name):
        raise HTTPException(400, "invalid name")
    path = config.FIGURES_DIR / name
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/png")


# ------------------------------------------------------------------ models and explanations
@app.get("/api/models")
def models(feature_set: FeatureSet = "engineered"):
    return predictor(feature_set).info()


@app.get("/api/explain/global")
def explain_global(feature_set: FeatureSet = "engineered"):
    return predictor(feature_set).global_explanations()


# ------------------------------------------------------------------ prediction
@app.post("/api/predict")
def predict(body: PredictIn):
    p = predictor(body.feature_set)
    rec = body.model_dump(include=set(FIELDS))
    try:
        out = p.predict_one(rec)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if body.save:
        doc = STORE.add({"machine_id": body.machine_id, "note": body.note, "feature_set": body.feature_set,
                         "input": out["input"], "probability": out["probability"], "risk": out["risk"],
                         "predicted_failure": out["predicted_failure"], "entropy": out["entropy"],
                         "disagreement": out["disagreement"], "review_recommended": out["review_recommended"],
                         "triggered_rules": [r["mode"] for r in out["rules"] if r["triggered"]]})
        out["history_id"], out["created_at"] = doc["id"], doc["created_at"]
    return out


ALIASES = {**{v.lower(): k for k, v in FIELDS.items()}, **{k: k for k in FIELDS},
           "air temperature": "air_temperature", "process temperature": "process_temperature",
           "rotational speed": "rotational_speed", "tool wear": "tool_wear"}


def _parse_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    if df.empty:
        raise HTTPException(400, "The CSV has no rows.")
    if len(df) > config.MAX_BATCH_ROWS:
        raise HTTPException(413, f"At most {config.MAX_BATCH_ROWS} rows per batch.")
    return df


@app.post("/api/batch")
async def batch(file: UploadFile = File(...), feature_set: FeatureSet = Form("engineered")):
    p = predictor(feature_set)
    df = _parse_csv(await file.read())
    cols = {c: ALIASES.get(c.strip().lower()) for c in df.columns}
    mapped = df.rename(columns={c: a for c, a in cols.items() if a})
    missing = [f for f in FIELDS if f not in mapped.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {', '.join(missing)}. Download the template for the expected format.")
    try:
        raw = to_frame(mapped[list(FIELDS)].to_dict("records"))
    except ValueError as e:
        raise HTTPException(422, str(e))
    out = p.predict_frame(raw)
    ident = next((c for c in (S.PRODUCT_ID, "machine_id", S.UDI) if c in df.columns), None)
    rows = []
    for i in range(len(df)):
        r = out.iloc[i]
        rows.append({"row": i + 1, "id": (str(df[ident].iloc[i]) if ident else str(i + 1)),
                     **{k: (str(raw.iloc[i][v]) if k == "type" else float(raw.iloc[i][v])) for k, v in FIELDS.items()},
                     "probability": float(r["p_failure"]), "risk": r["risk"], "predicted_failure": int(r["predicted_failure"]),
                     "entropy": float(r["entropy"]), "disagreement": float(r["disagreement"]),
                     "review_recommended": bool(r["review_recommended"])})
    summary = {"rows": len(rows), "risk_counts": out["risk"].value_counts().to_dict(),
               "predicted_failures": int(out["predicted_failure"].sum()),
               "review_recommended": int(out["review_recommended"].sum()), "threshold": p.b["thresholds"]["IEMAF"]}
    if S.MACHINE_FAILURE in df.columns:
        y = pd.to_numeric(df[S.MACHINE_FAILURE], errors="coerce")
        if y.notna().all() and set(y.unique()) <= {0, 1}:
            m = evaluate(y.astype(int).values, out["p_failure"].values, p.b["thresholds"]["IEMAF"])
            summary["evaluation"] = {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc",
                                                        "tp", "fp", "fn", "tn")}
            summary["evaluation_note"] = ("Labels found in the file. If these rows include the training split, the "
                                          "scores are optimistic.")
    return {"feature_set": feature_set, "summary": summary, "rows": rows}


@app.get("/api/batch/template", response_class=PlainTextResponse)
def batch_template():
    return ("machine_id,type,air_temperature,process_temperature,rotational_speed,torque,tool_wear\n"
            "M-001,M,298.1,308.6,1551,42.8,0\nM-002,L,302.5,310.5,1300,60.0,120\n"
            "M-003,L,300.0,310.0,1500,50.0,230\nM-004,H,298.8,309.0,2800,8.0,40\n")


# ------------------------------------------------------------------ history
@app.get("/api/history")
def history(limit: int = Query(200, ge=1, le=5000), risk: str | None = None):
    return STORE.list(limit, risk)


@app.get("/api/history/stats")
def history_stats():
    items = STORE.list(5000)
    counts: dict[str, int] = {}
    for it in items:
        counts[it["risk"]] = counts.get(it["risk"], 0) + 1
    return {"total": len(items), "risk_counts": counts, "recent": items[:8]}


@app.get("/api/history/{id_}")
def history_item(id_: str):
    d = STORE.get(id_)
    if not d:
        raise HTTPException(404, "not found")
    return d


@app.delete("/api/history/{id_}")
def history_delete(id_: str):
    if not STORE.delete(id_):
        raise HTTPException(404, "not found")
    return {"deleted": 1}


@app.delete("/api/history")
def history_clear():
    return {"deleted": STORE.clear()}


# ------------------------------------------------------------------ jobs
@app.post("/api/jobs/train")
def job_train(body: TrainIn):
    try:
        return JOBS.start("train", {"feature_sets": list(dict.fromkeys(body.feature_sets))}).to_dict()
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/jobs/experiments")
def job_experiments():
    try:
        return JOBS.start("experiments", {}).to_dict()
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/jobs")
def jobs():
    return [j.to_dict() for j in sorted(JOBS.jobs.values(), key=lambda j: j.started_at or "", reverse=True)]


@app.get("/api/jobs/{id_}")
def job(id_: str):
    j = JOBS.jobs.get(id_)
    if not j:
        raise HTTPException(404, "not found")
    return j.to_dict()


# ------------------------------------------------------------------ front end
if config.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        f = config.FRONTEND_DIST / path
        if path and f.is_file() and config.FRONTEND_DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(config.FRONTEND_DIST / "index.html")
