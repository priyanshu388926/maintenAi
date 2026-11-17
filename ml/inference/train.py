"""Train and save the production model bundle used by the MaintenAI web application.

    python -m ml.inference.train                 # both feature sets
    python -m ml.inference.train engineered      # one feature set

For each feature set ("raw" or "engineered") this fits the seven base models on the training split, builds
soft voting, stacking and Bayesian model averaging (BMA) from out-of-fold probabilities, calibrates BMA with Platt
scaling on the validation split, chooses decision and uncertainty thresholds on the validation split, and reports
test-set metrics. Everything is saved to artifacts/models/<feature_set>.joblib together with a SHA-256 entry in
artifacts/models/manifest.json; the loader refuses any bundle whose hash does not match (see predictor.py).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import shap
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import export_text

from ml.data import preprocessing as P
from ml.data import schema as S
from ml.data.ingestion import load_ai4i
from ml.evaluation.metrics import best_f1_threshold, evaluate
from ml.models.bma import BayesianModelAveraging
from ml.models.zoo import SEED, base_models

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "models"
RESULTS = ROOT / "artifacts" / "metrics" / "results.json"
FEATURE_SETS = {"raw": False, "engineered": True}
BASE = ["CART", "AdaBoost", "Gradient Boosting", "XGBoost", "MARS", "GAM", "Random Forest"]
DEFER_QUANTILE = 0.90   # predictions less certain than 90% of validation cases are routed to human review


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def entropy_bits(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tuned_params(fs: str):
    """CART / XGBoost settings tuned by repeated CV in the experiment run; documented defaults otherwise."""
    if RESULTS.exists():
        t = json.loads(RESULTS.read_text()).get("tuning", {}).get(fs)
        if t:
            cart = {k: (None if v == "None" else v) for k, v in t["cart_best"].items()}
            return cart, t["xgb_best"], "tuned by repeated 5-fold CV (results.json)"
    return None, None, "defaults (run the experiments to tune)"


def _gam_shapes(pipe, numeric_names):
    gam = pipe.named_steps["model"].gam_
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    out = {}
    for j, name in enumerate(numeric_names):
        term = 2 + j
        XX = gam.generate_X_grid(term=term, n=60)
        pdep, conf = gam.partial_dependence(term=term, X=XX, width=0.95)
        x = XX[:, term] * scaler.scale_[j] + scaler.mean_[j]
        out[name] = {"x": x.round(3).tolist(), "y": pdep.round(4).tolist(),
                     "lo": conf[:, 0].round(4).tolist(), "hi": conf[:, 1].round(4).tolist()}
    return out


def train_bundle(fs: str, progress=print) -> dict:
    eng = FEATURE_SETS[fs]
    t0 = time.time()
    df, _ = load_ai4i()
    split = P.stratified_split(df)
    Xtr, ytr = P.make_xy(df, split["train"], eng)
    Xva, yva = P.make_xy(df, split["validation"], eng)
    Xte, yte = P.make_xy(df, split["test"], eng)
    cart_p, xgb_p, tuning_note = tuned_params(fs)
    zoo = base_models(eng, cart_p, xgb_p)

    models, oof, pva, pte, thr, metrics = {}, {}, {}, {}, {}, {}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for i, (name, est) in enumerate(zoo.items(), 1):
        progress(f"[{fs}] {i}/7 {name}: 5-fold out-of-fold fit")
        o = np.zeros(len(ytr))
        for tr_i, va_i in skf.split(Xtr, ytr):
            o[va_i] = clone(est).fit(Xtr.iloc[tr_i], ytr[tr_i]).predict_proba(Xtr.iloc[va_i])[:, 1]
        m = clone(est).fit(Xtr, ytr)
        models[name], oof[name] = m, o
        pva[name], pte[name] = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        thr[name] = best_f1_threshold(yva, pva[name])
        metrics[name] = evaluate(yte, pte[name], thr[name])

    progress(f"[{fs}] ensembles: voting, stacking, BMA, calibration")
    meta = LogisticRegression(C=1.0, max_iter=5000).fit(np.column_stack([logit(oof[n]) for n in BASE]), ytr)
    bma = BayesianModelAveraging().fit(oof, ytr)
    ens_va = {"Soft Voting": np.mean([pva[n] for n in BASE], 0),
              "Stacking": meta.predict_proba(np.column_stack([logit(pva[n]) for n in BASE]))[:, 1],
              "BMA": bma.predict_proba(pva)}
    ens_te = {"Soft Voting": np.mean([pte[n] for n in BASE], 0),
              "Stacking": meta.predict_proba(np.column_stack([logit(pte[n]) for n in BASE]))[:, 1],
              "BMA": bma.predict_proba(pte)}
    platt = LogisticRegression(C=1e6, max_iter=5000).fit(logit(ens_va["BMA"])[:, None], yva)
    ens_va["IEMAF"] = platt.predict_proba(logit(ens_va["BMA"])[:, None])[:, 1]
    ens_te["IEMAF"] = platt.predict_proba(logit(ens_te["BMA"])[:, None])[:, 1]
    for name in ens_va:
        thr[name] = best_f1_threshold(yva, ens_va[name])
        metrics[name] = evaluate(yte, ens_te[name], thr[name])

    # uncertainty thresholds from the validation split only
    ent_va = entropy_bits(ens_va["IEMAF"])
    dis_va = np.column_stack([pva[n] for n in BASE]).std(1)
    tau_entropy = float(np.quantile(ent_va, DEFER_QUANTILE))
    tau_disagree = float(np.quantile(dis_va, DEFER_QUANTILE))
    # how deferral behaves on the test split with these validation thresholds
    ent_te = entropy_bits(ens_te["IEMAF"])
    dis_te = np.column_stack([pte[n] for n in BASE]).std(1)
    defer = (ent_te >= tau_entropy) | (dis_te >= tau_disagree)
    yhat = (ens_te["IEMAF"] >= thr["IEMAF"]).astype(int)
    deferral = {"deferred_share": float(defer.mean()), "failures_deferred": int(yte[defer].sum()),
                "failures_total": int(yte.sum()), "error_rate_kept": float((yhat[~defer] != yte[~defer]).mean()),
                "error_rate_all": float((yhat != yte).mean())}

    progress(f"[{fs}] explanations: SHAP, CART rules, GAM shapes")
    xgb = models["XGBoost"]
    names_tree = list(xgb.named_steps["prep"].get_feature_names_out())
    Zte = xgb.named_steps["prep"].transform(Xte)
    sv = shap.TreeExplainer(xgb.named_steps["model"]).shap_values(Zte)
    cart = models["CART"]
    numeric = P.numeric_columns(eng)
    glob = {
        "shap_mean_abs": dict(sorted(zip(names_tree, np.abs(sv).mean(0).round(4).tolist()), key=lambda kv: -kv[1])),
        "cart_rules": export_text(cart.named_steps["model"], feature_names=names_tree, decimals=1),
        "cart_importance": dict(sorted(zip(names_tree, cart.named_steps["model"].feature_importances_.round(4).tolist()),
                                       key=lambda kv: -kv[1])),
        "cart_depth": int(cart.named_steps["model"].get_depth()), "cart_leaves": int(cart.named_steps["model"].get_n_leaves()),
        "gam_shapes": _gam_shapes(models["GAM"], numeric),
        "mars_terms": models["MARS"].named_steps["model"].describe(),
        "stacking_coef": dict(zip(BASE, meta.coef_[0].round(4).tolist())),
    }
    ranges = {c: [float(df[c].min()), float(df[c].max())] for c in S.NUMERIC_FEATURES}
    bundle = {
        "feature_set": fs, "engineered": eng, "base_names": BASE, "models": models, "meta": meta,
        "bma_weights": bma.weights(), "bma_loglik": dict(zip(bma.names_, map(float, bma.loglik_))), "platt": platt,
        "thresholds": thr, "tau_entropy": tau_entropy, "tau_disagree": tau_disagree, "deferral_test": deferral,
        "metrics_test": metrics, "global": glob, "feature_names_tree": names_tree, "input_ranges": ranges,
        "split": {k: {"rows": int(len(v)), "failures": int(df.loc[v, S.MACHINE_FAILURE].sum())} for k, v in split.items()},
        "tuning": {"cart": cart_p, "xgb": xgb_p, "note": tuning_note},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": round(time.time() - t0, 1),
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__},
    }
    return bundle


def save_bundle(bundle: dict) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{bundle['feature_set']}.joblib"
    tmp = path.with_suffix(".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(path)
    manifest_path = MODEL_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[bundle["feature_set"]] = {"file": path.name, "sha256": sha256(path), "trained_at": bundle["trained_at"],
                                       "train_seconds": bundle["train_seconds"], "versions": bundle["versions"]}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return path


def main(argv=None):
    sets = (argv or sys.argv[1:]) or list(FEATURE_SETS)
    for fs in sets:
        b = train_bundle(fs)
        p = save_bundle(b)
        m = b["metrics_test"]["IEMAF"]
        print(f"saved {p} in {b['train_seconds']} s | IEMAF test PR-AUC {m['pr_auc']:.3f} recall {m['recall']:.3f} "
              f"| deferred {b['deferral_test']['deferred_share']:.1%}")


if __name__ == "__main__":
    main()
