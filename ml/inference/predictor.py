"""Load a verified model bundle and turn machine readings into probabilities, risk levels and explanations.

Security: a bundle is loaded only if its SHA-256 matches artifacts/models/manifest.json, which is written by
ml.inference.train on this machine, and only if it was trained with the installed library versions. Bundles from
elsewhere are never loaded; retrain instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost

from ml.data import preprocessing as P
from ml.data import schema as S
from ml.inference.train import MODEL_DIR, entropy_bits, logit, sha256

# API field names -> dataset column names
FIELDS = {"type": S.TYPE, "air_temperature": S.AIR_TEMP, "process_temperature": S.PROCESS_TEMP,
          "rotational_speed": S.ROT_SPEED, "torque": S.TORQUE, "tool_wear": S.TOOL_WEAR}
UNCERTAIN_BITS = 0.8
OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}
RISK_TEXT = {
    "Low": "Failure is unlikely under these conditions. Continue normal operation.",
    "Medium": "Elevated risk. Schedule an inspection at the next convenient stop.",
    "High": "Failure is likely. Inspect the machine before continuing operation.",
    "Uncertain": "The models are not confident or disagree. Refer this case to an engineer for review.",
}


class BundleError(RuntimeError):
    pass


def to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records).rename(columns=FIELDS)
    missing = [c for c in S.INPUT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    df = df[list(S.INPUT_FEATURES)].copy()
    df[S.TYPE] = df[S.TYPE].astype(str).str.strip().str.upper()
    bad = ~df[S.TYPE].isin(S.TYPE_LEVELS)
    if bad.any():
        raise ValueError(f"type must be one of L, M, H (row {int(np.argmax(bad.values)) + 1})")
    for c in S.NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise ValueError(f"{c} must be numeric (row {int(np.argmax(df[c].isna().values)) + 1})")
        lo, hi = S.VALUE_RANGES[c]
        out = (df[c] < lo) | (df[c] > hi)
        if out.any():
            raise ValueError(f"{c} must be between {lo} and {hi} (row {int(np.argmax(out.values)) + 1})")
    df[S.TYPE] = df[S.TYPE].astype("string")
    return df


def rule_checks(row: pd.Series) -> list[dict]:
    """Documented failure rules of the synthetic AI4I generator (Matzka, 2020), shown for transparency."""
    diff = row[S.PROCESS_TEMP] - row[S.AIR_TEMP]
    power = row[S.TORQUE] * row[S.ROT_SPEED] * 2 * np.pi / 60
    strain = row[S.TOOL_WEAR] * row[S.TORQUE]
    limit = OSF_LIMIT[str(row[S.TYPE])]
    return [
        {"mode": "HDF", "name": "Heat dissipation", "triggered": bool(diff < 8.6 and row[S.ROT_SPEED] < 1380),
         "detail": f"temperature difference {diff:.1f} K (< 8.6) and speed {row[S.ROT_SPEED]:.0f} rpm (< 1,380)"},
        {"mode": "PWF", "name": "Power", "triggered": bool(power < 3500 or power > 9000),
         "detail": f"power {power:,.0f} W (outside 3,500–9,000 W fails)"},
        {"mode": "OSF", "name": "Overstrain", "triggered": bool(strain > limit),
         "detail": f"tool wear × torque {strain:,.0f} min·Nm (limit {limit:,} for type {row[S.TYPE]})"},
        {"mode": "TWF", "name": "Tool wear", "triggered": bool(row[S.TOOL_WEAR] >= 200),
         "detail": f"tool wear {row[S.TOOL_WEAR]:.0f} min (random replacement failures occur at about 200–240 min)"},
    ]


class Predictor:
    def __init__(self, fs: str):
        manifest_path = MODEL_DIR / "manifest.json"
        path = MODEL_DIR / f"{fs}.joblib"
        if not path.exists() or not manifest_path.exists():
            raise BundleError(f"No trained '{fs}' model yet. Train the models first.")
        entry = json.loads(manifest_path.read_text()).get(fs)
        if not entry or sha256(path) != entry["sha256"]:
            raise BundleError(f"The '{fs}' model file does not match its manifest hash; refusing to load it. Retrain.")
        current = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__}
        if entry.get("versions") != current:
            raise BundleError(f"The '{fs}' model was trained with {entry.get('versions')} but {current} is installed. Retrain.")
        b = joblib.load(path)  # verified local artifact
        self.b, self.fs = b, fs
        self.models = b["models"]
        self.base = b["base_names"]
        xgb = self.models["XGBoost"]
        self._xgb_prep, self._xgb = xgb.named_steps["prep"], xgb.named_steps["model"]
        self._explainer = shap.TreeExplainer(self._xgb)
        cart = self.models["CART"]
        self._cart_prep, self._cart = cart.named_steps["prep"], cart.named_steps["model"]

    # ---------------------------------------------------------------- info
    def info(self) -> dict:
        b = self.b
        return {k: b[k] for k in ("feature_set", "trained_at", "train_seconds", "versions", "thresholds", "tau_entropy",
                                  "tau_disagree", "deferral_test", "metrics_test", "bma_weights", "bma_loglik", "split",
                                  "tuning", "input_ranges")}

    # ---------------------------------------------------------------- prediction
    def _features(self, raw: pd.DataFrame) -> pd.DataFrame:
        return P.DerivedFeatures().transform(raw) if self.b["engineered"] else raw

    def predict_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        X = self._features(raw).reset_index(drop=True)
        probs = {n: self.models[n].predict_proba(X)[:, 1] for n in self.base}
        M = np.column_stack([probs[n] for n in self.base])
        out = pd.DataFrame({f"p_{n}": probs[n] for n in self.base})
        out["p_soft_voting"] = M.mean(1)
        out["p_stacking"] = self.b["meta"].predict_proba(np.column_stack([logit(probs[n]) for n in self.base]))[:, 1]
        w = self.b["bma_weights"]
        out["p_bma"] = sum(w[n] * probs[n] for n in self.base)
        out["p_failure"] = self.b["platt"].predict_proba(logit(out["p_bma"].values)[:, None])[:, 1]
        out["entropy"] = entropy_bits(out["p_failure"].values)
        out["disagreement"] = M.std(1)
        thr = self.b["thresholds"]["IEMAF"]
        out["predicted_failure"] = (out["p_failure"] >= thr).astype(int)
        lo_ranges = self.b["input_ranges"]
        extrap = np.zeros(len(raw), bool)
        for c, (lo, hi) in lo_ranges.items():
            extrap |= ((raw[c] < lo) | (raw[c] > hi)).values
        out["outside_training_range"] = extrap
        # review flag: the validation-calibrated deferral policy evaluated in Experiment 6 (about 10-15% of cases)
        out["review_recommended"] = ((out["entropy"] >= self.b["tau_entropy"]) |
                                     (out["disagreement"] >= self.b["tau_disagree"]) | extrap)
        # risk level: "Uncertain" when the calibrated probability is close to a coin flip (entropy >= 0.8 bits,
        # i.e. probability between about 0.24 and 0.76) or the input lies outside the training data
        uncertain = (out["entropy"] >= UNCERTAIN_BITS) | extrap
        risk = np.where(out["p_failure"] >= thr, "High", np.where(out["p_failure"] >= thr / 3, "Medium", "Low"))
        out["risk"] = np.where(uncertain, "Uncertain", risk)
        return out

    def predict_one(self, record: dict) -> dict:
        raw = to_frame([record])
        r = self.predict_frame(raw).iloc[0]
        X = self._features(raw)
        thr = self.b["thresholds"]
        models = [{"model": n, "probability": float(r[f"p_{n}"]), "threshold": float(thr[n]),
                   "weight_bma": float(self.b["bma_weights"][n])} for n in self.base]
        ensembles = [{"model": "Soft Voting", "probability": float(r["p_soft_voting"]), "threshold": float(thr["Soft Voting"])},
                     {"model": "Stacking", "probability": float(r["p_stacking"]), "threshold": float(thr["Stacking"])},
                     {"model": "BMA", "probability": float(r["p_bma"]), "threshold": float(thr["BMA"])},
                     {"model": "IEMAF (calibrated BMA)", "probability": float(r["p_failure"]), "threshold": float(thr["IEMAF"])}]
        derived = P.DerivedFeatures().transform(raw).iloc[0]
        return {
            "feature_set": self.fs,
            "input": {k: (str(raw.iloc[0][v]) if k == "type" else float(raw.iloc[0][v])) for k, v in FIELDS.items()},
            "derived": {"temperature_difference": float(derived["temp_diff_K"]), "power_w": float(derived["power_W"]),
                        "wear_torque": float(derived["wear_torque_minNm"])},
            "probability": float(r["p_failure"]), "threshold": float(thr["IEMAF"]),
            "predicted_failure": bool(r["predicted_failure"]), "risk": r["risk"], "advice": RISK_TEXT[r["risk"]],
            "entropy": float(r["entropy"]), "disagreement": float(r["disagreement"]),
            "tau_entropy": self.b["tau_entropy"], "tau_disagree": self.b["tau_disagree"],
            "outside_training_range": bool(r["outside_training_range"]),
            "review_recommended": bool(r["review_recommended"]), "uncertain_bits": UNCERTAIN_BITS,
            "models": models, "ensembles": ensembles,
            "shap": self.shap_local(X), "cart_path": self.cart_path(X), "rules": rule_checks(raw.iloc[0]),
        }

    # ---------------------------------------------------------------- explanations
    def shap_local(self, X: pd.DataFrame) -> dict:
        Z = self._xgb_prep.transform(X)
        sv = self._explainer.shap_values(Z)[0]
        names = self.b["feature_names_tree"]
        base = float(np.ravel(self._explainer.expected_value)[0])
        contrib = sorted(({"feature": n, "value": float(Z[0, i]), "shap": float(sv[i])} for i, n in enumerate(names)),
                         key=lambda d: -abs(d["shap"]))
        return {"model": "XGBoost", "base_logit": base, "output_logit": base + float(sv.sum()),
                "output_probability": float(1 / (1 + np.exp(-(base + sv.sum())))), "contributions": contrib}

    def cart_path(self, X: pd.DataFrame) -> dict:
        Z = self._cart_prep.transform(X)
        tree = self._cart.tree_
        names = self.b["feature_names_tree"]
        node, steps = 0, []
        while tree.children_left[node] != -1:
            f, t = tree.feature[node], tree.threshold[node]
            v = Z[0, f]
            go_left = v <= t
            label = names[f]
            if label == S.TYPE:
                shown = f"Type {'≤' if go_left else '>'} {t:.1f}  (L=0, M=1, H=2; this machine = {S.TYPE_LEVELS[int(v)]})"
            else:
                shown = f"{label} {'≤' if go_left else '>'} {t:,.1f}  (value {v:,.1f})"
            steps.append(shown)
            node = tree.children_left[node] if go_left else tree.children_right[node]
        counts = tree.value[node][0]
        frac = counts / counts.sum()
        return {"steps": steps, "leaf_samples": int(tree.n_node_samples[node]),
                "leaf_failure_share": float(frac[1]) if len(frac) > 1 else 0.0}

    def global_explanations(self) -> dict:
        return self.b["global"]
