"""Runs Experiments 1-8 of the MaintenAI plan on AI4I 2020 and writes all results and figures.

    python -m ml.experiments.run_all

Outputs: artifacts/metrics/results.json, reports/figures/*.png, data/processed/splits.json
Every number in the report comes from this script.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from xgboost import XGBClassifier

from ml.data import preprocessing as P
from ml.data import schema as S
from ml.data.ingestion import load_ai4i
from ml.evaluation.metrics import best_f1_threshold, bootstrap_ci, evaluate
from ml.models.bma import BayesianModelAveraging, bernoulli_loglik
from ml.models.zoo import SEED, base_models

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "artifacts" / "metrics"; OUT.mkdir(parents=True, exist_ok=True)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#52514e"]
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "axes.axisbelow": True, "font.size": 9})
R: dict = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def proba(model, X):
    return model.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ data
df, _ = load_ai4i()
split = P.stratified_split(df)
P.save_split(split, ROOT / "data" / "processed" / "splits.json")
R["split"] = {k: {"rows": int(len(v)), "failures": int(df.loc[v, S.MACHINE_FAILURE].sum())} for k, v in split.items()}

DATA = {}
for eng in (False, True):
    DATA[eng] = {part: P.make_xy(df, split[part], eng) for part in ("train", "validation", "test")}
FS = {False: "raw", True: "engineered"}

# ------------------------------------------------------------------ CART and XGBoost tuning (repeated CV on train)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
TUNED = {}
for eng in (False, True):
    Xtr, ytr = DATA[eng]["train"]
    g = GridSearchCV(Pipeline([("prep", P.tree_preprocessor(eng)), ("model", DecisionTreeClassifier(random_state=SEED))]),
                     {"model__criterion": ["gini", "entropy"], "model__max_depth": [3, 4, 5, 6, 8, 10, None],
                      "model__min_samples_leaf": [1, 5, 20], "model__ccp_alpha": [0.0, 0.0005, 0.001]},
                     scoring="average_precision", cv=rcv, n_jobs=2).fit(Xtr, ytr)
    cart = {k.replace("model__", ""): v for k, v in g.best_params_.items()}
    # criterion comparison for the report: best PR-AUC per criterion and depth
    cvr = g.cv_results_
    crit = {}
    for c in ("gini", "entropy"):
        for d in (3, 4, 5, 6, 8, 10, None):
            m = [i for i, prm in enumerate(cvr["params"]) if prm["model__criterion"] == c and prm["model__max_depth"] == d]
            crit[f"{c}|{d}"] = float(max(cvr["mean_test_score"][i] for i in m))
    gx = GridSearchCV(Pipeline([("prep", P.tree_preprocessor(eng)), ("model", XGBClassifier(
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=2))]),
        {"model__n_estimators": [200, 400], "model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]},
        scoring="average_precision", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1).fit(Xtr, ytr)
    xgbp = {k.replace("model__", ""): v for k, v in gx.best_params_.items()}
    TUNED[eng] = {"cart": cart, "xgb": xgbp}
    R.setdefault("tuning", {})[FS[eng]] = {"cart_best": {k: (v if v is not None else "None") for k, v in cart.items()},
                                          "cart_cv_pr_auc": float(g.best_score_), "cart_criterion_depth": crit,
                                          "xgb_best": xgbp, "xgb_cv_pr_auc": float(gx.best_score_)}
    log("tuned", FS[eng], cart, round(g.best_score_, 3), xgbp, round(gx.best_score_, 3))

# ------------------------------------------------------------------ Experiment 1: individual models
MODELS, PROBS, OOF, THR = {}, {}, {}, {}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
R["exp1"] = {}
for eng in (False, True):
    Xtr, ytr = DATA[eng]["train"]; Xva, yva = DATA[eng]["validation"]; Xte, yte = DATA[eng]["test"]
    zoo = base_models(eng, TUNED[eng]["cart"], TUNED[eng]["xgb"])
    for name, est in zoo.items():
        # out-of-fold probabilities on train (for CV metrics, stacking and BMA evidence)
        oof = np.zeros(len(ytr)); fold_pr = []
        for tr_i, va_i in skf.split(Xtr, ytr):
            m = clone(est).fit(Xtr.iloc[tr_i], ytr[tr_i])
            oof[va_i] = proba(m, Xtr.iloc[va_i]); fold_pr.append(average_precision_score(ytr[va_i], oof[va_i]))
        t = time.time(); m = clone(est).fit(Xtr, ytr); fit_s = time.time() - t
        t = time.time(); pte = proba(m, Xte); inf_ms = (time.time() - t) / len(yte) * 1000
        pva = proba(m, Xva)
        thr = best_f1_threshold(yva, pva)
        MODELS[(eng, name)], OOF[(eng, name)], THR[(eng, name)] = m, oof, thr
        PROBS[(eng, name)] = {"validation": pva, "test": pte}
        res = evaluate(yte, pte, thr)
        res.update({"cv_pr_auc_mean": float(np.mean(fold_pr)), "cv_pr_auc_sd": float(np.std(fold_pr)),
                    "fit_seconds": fit_s, "inference_ms_per_row": inf_ms})
        R["exp1"].setdefault(FS[eng], {})[name] = res
        log("exp1", FS[eng], name, "cvPR", round(np.mean(fold_pr), 3), "test PR", round(res["pr_auc"], 3),
            "rec", round(res["recall"], 3), "prec", round(res["precision"], 3))

BASE = list(base_models(False).keys())

# ------------------------------------------------------------------ Experiment 2: ensembles and BMA
R["exp2"], BMAW, ENS = {}, {}, {}
for eng in (False, True):
    Xtr, ytr = DATA[eng]["train"]; _, yva = DATA[eng]["validation"]; _, yte = DATA[eng]["test"]
    va = {n: PROBS[(eng, n)]["validation"] for n in BASE}; te = {n: PROBS[(eng, n)]["test"] for n in BASE}
    oof = {n: OOF[(eng, n)] for n in BASE}
    out = {}
    # soft voting: unweighted mean of the seven base models
    vote_va, vote_te = np.mean([va[n] for n in BASE], 0), np.mean([te[n] for n in BASE], 0)
    # stacking: logistic regression on logit of out-of-fold probabilities (leakage-safe)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    meta = LogisticRegression(C=1.0, max_iter=5000).fit(np.column_stack([lg(oof[n]) for n in BASE]), ytr)
    st_va = meta.predict_proba(np.column_stack([lg(va[n]) for n in BASE]))[:, 1]
    st_te = meta.predict_proba(np.column_stack([lg(te[n]) for n in BASE]))[:, 1]
    # BMA: posterior weights from out-of-fold predictive log-likelihood, uniform prior
    bma = BayesianModelAveraging().fit(oof, ytr)
    bma_va, bma_te = bma.predict_proba(va), bma.predict_proba(te)
    BMAW[eng] = {"weights": bma.weights(), "oof_loglik": dict(zip(bma.names_, map(float, bma.loglik_)))}
    # sensitivity only (not BMA): tempered posterior
    tempered = {}
    for T in (10, 100):
        b = BayesianModelAveraging(temperature=T).fit(oof, ytr)
        tempered[str(T)] = {"weights": b.weights()}
        p_va, p_te = b.predict_proba(va), b.predict_proba(te)
        tempered[str(T)]["test"] = evaluate(yte, p_te, best_f1_threshold(yva, p_va))
    BMAW[eng]["tempered"] = tempered
    for name, pv, pt in [("Soft Voting", vote_va, vote_te), ("Stacking", st_va, st_te), ("BMA", bma_va, bma_te)]:
        thr = best_f1_threshold(yva, pv)
        out[name] = evaluate(yte, pt, thr)
        ENS[(eng, name)] = {"validation": pv, "test": pt, "thr": thr}
    out["Stacking"]["meta_coef"] = dict(zip(BASE, map(float, meta.coef_[0])))
    best_single = max(BASE, key=lambda n: R["exp1"][FS[eng]][n]["cv_pr_auc_mean"])
    out["best_individual_by_cv"] = best_single
    R["exp2"][FS[eng]] = out
    log("exp2", FS[eng], {k: round(v["pr_auc"], 3) for k, v in out.items() if isinstance(v, dict) and "pr_auc" in v},
        "BMA w", {k: round(v, 3) for k, v in bma.weights().items()})
R["bma"] = {FS[e]: BMAW[e] for e in BMAW}

# bootstrap CIs for the main models (test set)
R["ci"] = {}
for eng in (False, True):
    _, yte = DATA[eng]["test"]
    for name in ("CART", "XGBoost", "Random Forest", "GAM", "MARS"):
        R["ci"].setdefault(FS[eng], {})[name] = bootstrap_ci(yte, PROBS[(eng, name)]["test"], THR[(eng, name)])
    for name in ("Soft Voting", "Stacking", "BMA"):
        e = ENS[(eng, name)]; R["ci"][FS[eng]][name] = bootstrap_ci(yte, e["test"], e["thr"])
log("bootstrap done")

# ------------------------------------------------------------------ Experiment 3: class imbalance (raw features)
R["exp3"] = {}
Xtr, ytr = DATA[False]["train"]; Xva, yva = DATA[False]["validation"]; Xte, yte = DATA[False]["test"]
pos_w = float((ytr == 0).sum() / (ytr == 1).sum())
strategies = {
    "None": lambda base: ImbPipeline([("prep", P.tree_preprocessor(False)), ("model", base)]),
    "Class weights": None,
    "Random oversampling": lambda base: ImbPipeline([("prep", P.tree_preprocessor(False)),
                                                     ("res", RandomOverSampler(random_state=SEED)), ("model", base)]),
    "SMOTE": lambda base: ImbPipeline([("prep", P.tree_preprocessor(False)),
                                       ("res", SMOTE(random_state=SEED)), ("model", base)]),
}
for mname in ("CART", "XGBoost"):
    for sname, make in strategies.items():
        if mname == "CART":
            base = DecisionTreeClassifier(random_state=SEED, **TUNED[False]["cart"],
                                          class_weight="balanced" if sname == "Class weights" else None)
        else:
            base = XGBClassifier(subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
                                 n_jobs=2, scale_pos_weight=pos_w if sname == "Class weights" else 1.0,
                                 **TUNED[False]["xgb"])
        pipe = (make or strategies["None"])(base).fit(Xtr, ytr)  # resampling happens inside the pipeline, train only
        pva, pte = proba(pipe, Xva), proba(pipe, Xte)
        r05 = evaluate(yte, pte, 0.5); rtuned = evaluate(yte, pte, best_f1_threshold(yva, pva))
        R["exp3"].setdefault(mname, {})[sname] = {"at_0.5": r05, "tuned_threshold": rtuned}
    log("exp3", mname, {k: (round(v["at_0.5"]["recall"], 3), round(v["tuned_threshold"]["recall"], 3)) for k, v in R["exp3"][mname].items()})

# ------------------------------------------------------------------ Experiment 5: calibration (fit on validation)
R["exp5"] = {}
CAL = {}
for eng in (False, True):
    Xva, yva = DATA[eng]["validation"]; _, yte = DATA[eng]["test"]
    for name in ("CART", "XGBoost", "Random Forest", "GAM", "BMA", "Stacking"):
        p_va = PROBS[(eng, name)]["validation"] if name in BASE else ENS[(eng, name)]["validation"]
        p_te = PROBS[(eng, name)]["test"] if name in BASE else ENS[(eng, name)]["test"]
        platt = LogisticRegression(C=1e6, max_iter=5000).fit(np.log(np.clip(p_va, 1e-6, 1 - 1e-6) / (1 - np.clip(p_va, 1e-6, 1 - 1e-6)))[:, None], yva)
        p_cal = platt.predict_proba(np.log(np.clip(p_te, 1e-6, 1 - 1e-6) / (1 - np.clip(p_te, 1e-6, 1 - 1e-6)))[:, None])[:, 1]
        pva_cal = platt.predict_proba(np.log(np.clip(p_va, 1e-6, 1 - 1e-6) / (1 - np.clip(p_va, 1e-6, 1 - 1e-6)))[:, None])[:, 1]
        a = evaluate(yte, p_te, 0.5); b = evaluate(yte, p_cal, 0.5)
        R["exp5"].setdefault(FS[eng], {})[name] = {"brier": a["brier"], "ece": a["ece"], "brier_cal": b["brier"], "ece_cal": b["ece"]}
        CAL[(eng, name)] = (p_te, p_cal, pva_cal)
log("exp5 done", {k: (round(v["ece"], 4), round(v["ece_cal"], 4)) for k, v in R["exp5"]["raw"].items()})

# ------------------------------------------------------------------ Experiment 6: uncertainty and human review (raw)
R["exp6"] = {}
for eng in (False, True):
    _, yte = DATA[eng]["test"]; _, yva = DATA[eng]["validation"]
    p_bma_te, p_bma_cal, p_bma_vacal = CAL[(eng, "BMA")]
    thr = best_f1_threshold(yva, p_bma_vacal)
    yhat = (p_bma_cal >= thr).astype(int)
    base_te = np.column_stack([PROBS[(eng, n)]["test"] for n in BASE])
    disagreement = base_te.std(1)
    pc = np.clip(p_bma_cal, 1e-9, 1 - 1e-9)
    entropy = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
    cov_res = {}
    for meas, u in (("entropy", entropy), ("disagreement", disagreement)):
        order = np.argsort(u)  # most certain first
        rows = []
        for cov in (1.0, 0.98, 0.95, 0.90, 0.85, 0.80):
            keep = order[: int(round(cov * len(u)))]
            defer = np.setdiff1d(np.arange(len(u)), keep)
            err = float((yhat[keep] != yte[keep]).mean())
            fn = int(((yhat[keep] == 0) & (yte[keep] == 1)).sum())
            rows.append({"coverage": cov, "error_rate": err, "missed_failures_kept": fn,
                         "failures_deferred": int(yte[defer].sum()), "deferred": int(len(defer))})
        cov_res[meas] = rows
    wrong = yhat != yte
    R["exp6"][FS[eng]] = {"threshold": thr, "coverage": cov_res,
                          "mean_entropy_correct": float(entropy[~wrong].mean()), "mean_entropy_wrong": float(entropy[wrong].mean()),
                          "mean_disagreement_correct": float(disagreement[~wrong].mean()),
                          "mean_disagreement_wrong": float(disagreement[wrong].mean()),
                          "n_wrong": int(wrong.sum())}
    if not eng:
        UNC_RAW = (entropy, disagreement, wrong)
log("exp6", {k: v["coverage"]["entropy"][3] for k, v in R["exp6"].items()})

# ------------------------------------------------------------------ Experiment 7: ablation (validation-tuned thresholds, test metrics)
R["exp7"] = {}
for eng in (False, True):
    _, yva = DATA[eng]["validation"]; _, yte = DATA[eng]["test"]
    def combo(names):
        pv = np.mean([PROBS[(eng, n)]["validation"] for n in names], 0); pt = np.mean([PROBS[(eng, n)]["test"] for n in names], 0)
        return evaluate(yte, pt, best_f1_threshold(yva, pv))
    p_te, p_cal, p_vacal = CAL[(eng, "BMA")]
    iemaf = evaluate(yte, p_cal, best_f1_threshold(yva, p_vacal))
    # IEMAF also defers the 5% most uncertain predictions (entropy) to human review: metrics on the retained 95%
    pc = np.clip(p_cal, 1e-9, 1 - 1e-9); ent = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
    keep = np.argsort(ent)[: int(0.95 * len(ent))]
    iemaf_kept = evaluate(yte[keep], p_cal[keep], best_f1_threshold(yva, p_vacal))
    R["exp7"][FS[eng]] = {
        "A CART": R["exp1"][FS[eng]]["CART"],
        "B Boosting (AdaBoost+GB+XGBoost)": combo(["AdaBoost", "Gradient Boosting", "XGBoost"]),
        "C GAM+MARS": combo(["GAM", "MARS"]),
        "D Random Forest": R["exp1"][FS[eng]]["Random Forest"],
        "E Soft Voting": R["exp2"][FS[eng]]["Soft Voting"],
        "F Stacking": R["exp2"][FS[eng]]["Stacking"],
        "G BMA": R["exp2"][FS[eng]]["BMA"],
        "H IEMAF (calibrated BMA)": iemaf,
        "H IEMAF, 5% deferred to review": iemaf_kept,
    }
log("exp7 done")

# ------------------------------------------------------------------ Experiment 8: leave-one-type-out generalisation
R["exp8"] = {}
for eng in (False, True):
    for held in S.TYPE_LEVELS:
        tr_idx = df.index[df[S.TYPE] != held]; te_idx = df.index[df[S.TYPE] == held]
        Xa, ya = P.make_xy(df, tr_idx, eng); Xb, yb = P.make_xy(df, te_idx, eng)
        for name in ("CART", "XGBoost", "Random Forest", "GAM"):
            m = clone(base_models(eng, TUNED[eng]["cart"], TUNED[eng]["xgb"])[name]).fit(Xa, ya)
            p = proba(m, Xb)
            R["exp8"].setdefault(FS[eng], {}).setdefault(held, {})[name] = {
                "pr_auc": float(average_precision_score(yb, p)), "roc_auc": float(roc_auc_score(yb, p)),
                "failures": int(yb.sum()), "rows": int(len(yb))}
    log("exp8", FS[eng], {h: round(v["XGBoost"]["pr_auc"], 3) for h, v in R["exp8"][FS[eng]].items()})

# ------------------------------------------------------------------ Experiment 4: explainability
R["exp4"] = {}
# CART rules (raw) - a shallow readable tree refit with depth 4 for display, plus tuned tree importance
Xtr, ytr = DATA[False]["train"]
cart_raw = MODELS[(False, "CART")]
names_tree = list(cart_raw.named_steps["prep"].get_feature_names_out())
R["exp4"]["cart_raw_importance"] = dict(zip(names_tree, map(float, cart_raw.named_steps["model"].feature_importances_)))
cart_eng = MODELS[(True, "CART")]
names_tree_e = list(cart_eng.named_steps["prep"].get_feature_names_out())
R["exp4"]["cart_eng_importance"] = dict(zip(names_tree_e, map(float, cart_eng.named_steps["model"].feature_importances_)))
R["exp4"]["cart_eng_rules"] = export_text(cart_eng.named_steps["model"], feature_names=names_tree_e, max_depth=4, decimals=1)
R["exp4"]["cart_raw_depth"] = int(cart_raw.named_steps["model"].get_depth()); R["exp4"]["cart_raw_leaves"] = int(cart_raw.named_steps["model"].get_n_leaves())
R["exp4"]["cart_eng_depth"] = int(cart_eng.named_steps["model"].get_depth()); R["exp4"]["cart_eng_leaves"] = int(cart_eng.named_steps["model"].get_n_leaves())
fig, ax = plt.subplots(figsize=(13, 5.2))
plot_tree(cart_eng.named_steps["model"], feature_names=[n.replace("_", " ") for n in names_tree_e], class_names=["normal", "failure"],
          max_depth=3, filled=True, impurity=False, proportion=False, fontsize=7, ax=ax, rounded=True)
ax.set_title("CART on engineered features (top 4 levels)"); fig.tight_layout(); fig.savefig(FIG / "cart_tree.png"); plt.close(fig)
R["exp4"]["mars_terms_eng"] = MODELS[(True, "MARS")].named_steps["model"].describe()
R["exp4"]["mars_terms_raw"] = MODELS[(False, "MARS")].named_steps["model"].describe()

# GAM partial dependence (raw features) in original units
gam_pipe = MODELS[(False, "GAM")]; gam = gam_pipe.named_steps["model"].gam_; pre = gam_pipe.named_steps["prep"]
num = list(S.NUMERIC_FEATURES); sc = pre.named_transformers_["num"]
fig, axes = plt.subplots(1, 5, figsize=(15, 2.9))
for k, (ax, col) in enumerate(zip(axes, num)):
    term = 2 + k
    XX = gam.generate_X_grid(term=term); pdep, conf = gam.partial_dependence(term=term, X=XX, width=0.95)
    xs = XX[:, term] * sc.scale_[k] + sc.mean_[k]
    ax.plot(xs, pdep, color=COL[0], lw=2); ax.fill_between(xs, conf[:, 0], conf[:, 1], color=COL[0], alpha=0.15)
    ax.set_title(col, fontsize=9); ax.set_ylabel("effect on logit" if k == 0 else "")
fig.suptitle("GAM shape functions (raw features, 95% intervals)", fontsize=10); fig.tight_layout(); fig.savefig(FIG / "gam_shapes.png"); plt.close(fig)
R["exp4"]["gam_lam"] = [float(np.ravel(x)[0]) for x in gam.lam]

# SHAP for XGBoost (raw and engineered)
for eng in (False, True):
    xgbp = MODELS[(eng, "XGBoost")]; Xte, _ = DATA[eng]["test"]
    Z = xgbp.named_steps["prep"].transform(Xte); nm = list(xgbp.named_steps["prep"].get_feature_names_out())
    ex = shap.TreeExplainer(xgbp.named_steps["model"]); sv = ex.shap_values(Z)
    R["exp4"].setdefault("shap_mean_abs", {})[FS[eng]] = dict(zip(nm, map(float, np.abs(sv).mean(0))))
    plt.figure(figsize=(6, 3.4))
    shap.summary_plot(sv, Z, feature_names=nm, show=False, plot_size=None, max_display=8)
    plt.title(f"SHAP values, XGBoost ({FS[eng]} features)", fontsize=10); plt.tight_layout()
    plt.savefig(FIG / f"shap_{FS[eng]}.png", dpi=150); plt.close()
    if not eng:
        # local explanation for the highest-risk test row
        i = int(np.argmax(proba(xgbp, Xte)))
        R["exp4"]["shap_local_raw"] = {"row": {c: (float(v) if not isinstance(v, str) else v) for c, v in Xte.iloc[i].items()},
                                       "probability": float(proba(xgbp, Xte.iloc[[i]])[0]),
                                       "contributions": dict(zip(nm, map(float, sv[i]))), "base_value": float(np.ravel(ex.expected_value)[0])}
log("exp4 done")

# ------------------------------------------------------------------ summary figures
names_all = BASE + ["Soft Voting", "Stacking", "BMA"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, eng in zip(ax, (False, True)):
    vals = [R["exp1"][FS[eng]][n]["pr_auc"] if n in BASE else R["exp2"][FS[eng]][n]["pr_auc"] for n in names_all]
    ci = [R["ci"][FS[eng]].get(n, {}).get("pr_auc_ci") for n in names_all]
    err = np.array([[v - c[0], c[1] - v] if c else [0, 0] for v, c in zip(vals, ci)]).T
    colors = [COL[0]] * len(BASE) + [COL[1]] * 3
    a.barh(names_all, vals, color=colors, xerr=err, error_kw={"ecolor": "#52514e", "lw": 1, "capsize": 2})
    for y_, v in enumerate(vals): a.text(0.02, y_, f"{v:.3f}", va="center", color="white", fontsize=8, fontweight="bold")
    a.set_xlim(0, 1.05); a.invert_yaxis(); a.set_xlabel("Test PR-AUC (95% bootstrap CI where shown)"); a.set_title(f"{FS[eng].capitalize()} features")
fig.tight_layout(); fig.savefig(FIG / "pr_auc_comparison.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k, name in enumerate(("CART", "XGBoost", "GAM", "BMA")):
    p_te, p_cal, _ = CAL[(False, name)]
    _, yte = DATA[False]["test"]
    fp, mp = calibration_curve(yte, p_te, n_bins=8, strategy="quantile")
    ax[0].plot(mp, fp, "-o", ms=3, color=COL[k], label=name)
    fp, mp = calibration_curve(yte, p_cal, n_bins=8, strategy="quantile")
    ax[1].plot(mp, fp, "-o", ms=3, color=COL[k], label=name)
for a, t in zip(ax, ("(a) Before calibration", "(b) After Platt scaling on validation")):
    a.plot([0, 1], [0, 1], ls=":", color="#898781"); a.set_xlabel("Mean predicted probability"); a.set_ylabel("Observed failure frequency")
    a.set_title(t); a.legend(frameon=False, fontsize=8); a.set_xscale("symlog", linthresh=0.05); a.set_yscale("symlog", linthresh=0.05)
fig.tight_layout(); fig.savefig(FIG / "calibration.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
w = R["bma"]
for a, eng in zip(ax, ("raw", "engineered")):
    ww = w[eng]["weights"]; a.bar(range(len(ww)), list(ww.values()), color=COL[0])
    a.set_xticks(range(len(ww)), [n.replace(" ", "\n") for n in ww], fontsize=7.5); a.set_ylim(0, 1.05)
    a.set_ylabel("Posterior model probability"); a.set_title(f"BMA weights ({eng} features)")
    for i, v in enumerate(ww.values()): a.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "bma_weights.png"); plt.close(fig)

ent, dis, wrong = UNC_RAW
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for meas, c in (("entropy", COL[0]), ("disagreement", COL[1])):
    rows = R["exp6"]["raw"]["coverage"][meas]
    ax[0].plot([r["coverage"] * 100 for r in rows], [r["error_rate"] * 100 for r in rows], "-o", color=c, label=meas, ms=4)
ax[0].invert_xaxis(); ax[0].set_xlabel("Coverage: predictions kept (%)"); ax[0].set_ylabel("Error rate on kept (%)")
ax[0].set_title("(a) Risk-coverage, calibrated BMA (raw)"); ax[0].legend(frameon=False)
ax[1].hist(ent[~wrong], bins=30, color=COL[0], alpha=0.7, density=True, label="correct")
ax[1].hist(ent[wrong], bins=30, color=COL[1], alpha=0.7, density=True, label="wrong")
ax[1].set_xlabel("Predictive entropy (bits)"); ax[1].set_ylabel("density"); ax[1].set_title("(b) Entropy of correct vs wrong predictions"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "uncertainty.png"); plt.close(fig)

# CART criterion x depth figure
fig, ax = plt.subplots(figsize=(5.4, 3.2))
for k, c in enumerate(("gini", "entropy")):
    d = R["tuning"]["raw"]["cart_criterion_depth"]
    xs = ["3", "4", "5", "6", "8", "10", "None"]
    ax.plot(xs, [d[f"{c}|{x}"] for x in xs], "-o", ms=4, color=COL[k], label=c)
ax.set_xlabel("max_depth"); ax.set_ylabel("CV PR-AUC (best over leaf size, ccp_alpha)"); ax.set_title("CART tuning, raw features"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "cart_tuning.png"); plt.close(fig)

R["runtime_seconds"] = time.time() - T0
(OUT / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
log("saved", OUT / "results.json")
