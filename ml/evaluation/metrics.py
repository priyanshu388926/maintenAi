"""Metric suite for imbalanced binary failure prediction, including probability-quality metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum_b (n_b / N) |mean(y)_b - mean(p)_b|  (Guo et al., 2017)."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def best_f1_threshold(y, p) -> float:
    """Threshold that maximises failure-class F1 (chosen on validation data only)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def evaluate(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "f1": f1_score(y, yhat, zero_division=0),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "fnr": fn / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_ci(y, p, threshold: float, n: int = 1000, seed: int = 42) -> dict:
    """95% percentile bootstrap intervals for PR-AUC and recall on a fixed test set."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pr, rc = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        pr.append(average_precision_score(y[i], p[i]))
        rc.append(recall_score(y[i], (p[i] >= threshold).astype(int)))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"pr_auc_ci": q(pr), "recall_ci": q(rc)}
