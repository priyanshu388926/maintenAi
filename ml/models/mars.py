"""Multivariate Adaptive Regression Splines (Friedman, 1991) with a logistic link, implemented in NumPy.

Why our own implementation: the usual Python package (sklearn-contrib-py-earth) no longer builds on current Python.
This follows Friedman's algorithm and the approach of R's `earth` package for binary targets:

1. Forward pass: start from the intercept B_0(x) = 1. At each step consider every existing basis function B_m,
   every input variable v not already in B_m, and every candidate knot t, and add the *pair* of hinge functions
       B_m(x) * max(0, x_v - t)   and   B_m(x) * max(0, t - x_v)
   that most reduces the residual sum of squares of a least-squares fit to the 0/1 target.
2. Backward pass: repeatedly delete the term whose removal gives the lowest Generalised Cross-Validation score
       GCV = RSS / N / (1 - C(M)/N)^2,   C(M) = M + d * (M - 1) / 2,  d = 3 (Friedman's default for interactions)
   and keep the subset with the best GCV.
3. Fit a logistic regression on the selected basis functions so outputs are probabilities
   (as `earth(..., glm = list(family = binomial))` does).

The model is   logit P(failure | x) = beta_0 + sum_m beta_m B_m(x).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Hinge:
    var: int
    knot: float
    sign: int  # +1: max(0, x - t); -1: max(0, t - x)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.sign * (X[:, self.var] - self.knot))

    def describe(self, names) -> str:
        n = names[self.var]
        return f"max(0, {n} - {self.knot:.3g})" if self.sign > 0 else f"max(0, {self.knot:.3g} - {n})"


Term = tuple  # tuple of Hinge (empty tuple = intercept)


def _eval(term: Term, X: np.ndarray) -> np.ndarray:
    out = np.ones(X.shape[0])
    for h in term:
        out = out * h(X)
    return out


def _rss(B: np.ndarray, y: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    r = y - B @ coef
    return float(r @ r)


class MARSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_terms: int = 21, max_degree: int = 2, n_knots: int = 20, penalty: float = 3.0,
                 C: float = 10.0, feature_names=None):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.n_knots = n_knots
        self.penalty = penalty
        self.C = C
        self.feature_names = feature_names

    def _knots(self, X):
        ks = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            ks.append(u if len(u) <= self.n_knots else np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, self.n_knots))))
        return ks

    def _gcv(self, rss, n, m):
        c = m + self.penalty * (m - 1) / 2
        return rss / n / max(1e-12, (1 - c / n)) ** 2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.classes_ = np.array([0, 1])
        knots = self._knots(X)
        terms: list[Term] = [()]
        B = np.ones((n, 1))
        # ---------------- forward pass
        while len(terms) + 2 <= self.max_terms:
            best = None
            for t in terms:
                if len(t) >= self.max_degree:
                    continue
                used = {h.var for h in t}
                parent = _eval(t, X)
                if not parent.any():
                    continue
                for v in range(p):
                    if v in used:
                        continue
                    for k in knots[v]:
                        h1, h2 = Hinge(v, float(k), 1), Hinge(v, float(k), -1)
                        c1, c2 = parent * h1(X), parent * h2(X)
                        if not c1.any() or not c2.any():
                            continue
                        rss = _rss(np.column_stack([B, c1, c2]), y)
                        if best is None or rss < best[0]:
                            best = (rss, (t + (h1,), t + (h2,)), c1, c2)
            if best is None:
                break
            _, new_terms, c1, c2 = best
            terms.extend(new_terms)
            B = np.column_stack([B, c1, c2])
        # ---------------- backward pass (GCV)
        active = list(range(len(terms)))
        best_set, best_gcv = list(active), self._gcv(_rss(B, y), n, len(active))
        while len(active) > 1:
            cand = []
            for j in active[1:]:  # never drop the intercept
                keep = [a for a in active if a != j]
                cand.append((self._gcv(_rss(B[:, keep], y), n, len(keep)), keep))
            g, keep = min(cand, key=lambda c: c[0])
            active = keep
            if g < best_gcv:
                best_gcv, best_set = g, list(keep)
        self.terms_ = [terms[i] for i in best_set]
        self.gcv_ = best_gcv
        self.forward_terms_ = len(terms)
        # ---------------- logistic link on the selected basis
        self.glm_ = LogisticRegression(C=self.C, max_iter=5000).fit(self._basis(X), y.astype(int))
        return self

    def _basis(self, X):
        X = np.asarray(X, dtype=float)
        cols = [_eval(t, X) for t in self.terms_ if t]
        return np.column_stack(cols) if cols else np.zeros((X.shape[0], 1))

    def predict_proba(self, X):
        return self.glm_.predict_proba(self._basis(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def describe(self) -> list[str]:
        names = self.feature_names or [f"x{j}" for j in range(100)]
        out = []
        for t, c in zip([t for t in self.terms_ if t], self.glm_.coef_[0]):
            out.append(f"{c:+.3f} * " + " * ".join(h.describe(names) for h in t))
        return [f"{self.glm_.intercept_[0]:+.3f}"] + out
