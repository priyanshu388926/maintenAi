"""Bayesian Model Averaging over a fixed set of fitted classifiers.

Predictive distribution (Hoeting et al., 1999):
    P(y | x, D) = sum_m P(y | x, M_m, D) * P(M_m | D)

Posterior model probabilities:
    P(M_m | D) ∝ P(D | M_m) * P(M_m)

Approximation used here (stated explicitly): the marginal likelihood P(D | M_m) of a machine-learning model with
thousands of tree parameters cannot be computed exactly, and BIC needs a parameter count that is undefined for
tree ensembles. We therefore approximate log P(D | M_m) by the **cross-validated predictive log-likelihood**
    sum_i log p(y_i | x_i, M_m fitted without fold(i)),
i.e. out-of-fold Bernoulli log-likelihood on the training set. This is the log of the leave-fold-out predictive
evidence. With a uniform prior P(M_m) = 1/K the weights are a softmax of these log-likelihoods.

Caveat (Minka, 2000): BMA assumes one candidate is the true model; with N = 7,000 points the log-likelihood
differences are large, so the posterior typically concentrates on one model. The optional `temperature`
(default 1 = genuine BMA) is reported only as a sensitivity analysis, not as BMA.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def bernoulli_loglik(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


class BayesianModelAveraging:
    def __init__(self, prior: dict[str, float] | None = None, temperature: float = 1.0):
        self.prior = prior
        self.temperature = temperature

    def fit(self, oof_probs: dict[str, np.ndarray], y) -> "BayesianModelAveraging":
        names = list(oof_probs)
        loglik = np.array([bernoulli_loglik(y, oof_probs[n]) for n in names])
        prior = np.array([self.prior.get(n, 1.0) if self.prior else 1.0 for n in names])
        prior = prior / prior.sum()
        logpost = loglik / self.temperature + np.log(prior)
        logpost -= logpost.max()
        w = np.exp(logpost)
        self.names_, self.loglik_, self.weights_ = names, loglik, w / w.sum()
        return self

    def predict_proba(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        return sum(w * np.asarray(probs[n]) for n, w in zip(self.names_, self.weights_))

    def weights(self) -> dict[str, float]:
        return dict(zip(self.names_, map(float, self.weights_)))
