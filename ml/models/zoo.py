"""Factory for every base model, each wrapped in its own leakage-safe preprocessing pipeline."""
from __future__ import annotations

import numpy as np
from pygam import LogisticGAM, l, s
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml.data import preprocessing as P
from ml.models.mars import MARSClassifier

SEED = P.SEED


class GAMClassifier(BaseEstimator, ClassifierMixin):
    """Logistic GAM: logit P(failure) = b0 + sum_j f_j(x_j) with penalised cubic splines (pyGAM).

    The first `n_linear` columns (one-hot product type) enter linearly; the rest get a smooth spline term.
    The smoothing penalty lambda is chosen by pyGAM's UBRE/GCV grid search on the training data.
    """

    def __init__(self, n_linear: int = 2, n_splines: int = 12, lam_grid=(0.01, 0.1, 1, 10, 100)):
        self.n_linear = n_linear
        self.n_splines = n_splines
        self.lam_grid = lam_grid

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        terms = None
        for j in range(X.shape[1]):
            t = l(j) if j < self.n_linear else s(j, n_splines=self.n_splines)
            terms = t if terms is None else terms + t
        self.gam_ = LogisticGAM(terms).gridsearch(X, np.asarray(y), lam=np.array(self.lam_grid), progress=False)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.gam_.predict_proba(np.asarray(X, dtype=float))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def base_models(engineered: bool, cart_params: dict | None = None, xgb_params: dict | None = None) -> dict:
    tree_pre = lambda: P.tree_preprocessor(engineered)
    smooth_pre = lambda: P.scaled_preprocessor(engineered)
    cart_params = cart_params or {"max_depth": 6, "criterion": "gini", "ccp_alpha": 0.0}
    xgb_params = xgb_params or {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}
    return {
        "CART": Pipeline([("prep", tree_pre()), ("model", DecisionTreeClassifier(random_state=SEED, **cart_params))]),
        "AdaBoost": Pipeline([("prep", tree_pre()), ("model", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2), n_estimators=300, learning_rate=0.5, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("prep", tree_pre()), ("model", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED))]),
        "XGBoost": Pipeline([("prep", tree_pre()), ("model", XGBClassifier(
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=2, **xgb_params))]),
        "MARS": Pipeline([("prep", smooth_pre()), ("model", MARSClassifier(max_terms=21, max_degree=2))]),
        "GAM": Pipeline([("prep", smooth_pre()), ("model", GAMClassifier(n_linear=2))]),
        "Random Forest": Pipeline([("prep", tree_pre()), ("model", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=1, n_jobs=2, random_state=SEED))]),
    }
