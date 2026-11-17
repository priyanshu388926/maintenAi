"""Tests for preprocessing, metrics, MARS and Bayesian Model Averaging."""
import numpy as np
import pandas as pd
import pytest

from ml.data import preprocessing as P
from ml.data import schema as S
from ml.evaluation.metrics import best_f1_threshold, evaluate, expected_calibration_error
from ml.models.bma import BayesianModelAveraging, bernoulli_loglik
from ml.models.mars import MARSClassifier


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = ((X[:, 0] > 0.8) | (X[:, 1] * X[:, 2] > 1.0)).astype(int)
    return X, y


def test_split_is_disjoint_stratified_and_complete():
    df = pd.DataFrame({S.MACHINE_FAILURE: [1] * 100 + [0] * 900})
    sp = P.stratified_split(df)
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == 1000 and len(np.unique(allidx)) == 1000
    assert [len(sp[k]) for k in ("train", "validation", "test")] == [700, 150, 150]
    for k in sp:
        assert df.loc[sp[k], S.MACHINE_FAILURE].mean() == pytest.approx(0.1, abs=0.01)


def test_derived_features_values():
    X = pd.DataFrame({S.AIR_TEMP: [300.0], S.PROCESS_TEMP: [309.0], S.TORQUE: [40.0], S.ROT_SPEED: [1500],
                      S.TOOL_WEAR: [100]})
    Z = P.DerivedFeatures().transform(X)
    assert Z["temp_diff_K"].iloc[0] == pytest.approx(9.0)
    assert Z["power_W"].iloc[0] == pytest.approx(40 * 1500 * 2 * np.pi / 60)
    assert Z["wear_torque_minNm"].iloc[0] == pytest.approx(4000)


def test_make_xy_never_returns_leaky_columns():
    df = pd.DataFrame({c: [0, 1] * 5 for c in S.EXPECTED_COLUMNS})
    df[S.TYPE] = "L"
    X, _ = P.make_xy(df, df.index, engineered=True)
    assert not set(X.columns) & {*S.FAILURE_MODES, S.UDI, S.PRODUCT_ID, S.MACHINE_FAILURE}


def test_mars_probabilities_valid_and_better_than_chance():
    X, y = _toy()
    m = MARSClassifier(max_terms=11, n_knots=10).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert np.all((p >= 0) & (p <= 1))
    assert ((p >= 0.5) == y).mean() > 0.9
    assert len(m.terms_) <= 11


def test_bma_weights_form_a_distribution_and_favour_better_model():
    y = np.array([0, 0, 1, 1] * 50)
    good = np.where(y == 1, 0.9, 0.1); bad = np.full_like(good, 0.5)
    bma = BayesianModelAveraging().fit({"good": good, "bad": bad}, y)
    w = bma.weights()
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["good"] > 0.99
    p = bma.predict_proba({"good": good, "bad": bad})
    assert np.all((p >= 0) & (p <= 1))


def test_bma_uniform_when_models_identical():
    y = np.array([0, 1] * 20); p = np.full(40, 0.4)
    w = BayesianModelAveraging().fit({"a": p, "b": p.copy()}, y).weights()
    assert w["a"] == pytest.approx(0.5)


def test_loglik_and_metrics():
    y = np.array([0, 1]); assert bernoulli_loglik(y, np.array([0.5, 0.5])) == pytest.approx(2 * np.log(0.5))
    assert expected_calibration_error(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0)
    yv = np.array([0, 0, 0, 1, 1]); pv = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    t = best_f1_threshold(yv, pv); r = evaluate(yv, pv, t)
    assert r["recall"] == 1.0 and r["tp"] + r["fn"] == 2
