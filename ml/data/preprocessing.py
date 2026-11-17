"""Leakage-safe splitting, feature engineering and column transformers for AI4I 2020."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml.data import schema as S
from ml.data.validation import assert_no_leakage

SEED = 42
DERIVED = ("temp_diff_K", "power_W", "wear_torque_minNm")


def stratified_split(df: pd.DataFrame, target: str = S.MACHINE_FAILURE, test: float = 0.15, val: float = 0.15,
                     seed: int = SEED) -> dict[str, np.ndarray]:
    """70/15/15 stratified train/validation/test split returning row indices."""
    idx_trval, idx_test = train_test_split(df.index.values, test_size=test, stratify=df[target], random_state=seed)
    idx_train, idx_val = train_test_split(idx_trval, test_size=val / (1 - test),
                                          stratify=df.loc[idx_trval, target], random_state=seed)
    return {"train": np.sort(idx_train), "validation": np.sort(idx_val), "test": np.sort(idx_test)}


def save_split(split: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.tolist() for k, v in split.items()}))


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Adds physically motivated quantities. Stateless: no information is learned from the data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["temp_diff_K"] = X[S.PROCESS_TEMP] - X[S.AIR_TEMP]
        X["power_W"] = X[S.TORQUE] * X[S.ROT_SPEED] * 2 * np.pi / 60
        X["wear_torque_minNm"] = X[S.TOOL_WEAR] * X[S.TORQUE]
        return X


def feature_columns(engineered: bool) -> list[str]:
    cols = [*S.INPUT_FEATURES, *(DERIVED if engineered else ())]
    return cols


def make_xy(df: pd.DataFrame, idx, engineered: bool, target: str = S.MACHINE_FAILURE):
    X = df.loc[idx, list(S.INPUT_FEATURES)]
    if engineered:
        X = DerivedFeatures().transform(X)
    assert_no_leakage(X.columns, target)
    return X.reset_index(drop=True), df.loc[idx, target].to_numpy()


def numeric_columns(engineered: bool) -> list[str]:
    return [*S.NUMERIC_FEATURES, *(DERIVED if engineered else ())]


def tree_preprocessor(engineered: bool) -> ColumnTransformer:
    """Trees: ordinal Type (L<M<H), numeric features untouched so thresholds stay in physical units."""
    return ColumnTransformer([
        ("type", OrdinalEncoder(categories=[list(S.TYPE_LEVELS)]), [S.TYPE]),
        ("num", "passthrough", numeric_columns(engineered)),
    ], verbose_feature_names_out=False)


def scaled_preprocessor(engineered: bool) -> ColumnTransformer:
    """Smooth models (GAM, MARS, logistic): one-hot Type, standardised numeric features."""
    return ColumnTransformer([
        ("type", OneHotEncoder(categories=[list(S.TYPE_LEVELS)], drop="first", sparse_output=False), [S.TYPE]),
        ("num", StandardScaler(), numeric_columns(engineered)),
    ], verbose_feature_names_out=False)
