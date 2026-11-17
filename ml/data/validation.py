"""Schema validation and target-leakage guards for the AI4I 2020 dataset."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from ml.data import schema as S


class SchemaError(ValueError):
    """Raised when the raw dataset does not match the expected AI4I schema."""


class LeakageError(ValueError):
    """Raised when a forbidden column is used as a model input."""


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    missing_values: int
    duplicate_rows: int
    duplicate_udi: int
    out_of_range: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.out_of_range and self.missing_values == 0


def validate_schema(df: pd.DataFrame, expected_rows: int | None = S.EXPECTED_ROWS) -> ValidationReport:
    """Check columns, types, value ranges and label consistency. Raises SchemaError on structural problems."""
    missing_cols = [c for c in S.EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in S.EXPECTED_COLUMNS]
    if missing_cols:
        raise SchemaError(f"Missing columns: {missing_cols}")

    for col in (*S.NUMERIC_FEATURES, S.MACHINE_FAILURE, *S.FAILURE_MODES):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"Column {col!r} must be numeric, got {df[col].dtype}")
    for col in (S.MACHINE_FAILURE, *S.FAILURE_MODES):
        bad = set(df[col].dropna().unique()) - {0, 1}
        if bad:
            raise SchemaError(f"Column {col!r} must be binary 0/1, found {sorted(bad)}")
    bad_types = set(df[S.TYPE].dropna().unique()) - set(S.TYPE_LEVELS)
    if bad_types:
        raise SchemaError(f"Unexpected product types: {sorted(bad_types)}")

    report = ValidationReport(
        n_rows=len(df), n_columns=df.shape[1],
        missing_values=int(df[list(S.EXPECTED_COLUMNS)].isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_udi=int(df[S.UDI].duplicated().sum()),
    )
    for col, (lo, hi) in S.VALUE_RANGES.items():
        n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_bad:
            report.out_of_range[col] = n_bad

    if extra_cols:
        report.warnings.append(f"Unexpected extra columns ignored: {extra_cols}")
    if expected_rows is not None and len(df) != expected_rows:
        report.warnings.append(f"Expected {expected_rows} rows, found {len(df)}")

    # Known label inconsistencies of AI4I (documented, not errors): reported so they are never silently missed
    n_modes = df[list(S.FAILURE_MODES)].sum(axis=1)
    fail_no_mode = int(((df[S.MACHINE_FAILURE] == 1) & (n_modes == 0)).sum())
    mode_no_fail = int(((df[S.MACHINE_FAILURE] == 0) & (n_modes > 0)).sum())
    if fail_no_mode:
        report.warnings.append(f"{fail_no_mode} rows have Machine failure = 1 but no failure-mode flag")
    if mode_no_fail:
        report.warnings.append(f"{mode_no_fail} rows have a failure-mode flag but Machine failure = 0")
    return report


def assert_no_leakage(feature_columns: Iterable[str], target: str = S.MACHINE_FAILURE) -> None:
    """Refuse feature sets that contain identifiers, the target itself, or (for Machine failure) the failure modes."""
    cols = list(feature_columns)
    forbidden = set(S.IDENTIFIERS) | {target}
    if target == S.MACHINE_FAILURE:
        forbidden |= set(S.LEAKAGE_COLUMNS)
    elif target in S.FAILURE_MODES:
        # Predicting one mode: the overall label and the other modes are derived from the same events
        forbidden |= {S.MACHINE_FAILURE, *S.FAILURE_MODES}
    leaked = sorted(set(cols) & forbidden)
    if leaked:
        raise LeakageError(f"Forbidden input columns for target {target!r}: {leaked}")
