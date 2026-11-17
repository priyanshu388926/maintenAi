"""Milestone 01 tests: dataset integrity, schema validation and leakage guards."""
from pathlib import Path

import pandas as pd
import pytest

from ml.data import schema as S
from ml.data.ingestion import DEFAULT_PATH, load_ai4i
from ml.data.validation import LeakageError, SchemaError, assert_no_leakage, validate_schema

needs_data = pytest.mark.skipif(not Path(DEFAULT_PATH).exists(), reason="data/raw/ai4i2020.csv not present")


@needs_data
def test_dataset_loads_with_expected_shape_and_labels():
    df, rep = load_ai4i()
    assert rep.ok
    assert df.shape == (10_000, 14)
    assert tuple(df.columns) == S.EXPECTED_COLUMNS
    assert int(df[S.MACHINE_FAILURE].sum()) == 339
    assert df[S.TYPE].value_counts().to_dict() == {"L": 6000, "M": 2997, "H": 1003}


@needs_data
def test_known_label_inconsistencies_are_reported():
    _, rep = load_ai4i()
    assert any("no failure-mode flag" in w for w in rep.warnings)
    assert any("Machine failure = 0" in w for w in rep.warnings)


def test_missing_column_raises():
    df = pd.DataFrame({c: [0] for c in S.EXPECTED_COLUMNS if c != S.TORQUE})
    with pytest.raises(SchemaError):
        validate_schema(df, expected_rows=None)


def test_non_binary_label_raises():
    row = {c: 0 for c in S.EXPECTED_COLUMNS}
    row.update({S.TYPE: "L", S.PRODUCT_ID: "L1", S.MACHINE_FAILURE: 2, S.AIR_TEMP: 300, S.PROCESS_TEMP: 310,
                S.ROT_SPEED: 1500, S.TORQUE: 40})
    with pytest.raises(SchemaError):
        validate_schema(pd.DataFrame([row]), expected_rows=None)


def test_valid_feature_set_passes_leakage_guard():
    assert_no_leakage(S.INPUT_FEATURES, target=S.MACHINE_FAILURE)


@pytest.mark.parametrize("bad", ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID", "Machine failure"])
def test_leakage_guard_rejects_forbidden_columns(bad):
    with pytest.raises(LeakageError):
        assert_no_leakage([*S.INPUT_FEATURES, bad], target=S.MACHINE_FAILURE)


def test_leakage_guard_for_failure_mode_target():
    with pytest.raises(LeakageError):
        assert_no_leakage([*S.INPUT_FEATURES, S.MACHINE_FAILURE], target="HDF")
