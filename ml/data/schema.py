"""Schema of the UCI AI4I 2020 Predictive Maintenance dataset and the feature/target roles used by MaintenAI.

Single source of truth for column names, so that ingestion, preprocessing, models and the API all agree.
"""
from __future__ import annotations

# Raw column names exactly as they appear in ai4i2020.csv
UDI = "UDI"
PRODUCT_ID = "Product ID"
TYPE = "Type"
AIR_TEMP = "Air temperature [K]"
PROCESS_TEMP = "Process temperature [K]"
ROT_SPEED = "Rotational speed [rpm]"
TORQUE = "Torque [Nm]"
TOOL_WEAR = "Tool wear [min]"
MACHINE_FAILURE = "Machine failure"
FAILURE_MODES: tuple[str, ...] = ("TWF", "HDF", "PWF", "OSF", "RNF")

EXPECTED_COLUMNS: tuple[str, ...] = (
    UDI, PRODUCT_ID, TYPE, AIR_TEMP, PROCESS_TEMP, ROT_SPEED, TORQUE, TOOL_WEAR,
    MACHINE_FAILURE, *FAILURE_MODES,
)
EXPECTED_ROWS = 10_000

# Feature roles
IDENTIFIERS: tuple[str, ...] = (UDI, PRODUCT_ID)          # never used as predictive features
CATEGORICAL_FEATURES: tuple[str, ...] = (TYPE,)           # L / M / H product quality variant
NUMERIC_FEATURES: tuple[str, ...] = (AIR_TEMP, PROCESS_TEMP, ROT_SPEED, TORQUE, TOOL_WEAR)
INPUT_FEATURES: tuple[str, ...] = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)

# Columns that must never be inputs when the target is MACHINE_FAILURE (they are sub-labels of the target)
LEAKAGE_COLUMNS: tuple[str, ...] = FAILURE_MODES

TYPE_LEVELS: tuple[str, ...] = ("L", "M", "H")

# Plausible physical ranges used for schema validation (generous bounds around the documented generator)
VALUE_RANGES: dict[str, tuple[float, float]] = {
    AIR_TEMP: (280.0, 320.0),
    PROCESS_TEMP: (280.0, 330.0),
    ROT_SPEED: (0.0, 5000.0),
    TORQUE: (0.0, 150.0),
    TOOL_WEAR: (0.0, 400.0),
}

# MD5 of the copy inspected in Milestone 01 (obtained from two identical public GitHub mirrors)
REFERENCE_MD5 = "f0adaa9c5e25366c6df1cc2684479871"
