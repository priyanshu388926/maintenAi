"""Load the raw AI4I 2020 CSV, verify its integrity and return a validated DataFrame.

Usage:
    python -m ml.data.ingestion data/raw/ai4i2020.csv
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

from ml.data import schema as S
from ml.data.validation import ValidationReport, validate_schema

log = logging.getLogger(__name__)
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ai4i2020.csv"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ai4i(path: str | Path = DEFAULT_PATH, strict: bool = True) -> tuple[pd.DataFrame, ValidationReport]:
    """Read the raw CSV (handles the UTF-8 byte-order mark in the published file) and validate it.

    The raw file is never modified; callers receive a copy.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the AI4I 2020 dataset and place ai4i2020.csv in data/raw/ (see README).")
    digest = file_md5(path)
    if digest != S.REFERENCE_MD5:
        log.warning("MD5 %s differs from the reference copy %s; inspect before use.", digest, S.REFERENCE_MD5)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df[S.TYPE] = df[S.TYPE].astype("string")
    report = validate_schema(df)
    if strict and not report.ok:
        raise ValueError(f"Dataset failed validation: {report}")
    for w in report.warnings:
        log.info("validation: %s", w)
    return df, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    data, rep = load_ai4i(p)
    print(f"Loaded {rep.n_rows} rows x {rep.n_columns} columns from {p}")
    print(f"MD5: {file_md5(p)}  missing: {rep.missing_values}  duplicates: {rep.duplicate_rows}")
    print(f"Machine failure rate: {data[S.MACHINE_FAILURE].mean():.4f}")
    for w in rep.warnings:
        print("WARNING:", w)
