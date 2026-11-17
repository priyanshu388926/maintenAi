"""Dataset summaries for the dashboard and data explorer (computed once from the raw CSV and cached)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ml.data import schema as S
from ml.data.ingestion import load_ai4i
from ml.data.preprocessing import DerivedFeatures

NUMERIC = list(S.NUMERIC_FEATURES)
DERIVED = {"temp_diff_K": "Temperature difference [K]", "power_W": "Power [W]", "wear_torque_minNm": "Tool wear × torque [min·Nm]"}


@lru_cache(maxsize=1)
def frame() -> pd.DataFrame:
    df, _ = load_ai4i()
    d = DerivedFeatures().transform(df[list(S.INPUT_FEATURES)])
    for c in DERIVED:
        df[c] = d[c]
    return df


def summary() -> dict:
    df = frame()
    y = df[S.MACHINE_FAILURE]
    stats = []
    for c in NUMERIC + list(DERIVED):
        stats.append({"feature": DERIVED.get(c, c), "key": c, "mean": float(df[c].mean()), "std": float(df[c].std()),
                      "min": float(df[c].min()), "median": float(df[c].median()), "max": float(df[c].max()),
                      "mean_failure": float(df.loc[y == 1, c].mean()), "mean_normal": float(df.loc[y == 0, c].mean())})
    types = [{"type": t, "rows": int((df[S.TYPE] == t).sum()), "failures": int(y[df[S.TYPE] == t].sum()),
              "failure_rate": float(y[df[S.TYPE] == t].mean())} for t in S.TYPE_LEVELS]
    modes = [{"mode": m, "count": int(df[m].sum()),
              "name": {"TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power", "OSF": "Overstrain",
                       "RNF": "Random"}[m]} for m in S.FAILURE_MODES]
    quirks = {"failures_without_mode": int(((y == 1) & (df[list(S.FAILURE_MODES)].sum(axis=1) == 0)).sum()),
              "rnf_not_failure": int(((df["RNF"] == 1) & (y == 0)).sum()),
              "multi_mode": int((df[list(S.FAILURE_MODES)].sum(axis=1) > 1).sum())}
    return {"rows": int(len(df)), "columns": len(S.EXPECTED_COLUMNS), "failures": int(y.sum()),
            "failure_rate": float(y.mean()), "types": types, "modes": modes, "stats": stats, "quirks": quirks,
            "synthetic_note": "AI4I 2020 is a synthetic dataset; its failures follow documented threshold rules."}


def histograms(feature: str, bins: int = 30) -> dict:
    df = frame()
    if feature not in NUMERIC + list(DERIVED):
        raise KeyError(feature)
    x = df[feature].values
    edges = np.histogram_bin_edges(x, bins=bins)
    y = df[S.MACHINE_FAILURE].values
    n0, _ = np.histogram(x[y == 0], edges)
    n1, _ = np.histogram(x[y == 1], edges)
    return {"feature": DERIVED.get(feature, feature), "edges": edges.round(3).tolist(), "normal": n0.tolist(),
            "failure": n1.tolist(), "failure_rate": [float(b / (a + b)) if a + b else 0.0 for a, b in zip(n0, n1)]}


def scatter(x: str, y: str, n: int = 1500) -> dict:
    df = frame()
    for c in (x, y):
        if c not in NUMERIC + list(DERIVED):
            raise KeyError(c)
    fail = df[df[S.MACHINE_FAILURE] == 1]
    norm = df[df[S.MACHINE_FAILURE] == 0].sample(min(n, (df[S.MACHINE_FAILURE] == 0).sum()), random_state=42)
    pts = lambda d: [{"x": float(a), "y": float(b), "mode": next((m for m in S.FAILURE_MODES if r[m] == 1), "")}
                     for a, b, (_, r) in zip(d[x], d[y], d.iterrows())]
    return {"x": DERIVED.get(x, x), "y": DERIVED.get(y, y), "normal": pts(norm), "failure": pts(fail)}


def rows(offset: int = 0, limit: int = 50, failures_only: bool = False) -> dict:
    df = frame()
    d = df[df[S.MACHINE_FAILURE] == 1] if failures_only else df
    page = d.iloc[offset: offset + limit][list(S.EXPECTED_COLUMNS)]
    recs = page.astype(object).where(page.notna(), None).to_dict("records")
    return {"total": int(len(d)), "offset": offset, "rows": [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in recs]}
