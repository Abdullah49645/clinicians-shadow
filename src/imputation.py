"""Causal imputation for the physiology family.

Design goal: the physiology representation must not encode *whether/when* something was
measured. Two steps, both causal:
 1. LOCF (last observation carried forward) -- uses only past values.
 2. Rows before a patient's first measurement of a variable are filled with a
    per-(patient, variable) *draw* from the training distribution (deterministic hash
    of seed/pid/variable). A single constant (e.g. the median) would act as a sentinel
    for "not yet measured" that a tree could split on; random draws avoid a shared
    sentinel. No missingness-indicator columns are ever created.
Known residual: LOCF repeats stale values, so very weak timing information can remain
(see README, Limitations, and the missingness-leak audit in src/audit.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_M = np.uint64(0xFFFFFFFFFFFFFFFF)


def locf(df: pd.DataFrame, pid: pd.Series) -> pd.DataFrame:
    """Causal forward fill within patient. Remaining NaN = never measured yet."""
    return df.groupby(pid.to_numpy(), sort=False).ffill()


def _u01(seed: int, pid: np.ndarray, j: int) -> np.ndarray:
    """Deterministic uniform(0,1) per (seed, pid, column j) via splitmix64."""
    with np.errstate(over="ignore"):
        x = (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
             + pid.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
             + np.uint64(j + 1) * np.uint64(0x94D049BB133111EB))
        x ^= x >> np.uint64(30); x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27); x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class DrawImputer:
    """Fit on TRAIN rows only; transform any partition."""

    def __init__(self, seed: int = 42, n_q: int = 201):
        self.seed, self.n_q = seed, n_q
        self.grids: dict[str, np.ndarray] = {}

    def fit(self, locf_df: pd.DataFrame) -> "DrawImputer":
        qs = np.linspace(0, 1, self.n_q)
        for c in locf_df.columns:
            v = locf_df[c].dropna().to_numpy()
            self.grids[c] = np.quantile(v, qs) if len(v) else np.zeros(self.n_q)
        return self

    def transform(self, locf_df: pd.DataFrame, pid: pd.Series) -> pd.DataFrame:
        p = pid.to_numpy()
        out = {}
        for j, c in enumerate(locf_df.columns):
            col = locf_df[c].to_numpy(dtype=np.float64)
            nan = np.isnan(col)
            if nan.any():
                draw = np.interp(_u01(self.seed, p, j), np.linspace(0, 1, self.n_q), self.grids[c])
                col = np.where(nan, draw, col)
            out[c] = col.astype(np.float32)
        return pd.DataFrame(out, index=locf_df.index)
