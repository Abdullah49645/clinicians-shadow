"""PROCESS family: what clinicians chose to measure, and when.

The ONLY input is the boolean observation mask (+ patient id for grouping). The function
signature makes it impossible to read a physiological value. All features at row t use
rows <= t of the same patient (cumsum, ffill, shift(k>0)); "time" is row position within
the patient's record, so ICULOS stays exclusively in the base family.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import LABS, VALUE_COLS, VITALS, PREFIX

P = PREFIX["process"]


def process_features(mask: pd.DataFrame, pid: pd.Series, cap: int = 72, window: int = 6) -> pd.DataFrame:
    mask = mask[VALUE_COLS].astype(bool)
    keys = pid.to_numpy()
    pos = pd.Series(keys).groupby(keys, sort=False).cumcount().to_numpy()   # 0-based, past-only
    m = mask.to_numpy()
    cnt = pd.DataFrame(m.astype(np.int32), columns=VALUE_COLS).groupby(keys, sort=False).cumsum()
    last = pd.DataFrame(np.where(m, pos[:, None], np.nan), columns=VALUE_COLS).groupby(keys, sort=False).ffill()
    since = pos[:, None] - last.to_numpy()
    since = np.where(np.isnan(since), cap, np.minimum(since, cap))
    prev = cnt.groupby(keys, sort=False).shift(window).fillna(0)
    c6 = (cnt.to_numpy() - prev.to_numpy())
    cn = cnt.to_numpy()
    iv = [VALUE_COLS.index(c) for c in VITALS]
    il = [VALUE_COLS.index(c) for c in LABS]
    n = len(VALUE_COLS)
    denom6 = n * np.minimum(window, pos + 1)

    d: dict[str, np.ndarray] = {}
    for j, c in enumerate(VALUE_COLS):
        d[f"{P}obs__{c}"] = m[:, j].astype(np.float32)
        d[f"{P}since__{c}"] = since[:, j].astype(np.float32)
        d[f"{P}cnt__{c}"] = cn[:, j].astype(np.float32)
    d[f"{P}agg__n_obs_now"] = m.sum(1)
    d[f"{P}agg__n_vitals_now"] = m[:, iv].sum(1)
    d[f"{P}agg__n_labs_now"] = m[:, il].sum(1)           # lab panel size
    d[f"{P}agg__n_vars_ever"] = (cn > 0).sum(1)
    d[f"{P}agg__obs_6h_total"] = c6.sum(1)
    d[f"{P}agg__density_6h"] = c6.sum(1) / denom6
    d[f"{P}agg__rate_cum"] = cn.sum(1) / (n * (pos + 1))
    d[f"{P}agg__since_any"] = since.min(1)
    d[f"{P}agg__since_any_lab"] = since[:, il].min(1)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in d.items()}, index=mask.index)
