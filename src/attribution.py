"""Model-input group ablation: a model-based ASSOCIATIONAL diagnostic (not causal).

For the combined model M_VP, replace one feature family with a fixed reference (training
median of that family's columns) and measure the change in log-odds:
    c_P = logit(f(x)) - logit(f(x with PROCESS := ref))
    c_V = logit(f(x)) - logit(f(x with PHYSIOLOGY := ref))
PRI = c_P / (c_P + c_V) when both > 0, else undefined (None). Reference choice and feature
correlation make this model-dependent; ablated inputs are off-distribution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features import FeatureTable, assemble, source_of


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def family_reference(X_train: pd.DataFrame) -> pd.Series:
    return X_train.median()


def ablate(X: pd.DataFrame, family: str, ref: pd.Series) -> pd.DataFrame:
    X2 = X.copy()
    for c in X.columns:
        if source_of(c) == family:
            X2[c] = np.float32(ref[c])
    return X2


def contributions(est, X: pd.DataFrame, ref: pd.Series) -> dict[str, np.ndarray]:
    f = lambda M: est.predict_proba(M)[:, 1]
    p_all, p_noP, p_noV = f(X), f(ablate(X, "process", ref)), f(ablate(X, "physiology", ref))
    cP, cV = _logit(p_all) - _logit(p_noP), _logit(p_all) - _logit(p_noV)
    both = (cP > 0) & (cV > 0)
    pri = np.where(both, cP / np.where(both, cP + cV, 1), np.nan)
    return {"all": p_all, "physiology_only": p_noP, "process_only": p_noV,
            "c_process": cP, "c_physiology": cV, "pri": pri}
