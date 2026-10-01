"""Patient-level splitting. Rows from one patient never straddle a partition."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assert_disjoint(pid: pd.Series, train_rows: np.ndarray, test_rows: np.ndarray) -> None:
    a, b = set(pid.iloc[train_rows].unique()), set(pid.iloc[test_rows].unique())
    if a & b:
        raise AssertionError(f"patient leakage: {len(a & b)} patients in both partitions")


def patient_folds(pid: pd.Series, y: np.ndarray, n_splits: int, seed: int):
    """Yield (train_rows, test_rows), stratified on 'patient ever septic', grouped by patient."""
    ever = pd.Series(y).groupby(pid.to_numpy()).transform("max").to_numpy()
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgk.split(np.zeros(len(y)), ever, groups=pid.to_numpy()):
        assert_disjoint(pid, tr, te)
        yield tr, te
