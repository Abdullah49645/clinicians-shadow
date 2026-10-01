"""Assemble the three feature families and make their provenance explicit."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import BASE_COLS, LABEL, MODEL_FAMILIES, PREFIX, VALUE_COLS, Config
from .imputation import DrawImputer
from .physiology import physiology_locf
from .process import process_features


def base_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw[BASE_COLS].copy()
    out.columns = [PREFIX["base"] + c for c in BASE_COLS]
    return out.astype("float32")


def source_of(col: str) -> str:
    for fam, pre in PREFIX.items():
        if col.startswith(pre):
            return fam
    raise ValueError(f"column {col!r} has no feature-family prefix")


@dataclass
class FeatureTable:
    """Un-imputed feature families for one cohort (rows aligned across all members)."""
    pid: pd.Series
    iculos: pd.Series
    y: np.ndarray
    base: pd.DataFrame
    physiology: pd.DataFrame    # LOCF, NaN = not yet measured
    process: pd.DataFrame

    def take(self, rows: np.ndarray) -> "FeatureTable":
        return FeatureTable(self.pid.iloc[rows].reset_index(drop=True), self.iculos.iloc[rows].reset_index(drop=True),
                            self.y[rows], *(getattr(self, f).iloc[rows].reset_index(drop=True)
                                            for f in ("base", "physiology", "process")))


def build_feature_table(raw: pd.DataFrame, cfg: Config = Config()) -> FeatureTable:
    raw = raw.reset_index(drop=True)
    pid = raw["pid"]
    return FeatureTable(
        pid=pid, iculos=raw["ICULOS"], y=raw[LABEL].to_numpy(dtype=np.int8),
        base=base_features(raw),
        physiology=physiology_locf(raw[VALUE_COLS], pid),
        process=process_features(raw[VALUE_COLS].notna(), pid, cfg.hours_cap, cfg.window),
    )


def fit_imputer(train: FeatureTable, seed: int) -> DrawImputer:
    return DrawImputer(seed).fit(pd.concat([train.base, train.physiology], axis=1))


def assemble(t: FeatureTable, model: str, imp: DrawImputer) -> pd.DataFrame:
    """Model-ready matrix for `model`, containing exactly the families it is allowed."""
    fams = MODEL_FAMILIES[model]
    filled = imp.transform(pd.concat([t.base, t.physiology], axis=1), t.pid)
    parts = {"base": filled[t.base.columns], "physiology": filled[t.physiology.columns], "process": t.process}
    X = pd.concat([parts[f] for f in fams], axis=1)
    assert {source_of(c) for c in X.columns} == set(fams)
    return X
