"""PHYSIOLOGY family: what was happening to the patient (values only).

Input is the value matrix only. Output is the causal LOCF value matrix (NaN where never
measured yet); finalisation via DrawImputer happens inside the model pipeline so imputer
parameters are fit on training data only. No indicator/count/timing columns.
"""
from __future__ import annotations
import pandas as pd
from .config import PREFIX, VALUE_COLS
from .imputation import locf


def physiology_locf(values: pd.DataFrame, pid: pd.Series) -> pd.DataFrame:
    out = locf(values[VALUE_COLS], pid)
    out.columns = [PREFIX["physiology"] + c for c in VALUE_COLS]
    return out.astype("float32")
