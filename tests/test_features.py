"""Test A (physiology has no process info) and Test B (process has no values)."""
import numpy as np
import pandas as pd
from src.config import MODEL_FAMILIES, PREFIX, VALUE_COLS, Config
from src.features import assemble, build_feature_table, fit_imputer, source_of
from src.physiology import physiology_locf
from src.process import process_features

PROCESS_WORDS = ("obs__", "cnt__", "since__", "agg__", "mask", "missing", "nan")


def test_families_have_disjoint_prefixes(table):
    cols = {f: set(getattr(table, f).columns) for f in PREFIX}
    assert not (cols["base"] & cols["physiology"]) and not (cols["physiology"] & cols["process"])
    for f, c in cols.items():
        assert all(source_of(x) == f for x in c)


def test_A_physiology_has_no_process_columns_or_nans(table):
    imp = fit_imputer(table, 0)
    X = assemble(table, "M_V", imp)
    assert not X.isna().any().any()
    phys = [c for c in X.columns if source_of(c) == "physiology"]
    assert phys == [PREFIX["physiology"] + v for v in VALUE_COLS]       # values only, nothing derived
    assert not any(w in c for c in phys for w in PROCESS_WORDS)


def _patient(hr):
    n = len(hr)
    return pd.DataFrame({c: np.nan for c in VALUE_COLS} | {"HR": hr}, index=range(n)), pd.Series([1] * n)


def test_A_physiology_invariant_to_redundant_measurements():
    """Re-measuring an unchanged value (more observation) must not alter physiology features."""
    dense, pid = _patient([80.0] * 8)
    thin, _ = _patient([80.0] + [np.nan] * 7)
    a, b = physiology_locf(dense, pid), physiology_locf(thin, pid)
    pd.testing.assert_frame_equal(a, b)
    # ...while the process family DOES see the difference
    pa, pb = process_features(dense.notna(), pid), process_features(thin.notna(), pid)
    assert not pa.equals(pb)


def test_A_prefirst_fill_is_not_a_shared_sentinel(table):
    imp = fit_imputer(table, 0)
    filled = imp.transform(pd.concat([table.base, table.physiology], axis=1), table.pid)
    unfilled = table.physiology["v__Lactate"].isna().to_numpy()
    assert unfilled.any()
    assert filled.loc[unfilled, "v__Lactate"].nunique() > 5


def test_B_process_ignores_values(raw):
    pid = raw["pid"]
    mask = raw[VALUE_COLS].notna()
    rng = np.random.default_rng(0)
    scrambled = pd.DataFrame(np.where(mask, rng.normal(size=mask.shape) * 1e6, np.nan),
                             columns=VALUE_COLS, index=mask.index)               # new values, same mask
    pd.testing.assert_frame_equal(process_features(mask, pid), process_features(scrambled.notna(), pid))


def test_B_model_matrices_respect_families(table):
    imp = fit_imputer(table, 0)
    for m, fams in MODEL_FAMILIES.items():
        X = assemble(table, m, imp)
        assert {source_of(c) for c in X.columns} == set(fams)
    assert not any(source_of(c) == "physiology" for c in assemble(table, "M_P", imp).columns)
    assert not any(source_of(c) == "process" for c in assemble(table, "M_V", imp).columns)
