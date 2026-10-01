"""Features at row t depend only on rows <= t of that patient."""
import numpy as np
import pandas as pd
from src.config import Config
from src.features import build_feature_table, fit_imputer, assemble


def test_prefix_invariance(raw):
    cfg = Config()
    full = build_feature_table(raw, cfg)
    imp = fit_imputer(full, 0)
    Xf = assemble(full, "M_VP", imp)
    rng = np.random.default_rng(0)
    for p in rng.choice(raw["pid"].unique(), 8, replace=False):
        idx = np.where(raw["pid"].to_numpy() == p)[0]
        cut = int(rng.integers(2, len(idx)))
        sub = raw.iloc[idx[:cut]]                                  # future rows removed
        ts = build_feature_table(sub, cfg)
        Xs = assemble(ts, "M_VP", imp)
        pd.testing.assert_frame_equal(Xf.iloc[idx[:cut]].reset_index(drop=True), Xs, check_exact=False, atol=1e-6)


def test_future_edits_do_not_change_past_features(raw):
    cfg = Config()
    p = raw["pid"].iloc[0]
    idx = np.where(raw["pid"].to_numpy() == p)[0]
    cut = len(idx) // 2
    mod = raw.copy()
    from src.config import VALUE_COLS
    mod.loc[idx[cut:], VALUE_COLS] = np.where(np.isnan(mod.loc[idx[cut:], VALUE_COLS]), 1.0, np.nan)
    mod.loc[idx[cut:], "SepsisLabel"] = 1.0 - mod.loc[idx[cut:], "SepsisLabel"]
    a, b = build_feature_table(raw, cfg), build_feature_table(mod, cfg)
    for fam in ("base", "physiology", "process"):
        pd.testing.assert_frame_equal(getattr(a, fam).iloc[idx[:cut]], getattr(b, fam).iloc[idx[:cut]])
