"""Within-hospital (patient-grouped CV) and cross-hospital transfer experiments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Config, MODEL_FAMILIES
from .evaluate import full_report
from .features import FeatureTable, assemble, fit_imputer
from .models import make_model
from .splits import assert_disjoint, patient_folds

# (display name, model-set key, estimator kind). LR_V is Baseline A; M_V (hgb) is Baseline B.
SPECS = [("M_BASE", "M_BASE", "hgb"), ("M_V", "M_V", "hgb"), ("M_P", "M_P", "hgb"),
         ("M_VP", "M_VP", "hgb"), ("LR_V", "M_V", "logreg")]


def fit_model(train: FeatureTable, key: str, kind: str, cfg: Config):
    imp = fit_imputer(train, cfg.seed)
    X = assemble(train, key, imp)
    est = make_model(kind, cfg).fit(X, train.y)
    return est, imp


def predict(est, imp, t: FeatureTable, key: str) -> np.ndarray:
    return est.predict_proba(assemble(t, key, imp))[:, 1]


def within_hospital(t: FeatureTable, cfg: Config) -> dict[str, np.ndarray]:
    """Out-of-fold predictions for every spec (each patient scored only by models that never saw them)."""
    oof = {name: np.full(len(t.y), np.nan) for name, _, _ in SPECS}
    for tr, te in patient_folds(t.pid, t.y, cfg.n_splits, cfg.seed):
        a, b = t.take(tr), t.take(te)
        for name, key, kind in SPECS:
            est, imp = fit_model(a, key, kind, cfg)
            oof[name][te] = predict(est, imp, b, key)
    for v in oof.values():
        assert not np.isnan(v).any()
    return oof


def run_transfer(tables: dict[str, FeatureTable], cfg: Config):
    """Returns (metrics, trained) where metrics[spec][scenario] -> report dict."""
    metrics: dict[str, dict] = {n: {} for n, _, _ in SPECS}
    trained: dict[str, dict] = {}
    for h, t in tables.items():
        oof = within_hospital(t, cfg)
        for name, _, _ in SPECS:
            metrics[name][f"within_{h}"] = full_report(t.y, oof[name], t.pid.to_numpy(), cfg.n_boot, cfg.seed)
    for src, dst in (("A", "B"), ("B", "A")):
        tr, te = tables[src], tables[dst]
        assert not (set(tr.pid.unique()) & set(te.pid.unique())), "pid collision across hospitals"
        trained[f"{src}_to_{dst}"] = {}
        for name, key, kind in SPECS:
            est, imp = fit_model(tr, key, kind, cfg)
            p = predict(est, imp, te, key)
            metrics[name][f"{src}_to_{dst}"] = full_report(te.y, p, te.pid.to_numpy(), cfg.n_boot, cfg.seed)
            trained[f"{src}_to_{dst}"][name] = (est, imp, key)
    return metrics, trained
