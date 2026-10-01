"""Missingness-leak audit: can physiology features alone predict that a lab is being measured now?
AUROC near 0.5 = representation carries little measurement-timing information.
Reported as a diagnostic, not a guarantee."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from .config import Config
from .features import FeatureTable, fit_imputer
from .models import make_model
from .splits import patient_folds
import pandas as pd


def missingness_leak_audit(t: FeatureTable, cfg: Config, var: str = "Lactate") -> dict:
    target = t.process[f"p__obs__{var}"].to_numpy() > 0
    if target.min() == target.max():
        return {"variable": var, "auroc": None}
    X0 = pd.concat([t.base, t.physiology], axis=1)
    tr, te = next(patient_folds(t.pid, t.y, 2, cfg.seed))
    imp = fit_imputer(t.take(tr), cfg.seed)
    cols = list(t.physiology.columns)
    X = imp.transform(X0, t.pid)[cols]
    est = make_model("hgb", cfg).fit(X.iloc[tr], target[tr])
    return {"variable": var, "auroc": float(roc_auc_score(target[te], est.predict_proba(X.iloc[te])[:, 1])),
            "note": "AUROC of predicting 'variable measured at this hour' from physiology features only"}
