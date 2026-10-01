"""Optional measurement-thinning stress test on the target hospital."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import VALUE_COLS, Config
from .evaluate import full_report
from .features import build_feature_table


def thin_observations(raw: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Independently drop each observed measurement with probability `frac` (labels untouched)."""
    out = raw.copy()
    if frac <= 0:
        return out
    rng = np.random.default_rng(seed)
    vals = out[VALUE_COLS].to_numpy(copy=True)
    drop = (rng.random(vals.shape) < frac) & ~np.isnan(vals)
    vals[drop] = np.nan
    out[VALUE_COLS] = vals
    return out


def run_stress(raw_target: pd.DataFrame, trained_dir: dict, cfg: Config) -> dict:
    from .transfer import predict
    res: dict = {}
    for frac in cfg.thin_levels:
        t = build_feature_table(thin_observations(raw_target, frac, cfg.seed), cfg)
        res[str(frac)] = {name: full_report(t.y, predict(est, imp, t, key), t.pid.to_numpy(), 0, cfg.seed)
                          for name, (est, imp, key) in trained_dir.items()}
    return res
