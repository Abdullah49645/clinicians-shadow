"""Model factory. Identical hyperparameters across all feature sets (comparability)."""
from __future__ import annotations
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import Config


def make_model(kind: str, cfg: Config):
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=cfg.hgb_max_iter, learning_rate=cfg.hgb_lr, max_leaf_nodes=cfg.hgb_leaf_nodes,
            l2_regularization=cfg.hgb_l2, early_stopping=False, random_state=cfg.seed)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))
    raise ValueError(kind)
