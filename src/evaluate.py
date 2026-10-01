"""Metrics and patient-level bootstrap confidence intervals."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def calibration_bins(y, p, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        s = idx == b
        if s.any():
            out.append({"bin": b, "n": int(s.sum()), "mean_pred": float(p[s].mean()), "frac_pos": float(y[s].mean())})
    return out


def ece(y, p, n_bins: int = 10) -> float:
    bins = calibration_bins(y, p, n_bins)
    n = len(y)
    return float(sum(b["n"] / n * abs(b["mean_pred"] - b["frac_pos"]) for b in bins))


def point_metrics(y, p) -> dict:
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    if y.min() == y.max():
        return {"auroc": None, "auprc": None, "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}


def bootstrap_ci(y, p, pid, n_boot: int, seed: int) -> dict:
    """Percentile 95% CI resampling PATIENTS (not rows) with replacement."""
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64); pid = np.asarray(pid)
    order = np.argsort(pid, kind="stable")
    y, p, pid = y[order], p[order], pid[order]
    uniq, start, cnt = np.unique(pid, return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("auroc", "auprc", "brier")}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([np.arange(start[i], start[i] + cnt[i]) for i in pick])
        yb, pb = y[rows], p[rows]
        acc["brier"].append(brier_score_loss(yb, pb))
        if yb.min() != yb.max():
            acc["auroc"].append(roc_auc_score(yb, pb)); acc["auprc"].append(average_precision_score(yb, pb))
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 1 else None)
            for k, v in acc.items()}


def full_report(y, p, pid, n_boot: int, seed: int) -> dict:
    r = point_metrics(y, p)
    r["ci95"] = bootstrap_ci(y, p, pid, n_boot, seed) if n_boot > 0 else None
    r["n_rows"] = int(len(y)); r["n_patients"] = int(len(np.unique(pid)))
    r["row_prevalence"] = float(np.mean(y))
    r["calibration"] = calibration_bins(y, np.asarray(p))
    return r
