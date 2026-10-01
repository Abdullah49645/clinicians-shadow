"""Write machine-readable results and build patient cases for the viewer."""
from __future__ import annotations
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from .attribution import contributions
from .config import MODEL_FAMILIES, VALUE_COLS, Config
from .features import FeatureTable, assemble, source_of

DISPLAY_VARS = ["HR", "MAP", "Resp", "Temp", "Lactate", "WBC", "Creatinine"]
SCENARIOS = ["within_A", "within_B", "A_to_B", "B_to_A"]


def _clean(o):
    if isinstance(o, np.ndarray):
        o = o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_clean(obj), indent=1))


def transfer_matrix(metrics: dict) -> dict:
    keep = ("auroc", "auprc", "brier", "ece")
    return {m: {s: {**{k: r[k] for k in keep}, "ci95": r["ci95"]} for s, r in sc.items()} for m, sc in metrics.items()}


def results_table_md(metrics: dict) -> str:
    out = []
    for metric in ("auroc", "auprc", "brier"):
        out += [f"**{metric.upper()}** (95% patient-bootstrap CI)", "",
                "| model | " + " | ".join(SCENARIOS) + " |", "|---|" + "---|" * len(SCENARIOS)]
        for m, sc in metrics.items():
            cells = []
            for s in SCENARIOS:
                r = sc[s]; ci = (r["ci95"] or {}).get(metric)
                cells.append("n/a" if r[metric] is None else f"{r[metric]:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def build_cases(raw: pd.DataFrame, t: FeatureTable, train: FeatureTable, est, imp, cfg: Config,
                direction: str, m_v, m_p) -> tuple[list[dict], dict]:
    """Seeded random sample of septic / non-septic target patients (no hand-picking)."""
    X = assemble(t, "M_VP", imp)
    ref = assemble(train, "M_VP", imp).median()
    c = contributions(est, X, ref)
    pv = m_v[0].predict_proba(assemble(t, "M_V", m_v[1]))[:, 1]
    pp = m_p[0].predict_proba(assemble(t, "M_P", m_p[1]))[:, 1]
    pid = t.pid.to_numpy()
    ever = pd.Series(t.y).groupby(pid).max()
    rng = np.random.default_rng(cfg.seed)
    sep = rng.permutation(ever[ever == 1].index.to_numpy())[: cfg.n_case_septic]
    ctl = rng.permutation(ever[ever == 0].index.to_numpy())[: cfg.n_case_control]
    cases = []
    for p in list(sep) + list(ctl):
        r = np.where(pid == p)[0]
        sub = raw.iloc[r]
        cases.append({
            "id": f"{direction}:{int(p)}", "pid": int(p), "ever_septic": bool(ever[p]),
            "hours": sub["ICULOS"].astype(int).tolist(), "label": t.y[r].astype(int).tolist(),
            "values": {v: sub[v].where(sub[v].notna(), None).tolist() for v in DISPLAY_VARS},
            "observed_hours": {v: sub["ICULOS"][sub[v].notna()].astype(int).tolist() for v in VALUE_COLS},
            "pred": {"all": c["all"][r], "physiology_only": c["physiology_only"][r], "process_only": c["process_only"][r],
                     "M_V": pv[r], "M_P": pp[r]},
            "pri": c["pri"][r]})
    rows = rng.choice(len(pid), size=min(100_000, len(pid)), replace=False)
    pri = c["pri"][rows]
    summary = {"rows_sampled": int(len(rows)), "pri_defined_fraction": float(np.mean(~np.isnan(pri))),
               "pri_median": float(np.nanmedian(pri)) if np.any(~np.isnan(pri)) else None,
               "mean_c_process": float(np.mean(c["c_process"][rows])), "mean_c_physiology": float(np.mean(c["c_physiology"][rows])),
               "caveat": "model-based associational diagnostic; not causal"}
    return cases, summary


def write_all(out: Path, metrics, cases, attr_summary, stress, audit, meta) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "transfer_metrics.json", metrics)
    write_json(out / "transfer_matrix.json", transfer_matrix(metrics))
    write_json(out / "model_metrics.json", meta)
    write_json(out / "attribution_cases.json", {"summary": attr_summary, "cases": cases,
               "note": "Model-input counterfactuals: families replaced by a fixed reference; not clinical counterfactuals."})
    write_json(out / "stress_test.json", stress or {})
    write_json(out / "audit.json", audit)
    (out / "results_table.md").write_text(results_table_md(metrics))


def run_meta(cfg: Config, tables, synthetic: bool) -> dict:
    from dataclasses import asdict
    fam = tables["A"]
    return {"synthetic": synthetic, "config": asdict(cfg),
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            "cohorts": {h: {"patients": int(t.pid.nunique()), "rows": int(len(t.y)), "row_prevalence": float(t.y.mean()),
                            "septic_patients": int(pd.Series(t.y).groupby(t.pid.to_numpy()).max().sum())} for h, t in tables.items()},
            "feature_counts": {"base": fam.base.shape[1], "physiology": fam.physiology.shape[1], "process": fam.process.shape[1]},
            "model_families": {k: list(v) for k, v in MODEL_FAMILIES.items()}}
