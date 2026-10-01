"""End-to-end experiment: python -m src.run --data data --results results"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from .audit import missingness_leak_audit
from .config import Config
from .data_io import load_hospital
from .export_json import build_cases, run_meta, write_all
from .features import build_feature_table
from .stress import run_stress
from .transfer import run_transfer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--results", default="results")
    ap.add_argument("--max-patients", type=int, default=None, help="per-hospital subsample (smoke runs)")
    ap.add_argument("--n-splits", type=int, default=5); ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--no-stress", action="store_true")
    a = ap.parse_args(argv)
    cfg = replace(Config(), seed=a.seed, n_splits=a.n_splits, n_boot=a.n_boot, run_stress=not a.no_stress)
    synthetic = (Path(a.data) / "SYNTHETIC.txt").exists()

    raw = {h: load_hospital(a.data, h, a.max_patients, cfg.seed) for h in "AB"}
    tables = {h: build_feature_table(r, cfg) for h, r in raw.items()}
    print({h: (t.pid.nunique(), len(t.y)) for h, t in tables.items()})
    metrics, trained = run_transfer(tables, cfg)

    audit = {h: missingness_leak_audit(t, cfg) for h, t in tables.items()}
    cases, attr, stress = [], {}, {}
    for d, (s, t) in {"A_to_B": ("A", "B"), "B_to_A": ("B", "A")}.items():
        tm = trained[d]
        c, summ = build_cases(raw[t], tables[t], tables[s], tm["M_VP"][0], tm["M_VP"][1], cfg, d,
                              tm["M_V"][:2], tm["M_P"][:2])
        cases += c; attr[d] = summ
        if cfg.run_stress:
            stress[d] = run_stress(raw[t], {k: v for k, v in tm.items()}, cfg)
    write_all(Path(a.results), metrics, cases, attr, stress, audit, run_meta(cfg, tables, synthetic))
    print("wrote", a.results)


if __name__ == "__main__":
    main()
