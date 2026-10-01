"""SYNTHETIC fixture generator -- for exercising code paths in tests ONLY.
Output is NOT real data and must never be reported as results."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from src.config import ALL_COLS, LABS, VALUE_COLS, VITALS


def make_dataset(root: Path, n_per_hospital: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / "SYNTHETIC.txt").write_text("synthetic test fixture; not real data\n")
    for h, (d, base_rate, offset) in {"A": ("training_setA", 0.5, 0), "B": ("training_setB", 0.25, 100000)}.items():
        (root / d).mkdir(exist_ok=True)
        for i in range(n_per_hospital):
            T = int(rng.integers(12, 48)); septic = rng.random() < 0.3
            onset = int(rng.integers(6, T)) if septic else None
            rows = np.full((T, len(ALL_COLS)), np.nan)
            sev = np.where(np.arange(T) >= (onset or 10**6) - 6, 1.0, 0.0)
            for j, c in enumerate(VALUE_COLS):
                base = 80 if c in VITALS else 5
                vals = base + 5 * rng.standard_normal(T) + 8 * sev * (c in ("HR", "Lactate", "Resp"))
                rate = base_rate * (1.6 if septic else 1.0) * (0.5 if c in LABS else 1.0)
                rows[:, j] = np.where(rng.random(T) < rate, vals, np.nan)
            ix = {c: k for k, c in enumerate(ALL_COLS)}
            rows[:, ix["Age"]] = rng.integers(30, 90); rows[:, ix["Gender"]] = rng.integers(0, 2)
            rows[:, ix["HospAdmTime"]] = -rng.random() * 50; rows[:, ix["ICULOS"]] = np.arange(1, T + 1)
            rows[:, ix["SepsisLabel"]] = sev
            with open(root / d / f"p{offset + i + 1:06d}.psv", "w") as f:
                f.write("|".join(ALL_COLS) + "\n")
                for r in rows:
                    f.write("|".join("NaN" if np.isnan(x) else f"{x:.3f}" for x in r) + "\n")
    return root
