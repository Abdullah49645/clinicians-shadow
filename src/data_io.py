"""Dataset ingestion for the PhysioNet/CinC Challenge 2019 .psv files."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ALL_COLS, HOSPITAL_DIRS

SETUP_HELP = (
    "Dataset not found at {path}.\n"
    "Download the PhysioNet/CinC Challenge 2019 training data (see README, 'Dataset') and place:\n"
    "  data/training_setA/*.psv   (Hospital A)\n  data/training_setB/*.psv   (Hospital B)\n"
)


def load_hospital(data_dir: str | Path, hospital: str, max_patients: int | None = None,
                  seed: int = 0) -> pd.DataFrame:
    """Load one hospital group into a single frame sorted by (pid, ICULOS).

    Adds: patient_id (str), pid (int), hospital. Raises with instructions if missing.
    """
    d = Path(data_dir) / HOSPITAL_DIRS[hospital]
    files = sorted(d.glob("*.psv")) if d.is_dir() else []
    if not files:
        raise FileNotFoundError(SETUP_HELP.format(path=d))
    if max_patients and max_patients < len(files):
        rng = np.random.default_rng(seed)
        files = sorted(rng.choice(files, size=max_patients, replace=False).tolist())
    frames = []
    for f in files:
        df = pd.read_csv(f, sep="|")
        if list(df.columns) != ALL_COLS:
            raise ValueError(f"{f.name}: unexpected columns {list(df.columns)}")
        digits = re.sub(r"\D", "", f.stem)
        df["patient_id"] = f.stem
        df["pid"] = int(digits) if digits else hash(f.stem) % (10**9)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["hospital"] = hospital
    for c in ALL_COLS:
        out[c] = out[c].astype("float32")
    out = out.sort_values(["pid", "ICULOS"], kind="stable").reset_index(drop=True)
    validate_raw(out)
    return out


def validate_raw(df: pd.DataFrame) -> None:
    if df["pid"].duplicated().sum() == 0 and len(df) == 0:
        raise ValueError("empty dataset")
    if not df["SepsisLabel"].dropna().isin([0, 1]).all():
        raise ValueError("SepsisLabel must be binary")
    dif = df.groupby("pid", sort=False)["ICULOS"].diff().dropna()
    if (dif <= 0).any():
        raise ValueError("ICULOS must be strictly increasing within a patient")
