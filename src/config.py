"""Central configuration: column groups, constants, experiment settings."""
from __future__ import annotations
from dataclasses import dataclass

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
VALUE_COLS = VITALS + LABS                     # 34 time-varying measurements
DEMOG_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
BASE_COLS = ["Age", "Gender", "HospAdmTime", "ICULOS"]
LABEL = "SepsisLabel"
ALL_COLS = VALUE_COLS + DEMOG_COLS + [LABEL]   # 41 columns in each .psv
HOSPITAL_DIRS = {"A": "training_setA", "B": "training_setB"}

# Feature-family prefixes: the *only* way a column's information source is defined.
PREFIX = {"base": "b__", "physiology": "v__", "process": "p__"}

# Model name -> feature families it may see.
MODEL_FAMILIES = {
    "M_BASE": ("base",),
    "M_V": ("base", "physiology"),
    "M_P": ("base", "process"),
    "M_VP": ("base", "physiology", "process"),
}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_splits: int = 5
    n_boot: int = 100
    hours_cap: int = 72          # cap for "hours since last measurement"
    window: int = 6              # recent-observation window (rows/hours)
    thin_levels: tuple[float, ...] = (0.0, 0.25, 0.5)
    n_case_septic: int = 6
    n_case_control: int = 3
    hgb_max_iter: int = 200
    hgb_lr: float = 0.1
    hgb_leaf_nodes: int = 31
    hgb_l2: float = 1.0
    run_stress: bool = True
