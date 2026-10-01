"""Test D: labels stay aligned with the right patient and hour."""
import numpy as np
import pandas as pd
from src.config import ALL_COLS


def test_D_labels_match_source_files(root, raw, table):
    for f in sorted((root / "training_setA").glob("*.psv"))[:10]:
        src = pd.read_csv(f, sep="|")
        mask = (raw["patient_id"] == f.stem).to_numpy()
        assert (table.y[mask] == src["SepsisLabel"].to_numpy()).all()
        assert (table.iculos[mask].to_numpy() == src["ICULOS"].to_numpy()).all()


def test_D_rows_sorted_and_label_binary(raw, table):
    assert set(np.unique(table.y)) <= {0, 1}
    assert (raw.groupby("pid")["ICULOS"].diff().dropna() > 0).all()
    assert len(table.y) == len(table.base) == len(table.physiology) == len(table.process)


def test_D_take_keeps_alignment(table):
    rows = np.array([5, 3, 17])
    t = table.take(rows)
    assert (t.y == table.y[rows]).all() and (t.pid.to_numpy() == table.pid.to_numpy()[rows]).all()
    assert (t.process.to_numpy() == table.process.to_numpy()[rows]).all()
