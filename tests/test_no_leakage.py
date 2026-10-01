"""Test C: a patient never appears in both partitions."""
import numpy as np
import pytest
from src.data_io import load_hospital
from src.splits import assert_disjoint, patient_folds


def test_C_folds_are_patient_disjoint_and_cover_all_rows(table):
    seen = np.zeros(len(table.y), int)
    for tr, te in patient_folds(table.pid, table.y, 4, 0):
        assert not (set(table.pid.iloc[tr]) & set(table.pid.iloc[te]))
        seen[te] += 1
    assert (seen == 1).all()


def test_C_assert_disjoint_catches_row_split(table):
    rows = np.arange(len(table.y))
    with pytest.raises(AssertionError):
        assert_disjoint(table.pid, rows[::2], rows[1::2])      # naive row split must fail


def test_C_hospitals_have_distinct_patient_ids(root, raw):
    assert not (set(raw["pid"]) & set(load_hospital(root, "B")["pid"]))


def test_missing_dataset_error_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="training_setA"):
        load_hospital(tmp_path, "A")
