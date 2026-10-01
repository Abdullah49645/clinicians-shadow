import pytest
from src.config import Config
from src.data_io import load_hospital
from src.features import build_feature_table
from tests.synth import make_dataset


@pytest.fixture(scope="session")
def root(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("syn"), n_per_hospital=40, seed=1)


@pytest.fixture(scope="session")
def raw(root):
    return load_hospital(root, "A")


@pytest.fixture(scope="session")
def table(raw):
    return build_feature_table(raw, Config())
