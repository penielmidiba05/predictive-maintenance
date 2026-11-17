from pathlib import Path

import numpy as np
import pytest

from src.data.load import COLUMNS, FEATURE_SENSORS, load_cmapss

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train_FD001.txt"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="données absentes (ex. sur la CI)")


@pytest.fixture(scope="module")
def train():
    return load_cmapss("train")


def test_columns_and_no_missing_values(train):
    assert list(train.columns) == COLUMNS
    assert not train.isna().any().any()


def test_unit_ids(train):
    assert sorted(train["unit_id"].unique()) == list(range(1, 101))


def test_cycles_are_consecutive_for_each_unit(train):
    consecutive = train.groupby("unit_id")["cycle"].apply(
        lambda c: np.array_equal(c.to_numpy(), np.arange(1, len(c) + 1))
    )
    assert consecutive.all()


def test_feature_sensors_vary(train):
    assert (train[FEATURE_SENSORS].std() > 0).all()
