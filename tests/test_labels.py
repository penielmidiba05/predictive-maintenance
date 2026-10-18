from pathlib import Path

import pandas as pd
import pytest

from src.data.labels import make_labels


@pytest.fixture
def toy_df():
    """Moteur 1 : 5 cycles. Moteur 2 : 3 cycles."""
    return pd.DataFrame({
        "unit_id": [1, 1, 1, 1, 1, 2, 2, 2],
        "cycle":   [1, 2, 3, 4, 5, 1, 2, 3],
    })


def test_rul_values(toy_df):
    out = make_labels(toy_df, horizon=2)
    assert out["RUL"].tolist() == [4, 3, 2, 1, 0, 2, 1, 0]


def test_rul_is_zero_at_last_cycle(toy_df):
    out = make_labels(toy_df, horizon=2)
    last_rows = out.groupby("unit_id").tail(1)
    assert (last_rows["RUL"] == 0).all()


def test_target_uses_horizon(toy_df):
    out = make_labels(toy_df, horizon=2)
    assert out["target"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]


def test_input_is_not_modified(toy_df):
    before = toy_df.copy()
    make_labels(toy_df, horizon=2)
    pd.testing.assert_frame_equal(toy_df, before)


def test_negative_horizon_raises(toy_df):
    with pytest.raises(ValueError):
        make_labels(toy_df, horizon=-1)


DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "train_FD001.txt"


@pytest.mark.skipif(not DATA.exists(), reason="données absentes (ex. sur la CI)")
def test_positive_count_on_real_data():
    from src.data.load import load_cmapss
    out = make_labels(load_cmapss("train"), horizon=30)
    # chaque moteur a exactement 31 cycles avec RUL entre 0 et 30
    assert out["target"].sum() == 100 * 31
    