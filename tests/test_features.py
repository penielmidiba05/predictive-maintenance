import pandas as pd
import pytest

from src.features.build import build_features, feature_columns


@pytest.fixture
def toy_df():
    """Un seul capteur 's_a'. Moteur 1 : 4 cycles. Moteur 2 : 2 cycles."""
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2],
            "s_a": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0],
        }
    )


def test_rolling_mean_values(toy_df):
    out = build_features(toy_df, sensors=["s_a"], windows=(2,))
    assert out["s_a_mean_2"].tolist() == [1.0, 1.5, 2.5, 3.5, 10.0, 15.0]


def test_diff_restarts_for_each_unit(toy_df):
    out = build_features(toy_df, sensors=["s_a"], windows=(2,))
    # le premier cycle du moteur 2 ne doit pas être comparé au moteur 1
    assert out["s_a_diff_1"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 10.0]


def test_no_missing_values(toy_df):
    out = build_features(toy_df, sensors=["s_a"], windows=(2, 3))
    assert not out.isna().any().any()


def test_no_future_leakage(toy_df):
    """Changer le DERNIER cycle ne doit rien changer aux cycles précédents."""
    modified = toy_df.copy()
    modified.loc[3, "s_a"] = 999.0  # moteur 1, cycle 4
    cols = feature_columns(sensors=["s_a"], windows=(2, 3))
    before = build_features(toy_df, sensors=["s_a"], windows=(2, 3)).loc[[0, 1, 2], cols]
    after = build_features(modified, sensors=["s_a"], windows=(2, 3)).loc[[0, 1, 2], cols]
    pd.testing.assert_frame_equal(before, after)


def test_default_feature_count():
    assert len(feature_columns()) == 112
