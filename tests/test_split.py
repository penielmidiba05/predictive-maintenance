import pandas as pd
from src.data.split import split_units

def make_df(n_units=10):
    return pd.DataFrame ({
        "unit_id": [u for u in range (1, n_units +1) for _ in range(3)],
        "cycle": [c for _ in range(n_units) for c in (1,2,3)],

})

def test_sizes():
    train_units, test_units = split_units(make_df(10), n_test = 3, seed=0)
    assert len(test_units) == 3
    assert len(train_units) ==7

def test_no_overlap_and_complete():
    df = make_df(10)
    train_units, test_units = split_units(df, n_test=3, seed=0)
    assert set(train_units).isdisjoint(test_units)
    assert set(train_units) | set(test_units) == set(df["unit_id"])

def test_reproducible_with_the_same_seed():
    df = make_df(10)
    assert split_units(df, n_test=3, seed=42) == split_units(df, n_test=3, seed=42)

def test_returns_sorted_python_ints():
    train_units , test_units = split_units(make_df(10), n_test=3, seed=0)
    assert train_units == sorted (train_units)
    assert all(type(u) is int for u in train_units + test_units)
