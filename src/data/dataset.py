import json

import pandas as pd

from src.data.labels import make_labels
from src.data.load import load_cmapss
from src.features.build import build_features, feature_columns


def load_dev_test(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Charge les données, ajoute labels et features, puis sépare dev (80 moteurs) et test (20)."""
    windows = tuple(cfg["windows"])
    df = make_labels(load_cmapss("train", cfg["subset"]), horizon=cfg["horizon"])
    df = build_features(df, windows=windows)

    with open(cfg["split_path"], encoding="utf-8") as f:
        split = json.load(f)

    dev = df[df["unit_id"].isin(split["train_units"])].reset_index(drop=True)
    test = df[df["unit_id"].isin(split["test_units"])].reset_index(drop=True)
    return dev, test, feature_columns(windows=windows)