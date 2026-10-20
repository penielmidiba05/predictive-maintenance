import json
from pathlib import Path
import numpy as np
import pandas as pd

def split_units(df: pd.DataFrame, n_test: int = 20, seed: int=42) -> tuple[list[int], list[int]] :
    """Découpe les MOTEURS (pas les lignes) en train et test.
    Renvoie deux listes triées d'identifiants (int Python)."""
    units = np.sort(df["unit_id"].unique()) #liste des moteurs triés

    if not 0 < n_test < len(units):
        raise ValueError(f"n_test doit etre entre 1 et {len(units)}, reçu : {n_test}")

    rng = np.random.default_rng(seed)
    test_draw = rng.choice(units, size = n_test, replace=False)

    test_units= sorted(int(u) for u in test_draw)
    test_set = set(test_units)
    train_units = [int(u) for u in units if int(u) not in test_set]

    return train_units, test_units

def save_split(train_units: list[int], test_units: list[int], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"train_units": train_units, "test_units": test_units}, f, indent=2)