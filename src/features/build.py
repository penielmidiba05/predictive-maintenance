import pandas as pd

from src.data.load import FEATURE_SENSORS

DEFAULT_WINDOWS = (5, 10, 20)


def build_features(
    df: pd.DataFrame,
    sensors: list[str] = FEATURE_SENSORS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Ajoute des features glissantes calculées par moteur, uniquement sur le passé.

    Pour chaque capteur :
    - moyenne et écart-type sur les `w` derniers cycles (pour chaque fenêtre w)
    - différence avec le cycle précédent
    Fonction partagée entre l'entraînement et l'API.
    """
    out = df.sort_values(["unit_id", "cycle"]).copy()
    grouped = out.groupby("unit_id")[sensors]

    new_cols = {}
    for w in windows:
        rolling = grouped.rolling(window=w, min_periods=1)
        mean = rolling.mean().reset_index(level=0, drop=True)
        std = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
        for s in sensors:
            new_cols[f"{s}_mean_{w}"] = mean[s]
            new_cols[f"{s}_std_{w}"] = std[s]

    diff = grouped.diff().fillna(0.0)
    for s in sensors:
        new_cols[f"{s}_diff_1"] = diff[s]

    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def feature_columns(
    sensors: list[str] = FEATURE_SENSORS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> list[str]:
    """Liste ordonnée des colonnes données au modèle."""
    cols = list(sensors)
    for w in windows:
        for s in sensors:
            cols += [f"{s}_mean_{w}", f"{s}_std_{w}"]
    cols += [f"{s}_diff_1" for s in sensors]
    return cols