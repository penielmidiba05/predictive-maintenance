from pathlib import Path

import joblib
import pandas as pd

from src.data.load import PROJECT_ROOT
from src.features.build import build_features

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def load_bundle(path: Path = MODEL_PATH) -> dict:
    """Charge le modèle et tout ce qu'il faut pour l'utiliser (seuil, colonnes, fenêtres)."""
    return joblib.load(path)


def predict_cycles(bundle: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prédit, pour chaque cycle, la probabilité de panne dans l'horizon et l'alerte.

    raw_df : cycles bruts (unit_id, cycle, réglages, capteurs), comme dans les fichiers NASA.
    """
    features = build_features(raw_df, windows=tuple(bundle["windows"]))
    proba = bundle["model"].predict_proba(features[bundle["feature_columns"]])[:, 1]

    out = features[["unit_id", "cycle"]].copy()
    out["probability"] = proba
    out["alert"] = (proba >= bundle["threshold"]).astype(int)
    return out
