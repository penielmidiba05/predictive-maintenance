from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, params: dict, seed: int = 42):
    """Construit un modèle (non entraîné) à partir de son nom et de ses paramètres."""
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, **params),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=seed, **params
        )
    if name == "lightgbm":
        return LGBMClassifier(
            class_weight="balanced", random_state=seed, verbose=-1, **params
        )
    raise ValueError(f"Modèle inconnu : {name!r}")