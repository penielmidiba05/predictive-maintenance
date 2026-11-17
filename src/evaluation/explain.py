import argparse
import logging  # ← logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import load_config
from src.data.dataset import load_dev_test
from src.logging_config import setup_logging
from src.models.predict import load_bundle

logger = logging.getLogger(__name__)

FIG_DIR = Path("reports/figures")


def sensor_of(feature: str) -> str:
    """'s_11_mean_10' -> 's_11'."""
    return "_".join(feature.split("_")[:2])


def main(config_path: str) -> None:
    setup_logging()
    cfg = load_config(config_path)
    bundle = load_bundle()
    _, test, _ = load_dev_test(cfg)
    X = test[bundle["feature_columns"]]
    logger.info("Calcul des valeurs SHAP sur %d lignes", len(X))  # ← logging

    explainer = shap.TreeExplainer(bundle["model"])
    sv = explainer(X)
    if sv.values.ndim == 3:  # certaines versions renvoient une explication par classe
        sv = sv[:, :, 1]

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Vue globale : les features les plus influentes
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP — importance globale (20 moteurs de test)")
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Importance regroupée par capteur
    importance = pd.DataFrame(abs(sv.values), columns=X.columns).mean()
    by_sensor = importance.groupby(sensor_of).sum().sort_values()
    ax = by_sensor.plot.barh(figsize=(6, 6))
    ax.set_xlabel("Somme des |valeurs SHAP| moyennes")
    ax.set_title("Importance par capteur (toutes features confondues)")
    plt.savefig(FIG_DIR / "shap_by_sensor.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Vue locale : un moteur 10 cycles avant sa panne
    row = test.index[test["RUL"] == 10][0]
    unit = test.loc[row, "unit_id"]
    shap.plots.waterfall(sv[row], max_display=12, show=False)
    plt.title(f"Moteur {unit}, 10 cycles avant la panne")
    plt.savefig(FIG_DIR / "shap_waterfall_rul10.png", dpi=150, bbox_inches="tight")
    plt.close()

    top = by_sensor.sort_values(ascending=False).head(5).round(3).to_string()
    logger.info("Capteurs les plus influents :\n%s", top)  # ← logging
    logger.info("Figures enregistrées dans %s/", FIG_DIR)  # ← logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explique le modèle avec SHAP.")
    parser.add_argument("--config", default="configs/train.yaml")
    main(parser.parse_args().config)
