import argparse
import logging

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay

from src.config import load_config
from src.data.dataset import load_dev_test
from src.evaluation.cv import cross_validated_proba
from src.evaluation.metrics import classification_metrics
from src.logging_config import setup_logging
from src.models.factory import make_model
from src.models.threshold import best_threshold

logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    setup_logging()  # ← logging
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    dev, _, feature_cols = load_dev_test(cfg)  # le test (_) n'est PAS utilisé ici
    X, y, groups = dev[feature_cols], dev["target"], dev["unit_id"]
    logger.info(  # ← logging
        "%d lignes, %d features, %d moteurs", len(X), X.shape[1], groups.nunique()
    )

    summary = []
    for name, params in cfg["models"].items():
        logger.info("Entraînement : %s", name)  # ← logging
        with mlflow.start_run(run_name=name):
            model = make_model(name, params, cfg["seed"])
            oof, fold_scores = cross_validated_proba(model, X, y, groups, cfg["cv_folds"])
            threshold, cost = best_threshold(y, oof, cfg["cost_fn"], cfg["cost_fp"])
            metrics = classification_metrics(y, oof, threshold)

            mlflow.log_params(
                {
                    "model": name,
                    **params,
                    "horizon": cfg["horizon"],
                    "windows": str(cfg["windows"]),
                    "n_features": len(feature_cols),
                    "cost_fn": cfg["cost_fn"],
                    "cost_fp": cfg["cost_fp"],
                }
            )
            mlflow.log_metrics(
                {
                    **{f"cv_{k}": v for k, v in metrics.items()},
                    "cv_pr_auc_fold_mean": float(fold_scores.mean()),
                    "cv_pr_auc_fold_std": float(fold_scores.std()),
                    "threshold": threshold,
                    "cost": cost,
                }
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay.from_predictions(y, oof, ax=ax, name=name)
            ax.set_title(f"Courbe précision-rappel — {name} (validation croisée)")
            mlflow.log_figure(fig, "pr_curve.png")
            plt.close(fig)

        summary.append(
            {
                "modèle": name,
                "PR-AUC": fold_scores.mean(),
                "± écart-type": fold_scores.std(),
                "seuil": threshold,
                "rappel": metrics["recall"],
                "précision": metrics["precision"],
                "coût": cost,
            }
        )

    table = pd.DataFrame(summary).round(3).to_string(index=False)
    logger.info("Résultats :\n%s", table)  # ← logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare les modèles en validation croisée groupée."
    )
    parser.add_argument("--config", default="configs/train.yaml")
    main(parser.parse_args().config)
