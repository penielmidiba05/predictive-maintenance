import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow

from src.config import load_config
from src.data.dataset import load_dev_test
from src.evaluation.cv import cross_validated_proba
from src.evaluation.metrics import classification_metrics
from src.logging_config import setup_logging
from src.models.factory import make_model
from src.models.threshold import best_threshold

logger = logging.getLogger(__name__)


def main(config_path: str, model_name: str, params_path: str | None) -> None:
    setup_logging()
    cfg = load_config(config_path)
    params = load_config(params_path) if params_path else cfg["models"][model_name]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    dev, test, feature_cols = load_dev_test(cfg)
    X_dev, y_dev, g_dev = dev[feature_cols], dev["target"], dev["unit_id"]
    model = make_model(model_name, params, cfg["seed"])
    logger.info("Évaluation finale du modèle : %s", model_name)  # ← logging

    # 1. Seuil choisi sur les 80 moteurs de développement (prédictions hors pli)
    oof, fold_scores = cross_validated_proba(model, X_dev, y_dev, g_dev, cfg["cv_folds"])
    threshold, _ = best_threshold(y_dev, oof, cfg["cost_fn"], cfg["cost_fp"])
    cv_metrics = classification_metrics(y_dev, oof, threshold)

    # 2. Entraînement final sur les 80 moteurs
    model.fit(X_dev, y_dev)

    # 3. Évaluation UNIQUE sur les 20 moteurs de test
    test_proba = model.predict_proba(test[feature_cols])[:, 1]
    test_metrics = classification_metrics(test["target"], test_proba, threshold)

    # 4. Sauvegarde du modèle et de tout ce qu'il faut pour l'utiliser
    Path("models").mkdir(exist_ok=True)
    bundle = {
        "model": model,
        "model_name": model_name,
        "params": params,
        "threshold": threshold,
        "feature_columns": feature_cols,
        "windows": cfg["windows"],
        "horizon": cfg["horizon"],
    }
    joblib.dump(bundle, "models/model.joblib")

    results = {
        "model": model_name,
        "threshold": threshold,
        "cv": cv_metrics,
        "cv_pr_auc_fold_std": float(fold_scores.std()),
        "test": test_metrics,
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with mlflow.start_run(run_name=f"final_{model_name}"):
        mlflow.log_params({"model": model_name, **params, "threshold": threshold})
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact("models/model.joblib")
        mlflow.log_artifact("reports/test_results.json")

    logger.info(  # ← logging
        "Résultats :\n%s", json.dumps(results, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation finale sur les 20 moteurs de test.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument(
        "--params", default=None, help="YAML de paramètres (ex. configs/lightgbm_best.yaml)"
    )
    args = parser.parse_args()
    main(args.config, args.model, args.params)
