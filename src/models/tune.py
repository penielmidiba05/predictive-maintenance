import argparse
import logging  # ← logging

import mlflow
import optuna
import yaml

from src.config import load_config
from src.data.dataset import load_dev_test
from src.evaluation.cv import cross_validated_proba
from src.logging_config import setup_logging
from src.models.factory import make_model

logger = logging.getLogger(__name__)


def main(config_path: str, n_trials: int, output_path: str) -> None:
    setup_logging()
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    dev, _, feature_cols = load_dev_test(cfg)
    X, y, groups = dev[feature_cols], dev["target"], dev["unit_id"]
    logger.info("Tuning de LightGBM : %d essais", n_trials)  # ← logging

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            model = make_model("lightgbm", params, cfg["seed"])
            _, fold_scores = cross_validated_proba(model, X, y, groups, cfg["cv_folds"])
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "cv_pr_auc_fold_mean": float(fold_scores.mean()),
                    "cv_pr_auc_fold_std": float(fold_scores.std()),
                }
            )
        return float(fold_scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mlflow.start_run(run_name="optuna_lightgbm"):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = {**study.best_params, "subsample_freq": 1}
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_pr_auc", study.best_value)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, sort_keys=False)

    logger.info("Meilleure PR-AUC (validation croisée) : %.3f", study.best_value)  # ← logging
    logger.info("Paramètres sauvegardés dans %s", output_path)  # ← logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimise LightGBM avec Optuna.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--output", default="configs/lightgbm_best.yaml")
    args = parser.parse_args()
    main(args.config, args.n_trials, args.output)
