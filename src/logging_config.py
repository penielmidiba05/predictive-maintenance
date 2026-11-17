import logging


def setup_logging(level: str = "INFO") -> None:

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Bibliothèques trop bavardes : on ne garde que leurs avertissements
    for noisy in ["alembic", "mlflow", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
