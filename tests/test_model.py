import json
from pathlib import Path

import pytest
from sklearn.metrics import average_precision_score

from src.data.labels import make_labels
from src.data.load import load_cmapss
from src.models.predict import MODEL_PATH, load_bundle, predict_cycles

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "train_FD001.txt"
RESULTS = ROOT / "reports" / "test_results.json"
pytestmark = pytest.mark.skipif(
    not (DATA.exists() and MODEL_PATH.exists()),
    reason="données ou modèle absents (ex. sur la CI)",
)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def test_engines():
    with open(ROOT / "configs" / "split_FD001.json", encoding="utf-8") as f:
        test_units = json.load(f)["test_units"]
    df = make_labels(load_cmapss("train"), horizon=30)
    return df[df["unit_id"].isin(test_units)]


def test_bundle_contains_everything(bundle):
    for key in ["model", "threshold", "feature_columns", "windows", "horizon"]:
        assert key in bundle
    assert 0 < bundle["threshold"] < 1


def test_probabilities_are_valid(bundle, test_engines):
    out = predict_cycles(bundle, test_engines)
    assert len(out) == len(test_engines)
    assert out["probability"].between(0, 1).all()
    assert set(out["alert"].unique()) <= {0, 1}


def test_works_with_short_history(bundle, test_engines):
    """Un moteur avec seulement 3 cycles (moins que la plus grande fenêtre)."""
    first_unit = test_engines["unit_id"].iloc[0]
    short = test_engines[test_engines["unit_id"] == first_unit].head(3)
    out = predict_cycles(bundle, short)
    assert len(out) == 3
    assert not out["probability"].isna().any()


def test_reproduces_reported_test_score(bundle, test_engines):
    """Le modèle sauvegardé redonne exactement le score publié dans le README."""
    out = predict_cycles(bundle, test_engines)
    merged = out.merge(test_engines[["unit_id", "cycle", "target"]], on=["unit_id", "cycle"])
    pr_auc = average_precision_score(merged["target"], merged["probability"])
    with open(RESULTS, encoding="utf-8") as f:
        reported = json.load(f)["test"]["pr_auc"]
    assert pr_auc == pytest.approx(reported, abs=1e-6)
