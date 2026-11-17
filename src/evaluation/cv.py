import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold


def cross_validated_proba(
    model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5
):
    """Validation croisée groupée par moteur.

    Renvoie :
    - oof : probabilité prédite pour chaque ligne par un modèle qui ne l'a jamais vue
    - fold_scores : PR-AUC de chaque pli
    """
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_scores.append(average_precision_score(y.iloc[val_idx], oof[val_idx]))
    return oof, np.array(fold_scores)
