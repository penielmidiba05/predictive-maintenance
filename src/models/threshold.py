import numpy as np


def total_cost(y_true, proba, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Coût total des erreurs pour un seuil donné"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(proba) >= threshold
    fn = np.sum((y_true == 1) & ~y_pred)   # pannes ratées
    fp = np.sum((y_true == 0) & y_pred)    # fausses alertes
    return float(cost_fn * fn + cost_fp * fp)


def best_threshold(y_true, proba, cost_fn: float = 10.0, cost_fp: float = 1.0) -> tuple[float, float]:
    """Teste les seuils de 0,01 à 0,99 et renvoie celui qui minimise le coût."""
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [total_cost(y_true, proba, t, cost_fn, cost_fp) for t in thresholds]
    best = int(np.argmin(costs))
    return float(thresholds[best]), costs[best]