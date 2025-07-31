import numpy as np
from typing import Callable, Union
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import get_scorer as sklearn_get_scorer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)

# --- Normalized Gini
def gini(actual, pred):
    """Unnormalized Gini coefficient"""
    assert len(actual) == len(pred)
    all_data = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float64)
    all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
    total_losses = all_data[:, 0].sum()
    gini_sum = all_data[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.0
    return gini_sum / len(actual)

def neg_normalized_gini(y_true, y_pred):
    """Negative Normalized Gini coefficient for use with Bayesian search"""
    return -(gini(y_true, y_pred) / gini(y_true, y_true))

# --- Example template for custom scorer - can be passed as an evaluation function to src.training.GBMModelTrainer
def gini_mape_custom_metric(y_true, y_pred):
    """Balance rank-ordering with prediction accuracy where both metrics are on the same scale (0, 1)."""
    return (0.7 * neg_normalized_gini(y_true, y_pred)) - (0.3 * (1 - mape(y_true, y_pred)))

# Mapping for custom scorers
METRICS = {
    "neg_normalized_gini": neg_normalized_gini,
    "gini_mape_0.7_0.3_weighted_avg": gini_mape_custom_metric,
    "logloss": log_loss,
    "roc_auc": roc_auc_score,
    "f1": f1_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
}

def get_scorer(name: str) -> Callable:
    """
    Fetch a scorer by name. Falls back to sklearn's built-in scorers if not custom.

    Args:
        name (str): Scorer name (e.g., "roc_auc", "normalized_gini")

    Returns:
        Callable: A scoring function y_true, y_pred -> float
    """
    if name in METRICS:
        return METRICS[name]

    try:
        # Sklearn scorers need to be converted to callables
        scorer = sklearn_get_scorer(name)
        return lambda y_true, y_pred: scorer._score_func(y_true, y_pred)
    except ValueError:
        raise ValueError(
            f"Unknown scoring function '{name}'. "
            f"Must be one of: {list(METRICS.keys())} or a valid sklearn scorer."
        )
