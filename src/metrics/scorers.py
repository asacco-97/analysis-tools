import numpy as np
from typing import Callable, Union
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import get_scorer as sklearn_get_scorer

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


def normalized_gini(y_true, y_pred):
    """Normalized Gini coefficient"""
    return gini(y_true, y_pred) / gini(y_true, y_true)


gini_scorer = make_scorer(normalized_gini, needs_proba=True, greater_is_better=True)


# --- Example template for custom scorer - can be passed as an evaluation function to src.training.GBMModelTrainer
def gini_mape_custom_metric(y_true, y_pred):
    """Balance rank-ordering with prediction accuracy where both metrics are on the same scale (0, 1)."""
    return (0.7 * normalized_gini(y_true, y_pred)) + (0.3 * (1 - mape(y_true, y_pred)))


gini_mape_scorer = make_scorer(gini_mape_custom_metric, needs_proba=False, greater_is_better=True)


# Mapping for custom scorers
CUSTOM_SCORERS = {
    "normalized_gini": gini_scorer,
    "gini_mape_0.7_0.3_weighted_avg": gini_mape_scorer,
}


def get_scorer(name: str) -> Union[str, Callable]:
    """
    Fetch a scorer by name. Falls back to sklearn's built-in scorers if not custom.
    
    Args:
        name (str): Scorer name (e.g., "roc_auc", "normalized_gini")
        
    Returns:
        A scorer function or string
    """
    if name in CUSTOM_SCORERS:
        return CUSTOM_SCORERS[name]
    try:
        return sklearn_get_scorer(name)
    except ValueError:
        raise ValueError(f"Unknown scoring function '{name}'. "
                         f"Must be one of: {list(CUSTOM_SCORERS.keys())} or a valid sklearn scorer.")
