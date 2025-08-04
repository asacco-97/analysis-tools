import numpy as np
import pandas as pd
from typing import Callable, Union, Optional
from functools import partial
import warnings 
from sklearn.metrics import mean_absolute_percentage_error as mape
import sklearn.metrics as sk_metrics

gini_weight = 0.7  # Default weight for gini + mape custom metric
partial_gini_top_percent = 10  # Default top percent for partial gini custom metric

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

def normalized_gini(y_true, y_pred) -> float:
    """Normalized Gini coefficient for use with Bayesian search"""
    return gini(y_true, y_pred) / gini(y_true, y_true)

# --- Normalized Partial Gini (top X% of predictions)
def partial_gini_index(y_true, y_pred, top_percent: float = 10) -> float:
    """Normalized Partial Gini over the top X% of predicted scores (no weights)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
    })

    df["weighted_actual"] = df["actual"]

    df.sort_values("predicted", ascending=False, inplace=True)
    df["cum_exposure"] = np.linspace(1 / len(df), 1, len(df))
    df["cum_actual"] = df["weighted_actual"].cumsum() / df["weighted_actual"].sum()

    pop = np.insert(df["cum_exposure"].values, 0, 0) # type: ignore
    cum = np.insert(df["cum_actual"].values, 0, 0) # type: ignore

    # Perfect ordering
    df.sort_values("weighted_actual", ascending=False, inplace=True)
    df["cum_exposure"] = np.linspace(1 / len(df), 1, len(df))
    df["cum_actual"] = df["weighted_actual"].cumsum() / df["weighted_actual"].sum()
    perf_pop = np.insert(df["cum_exposure"].values, 0, 0) # type: ignore
    perf_cum = np.insert(df["cum_actual"].values, 0, 0) # type: ignore

    def _partial(pop, cum):
        cutoff = top_percent / 100
        idx = np.searchsorted(pop, cutoff, side="right")
        pop_cut = pop[:idx]
        cum_cut = cum[:idx]
        if len(pop_cut) == 0 or pop_cut[-1] < cutoff:
            pop_cut = np.append(pop_cut, cutoff)
            cum_cut = np.append(cum_cut, np.interp(cutoff, pop, cum))
        auc = np.trapezoid(cum_cut, x=pop_cut)
        baseline = (cutoff ** 2) / 2
        return (auc - baseline) / (1 - baseline)

    return _partial(pop, cum) / _partial(perf_pop, perf_cum) # type: ignore

# --- Example template for custom scorer - can be passed as an evaluation function to src.training.GBMModelTrainer
def gini_mape_custom_metric(y_true, y_pred, gini_weight: float = 0.7) -> float:
    """Balance rank-ordering with prediction accuracy where both metrics are on the same scale (0, 1)."""
    return (gini_weight * normalized_gini(y_true, y_pred)) + ((1 - gini_weight) * (1 - mape(y_true, y_pred)))

# --- Custom metrics added here
normalized_gini.__name__ = "normalized_gini"
normalized_gini.__name__ = "normalized_gini"

gini_mape_70_30_weighted_avg = partial(gini_mape_custom_metric, gini_weight=0.7)
gini_mape_70_30_weighted_avg.__name__ = "gini_mape_70_30_weighted_avg"

normalized_partial_gini_10 = partial(partial_gini_index, top_percent=10)
normalized_partial_gini_10.__name__ = "normalized_partial_gini_10"

normalized_partial_gini_20 = partial(partial_gini_index, top_percent=20)
normalized_partial_gini_20.__name__ = "normalized_partial_gini_20"



def get_scorer(name_or_func: Optional[Union[str, Callable]] = None) -> tuple[Callable, str, bool]:
    """
    Returns a scoring function and metadata: (scoring_func, name, maximize).
    Supports any function from sklearn.metrics or a user-passed callable.
    """
    # Check if the input is a custom callable function
    if callable(name_or_func):
        return name_or_func, getattr(name_or_func, "__name__", "custom_metric"), False

    if isinstance(name_or_func, str):
        # Try sklearn.metrics lookup
        if hasattr(sk_metrics, name_or_func):
            scorer = getattr(sk_metrics, name_or_func)
            maximize = not name_or_func.startswith(("neg_", "log_loss", "mean_", "mse", "mae", "brier"))
            return scorer, name_or_func, maximize
        
        raise ValueError(
            f"Unknown scoring function '{name_or_func}'. "
            f"Must be a valid sklearn.metrics function or custom callable."
        )
    
    warnings.warn("Scorer not specified: defaulting to mean_squared_error.")
    return sk_metrics.mean_squared_error, "mean_squared_error", False

