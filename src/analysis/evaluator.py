"""High level interface for model evaluation and analysis."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import plots, report, tabulation


class ModelEvaluator:
    """Bundle metrics, plots and report generation for a model."""

    def __init__(
        self,
        data: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
        *,
        exposure_col: str | None = None,
        split_col: str = "split",
    ) -> None:
        self.data = data.copy()
        self.actual_col = actual_col
        self.predicted_col = predicted_col
        self.exposure_col = exposure_col
        self.split_col = split_col

    # ------------------------------------------------------------------
    # Metrics
    def mse(self) -> float:
        """Return mean squared error."""
        return float(((self.data[self.actual_col] - self.data[self.predicted_col]) ** 2).mean())

    def mae(self) -> float:
        """Return mean absolute error."""
        return float((self.data[self.actual_col] - self.data[self.predicted_col]).abs().mean())

    def dislocation(self) -> pd.Series:
        """Return the difference between actuals and predictions."""
        return self.data[self.actual_col] - self.data[self.predicted_col]

    # ------------------------------------------------------------------
    # Plots
    def plot_gain(self, **kwargs) -> Any:
        return plots.gain_curve_with_gini(
            self.data,
            self.actual_col,
            self.predicted_col,
            exposure_col=self.exposure_col,
            split_name=None,
            **kwargs,
        )

    def plot_lift(self, **kwargs) -> Any:
        return plots.lift_chart(
            self.data,
            self.actual_col,
            self.predicted_col,
            exposure_col=self.exposure_col,
            split_name=None,
            **kwargs,
        )

    def plot_residuals(self, **kwargs) -> Any:
        return plots.crunched_residual_plot(
            self.data,
            self.actual_col,
            self.predicted_col,
            exposure_col=self.exposure_col,
            split_name=None,
            **kwargs,
        )

    def plot_partial_gini(self, top_percent: int = 20, **kwargs) -> Any:
        """Plot the partial Gini curve for the top ``top_percent`` of exposure."""
        return plots.partial_gini_plot(
            self.data,
            self.actual_col,
            self.predicted_col,
            exposure_col=self.exposure_col,
            split_name=None,
            top_percent=top_percent,
            **kwargs,
        )

    def plot_error_by_group_grid(self, group_cols: Iterable[str], **kwargs) -> Any:
        """Plot prediction error by multiple grouping variables."""
        return plots.plot_error_by_group_grid(
            self.data,
            self.actual_col,
            self.predicted_col,
            group_cols=group_cols,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Comparison utilities
    def compare_models_discrepancy(
        self,
        other_pred_col: str,
        n: int = 10,
        by: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return rows with the largest absolute difference between two predictions."""
        cols = [self.actual_col, self.predicted_col, other_pred_col]
        if by:
            cols.extend(by)
        df = self.data[cols].copy()
        df["abs_diff"] = (df[self.predicted_col] - df[other_pred_col]).abs()
        return df.nlargest(n, "abs_diff")

    # ------------------------------------------------------------------
    # Report
    def export_html(
        self,
        output_html: str = "model_analysis.html",
        title: str | None = None,
        *,
        error_group_cols: Iterable[str] | None = None,
        tabulation_vars: Iterable[str] | None = None,
        gain_kwargs: Dict[str, Any] | None = None,
        lift_kwargs: Dict[str, Any] | None = None,
        residual_kwargs: Dict[str, Any] | None = None,
        residual_fit_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        report.generate_model_analysis_report(
            self.data,
            self.actual_col,
            self.predicted_col,
            split_col=self.split_col,
            exposure_col=self.exposure_col,
            output_html=output_html,
            error_group_cols=error_group_cols,
            tabulation_vars=tabulation_vars,
            title=title,
            gain_kwargs=gain_kwargs,
            lift_kwargs=lift_kwargs,
            residual_kwargs=residual_kwargs,
            residual_fit_kwargs=residual_fit_kwargs,
        )

    def tabulate(
        self,
        group_vars: Iterable[str],
        *,
        output_html: str,
        n_bins: int = 5,
        factor: bool = False,
    ) -> str | None:
        """Return HTML tabulations and write them to ``output_html`` if provided."""
        tabulation.generate_and_save_tabulations(
            df=self.data,
            prediction_col=self.predicted_col,
            truth_col=self.actual_col,
            group_vars=list(group_vars),
            split_col=self.split_col,
            weights_col=self.exposure_col,
            n_bins=n_bins,
            factor=factor,
            output_html=output_html,
        )
        return None

