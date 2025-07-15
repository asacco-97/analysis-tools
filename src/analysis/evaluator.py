"""High level interface for model evaluation and analysis."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from . import plots
from . import report


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
        return report.gain_curve_with_gini(
            self.data,
            self.actual_col,
            self.predicted_col,
            exposure_col=self.exposure_col,
            split_name=None,
            **kwargs,
        )

    def plot_lift(self, **kwargs) -> Any:
        return report.lift_chart(
            self.data,
            self.actual_col,
            self.predicted_col,
            exposure_col=self.exposure_col,
            split_name=None,
            **kwargs,
        )

    def plot_residuals(self, **kwargs) -> Any:
        return report.crunched_residual_plot(
            self.data,
            self.actual_col,
            self.predicted_col,
            exposure_col=self.exposure_col,
            split_name=None,
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
    def export_html(self, output_html: str = "model_analysis.html", title: str | None = None) -> None:
        report.generate_model_analysis_report(
            self.data,
            self.actual_col,
            self.predicted_col,
            split_col=self.split_col,
            exposure_col=self.exposure_col,
            output_html=output_html,
            title=title,
        )
