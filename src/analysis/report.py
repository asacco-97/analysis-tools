"""HTML reporting utilities for model evaluation plots."""
from __future__ import annotations

import base64
import io
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig_to_base64_png(fig: plt.Figure) -> str:
    """Return a figure as a base64 encoded PNG for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded}" />'


# --- Residual utilities -----------------------------------------------------

def tweedie_deviance_residuals(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
    """Compute deviance residuals for a Tweedie GLM."""
    y = np.asarray(y)
    mu = np.asarray(mu)

    if np.any(mu <= 0):
        raise ValueError("Fitted values mu must be strictly positive for Tweedie deviance residuals.")
    if np.any(y < 0):
        raise ValueError("Observed values y must be non-negative for Tweedie.")

    term1 = np.where(
        (y == 0),
        -mu ** (2 - p) / ((1 - p) * (2 - p)),
        (y * (y ** (1 - p) - mu ** (1 - p))) / (1 - p) - (y ** (2 - p) - mu ** (2 - p)) / (2 - p),
    )
    deviance = 2 * term1
    residuals = np.sign(y - mu) * np.sqrt(np.abs(deviance))
    return residuals


def plot_deviance_residuals(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    exposure_col: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    split_name: str | None = None,
    p: float = 1.70,
    num_groups: int = 400,
    title: str | None = None,
) -> plt.Figure:
    """Return a plot of mean deviance residuals vs. fitted values."""
    fig, ax = plt.subplots(figsize=figsize)

    residual = tweedie_deviance_residuals(df[actual_col], df[predicted_col], p)
    df = df.copy()
    df["residual"] = residual
    df["group"] = pd.qcut(df[predicted_col], num_groups, duplicates="drop")
    df["normalized_residual"] = (df[actual_col] - df[predicted_col]) / np.sqrt(df[predicted_col])

    grouped = (
        df.groupby("group")
        .agg(
            count=(predicted_col, "count"),
            avg_predicted=(predicted_col, "mean"),
            avg_residual=("normalized_residual", "mean"),
            std_residual=("normalized_residual", "std"),
            mean_abs_residual=("normalized_residual", lambda x: np.mean(np.abs(x))),
        )
        .reset_index()
    )

    group_size = int(np.nanpercentile(grouped["count"], 50))
    ax.scatter(grouped["avg_predicted"], grouped["avg_residual"], s=8, alpha=0.3)
    ax.axhline(0, linestyle="--", color="gray")
    ax.errorbar(
        grouped["avg_predicted"],
        grouped["avg_residual"],
        yerr=grouped["std_residual"],
        fmt="o",
        alpha=0.6,
        capsize=2,
        label="±1 SD",
    )
    ax.plot(
        grouped["avg_predicted"],
        grouped["mean_abs_residual"],
        marker="x",
        linestyle="--",
        color="red",
        label="Mean |Residual|",
    )

    effective_title = title or "Avg. and Std. of Deviance Residuals vs Fitted"
    effective_title += f"\n(Group Size: {group_size})"
    if split_name:
        effective_title += f" ({split_name})"
    ax.set_title(effective_title, fontsize=14)
    ax.set_xlabel("Average Fitted Value")
    ax.set_ylabel("Average Residual")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


# --- Gain and lift charts ---------------------------------------------------

def gain_curve_with_gini(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    exposure_col: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    split_name: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Return a cumulative gain curve with Gini index."""
    df = df.copy()
    if exposure_col:
        df["weighted_actual"] = df[actual_col] * df[exposure_col]
        df["exposure"] = df[exposure_col]
    else:
        df["weighted_actual"] = df[actual_col]
        df["exposure"] = 1

    df.sort_values(predicted_col, ascending=False, inplace=True)
    df["cum_exposure"] = df["exposure"].cumsum() / df["exposure"].sum()
    df["cum_actual"] = df["weighted_actual"].cumsum() / df["weighted_actual"].sum()

    x = np.insert(df["cum_exposure"].values, 0, 0)
    y = np.insert(df["cum_actual"].values, 0, 0)

    df_perfect = df.sort_values("weighted_actual", ascending=False).copy()
    df_perfect["cum_exposure"] = df_perfect["exposure"].cumsum() / df_perfect["exposure"].sum()
    df_perfect["cum_actual"] = df_perfect["weighted_actual"].cumsum() / df_perfect["weighted_actual"].sum()
    perfect_x = np.insert(df_perfect["cum_exposure"].values, 0, 0)
    perfect_y = np.insert(df_perfect["cum_actual"].values, 0, 0)

    gini_model = 2 * (np.trapz(y, x) - 0.5)
    gini_perfect = 2 * (np.trapz(perfect_y, perfect_x) - 0.5)
    normalized_gini = gini_model / gini_perfect

    effective_title = title or "Gain Curve / Lorenz"
    if split_name:
        effective_title += f" ({split_name})"
    effective_title += f"\nGini={gini_model:.3f}, Norm Gini={normalized_gini:.3f}"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, label="Model", color="blue")
    ax.plot(perfect_x, perfect_y, color="green", label="Perfect")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.fill_between(x, y, np.linspace(0, 1, len(x)), alpha=0.3)
    ax.set_title(effective_title, fontsize=14)
    ax.set_xlabel("Cumulative % of Exposure")
    ax.set_ylabel("Cumulative % of Loss")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig


def lift_chart(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    exposure_col: str | None = None,
    n_bins: int = 10,
    figsize: tuple[int, int] = (8, 5),
    annotate_points: bool = False,
    title: str | None = None,
    split_name: str | None = None,
) -> plt.Figure:
    """Return a lift chart comparing average actual vs. predicted by deciles."""
    df = df.copy()
    if exposure_col:
        df["weighted_actual"] = df[actual_col] * df[exposure_col]
        df["exposure"] = df[exposure_col]
    else:
        df["weighted_actual"] = df[actual_col]
        df["exposure"] = 1

    df["bin"] = pd.qcut(df[predicted_col], n_bins, duplicates="drop")

    grouped = (
        df.groupby("bin")
        .apply(
            lambda g: pd.Series({
                "Avg Actual": np.average(g[actual_col], weights=g["exposure"]),
                "Avg Predicted": np.average(g[predicted_col], weights=g["exposure"]),
                "Total Exposure": g["exposure"].sum(),
            })
        )
        .reset_index()
    )

    effective_title = title or "Lift Chart by Prediction Decile"
    if split_name:
        effective_title += f" ({split_name})"

    fig, ax = plt.subplots(figsize=figsize)
    x_values = np.arange(1, len(grouped) + 1)

    ax.plot(x_values, grouped["Avg Actual"], marker="o", linestyle="-", color="navy", label="Actual")
    ax.plot(x_values, grouped["Avg Predicted"], marker="x", linestyle="--", color="darkorange", label="Predicted")

    if annotate_points:
        for x, y in zip(x_values, grouped["Avg Actual"]):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    ax.set_title(effective_title, fontsize=14)
    ax.set_xlabel("Decile of Predicted Pure Premium", fontsize=12)
    ax.set_ylabel("Average Actual / Predicted Pure Premium", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def crunched_residual_plot(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    exposure_col: str | None = None,
    num_groups: int = 400,
    figsize: tuple[int, int] = (8, 5),
    split_name: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Return a plot of average residuals vs. fitted values."""
    df = df.copy()
    residual = df[actual_col] - df[predicted_col]
    df["residual"] = residual

    df["group"] = pd.qcut(df[predicted_col], num_groups, duplicates="drop")
    grouped = (
        df.groupby("group")
        .agg(
            count=(predicted_col, "count"),
            avg_predicted=(predicted_col, "mean"),
            avg_residual=("residual", "mean"),
        )
        .reset_index()
    )

    group_size = int(np.nanpercentile(grouped["count"], 50))

    effective_title = title or "Crunched Residuals vs Fitted"
    effective_title += f"\n(Group Size: {group_size})"
    if split_name:
        effective_title += f" ({split_name})"

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(grouped["avg_predicted"], grouped["avg_residual"], s=10, alpha=0.6, color="darkred")
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_title(effective_title, fontsize=14)
    ax.set_xlabel("Average Fitted Value")
    ax.set_ylabel("Average Residual")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig


# --- Report assembly --------------------------------------------------------

def generate_model_analysis_report(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    split_col: str = "split",
    exposure_col: str | None = None,
    output_html: str = "model_analysis.html",
    gain_kwargs: Dict[str, Any] | None = None,
    lift_kwargs: Dict[str, Any] | None = None,
    residual_kwargs: Dict[str, Any] | None = None,
    dev_residual_kwargs: Dict[str, Any] | None = None,
    title: str | None = None,
) -> None:
    """Generate a full HTML model analysis report."""
    gain_kwargs = gain_kwargs or {}
    lift_kwargs = lift_kwargs or {}
    residual_kwargs = residual_kwargs or {}
    dev_residual_kwargs = dev_residual_kwargs or {}

    html_output = "<html><head><title>Model Analysis Report</title></head><body>"
    html_output += "<h1>Model Analysis Report</h1>"

    html_output += "<h2>Gain Curve / Lorenz Curve with Gini</h2><div style='display:flex;flex-wrap:wrap'>"
    for split_value in df[split_col].unique():
        df_split = df[df[split_col] == split_value].copy()
        fig_gain = gain_curve_with_gini(
            df_split,
            actual_col,
            predicted_col,
            exposure_col,
            split_name=split_value,
            title=title,
            **gain_kwargs,
        )
        html_output += f"<div style='margin:10px'><h4>{split_value} Split</h4>{fig_to_base64_png(fig_gain)}</div>"
    html_output += "</div>"

    html_output += "<h2>Lift Charts</h2><div style='display:flex;flex-wrap:wrap'>"
    for split_value in df[split_col].unique():
        df_split = df[df[split_col] == split_value].copy()
        fig_lift = lift_chart(
            df_split,
            actual_col,
            predicted_col,
            exposure_col,
            split_name=split_value,
            title=title,
            **lift_kwargs,
        )
        html_output += f"<div style='margin:10px'><h4>{split_value} Split</h4>{fig_to_base64_png(fig_lift)}</div>"
    html_output += "</div>"

    html_output += "<h2>Crunched Residual Plots</h2><div style='display:flex;flex-wrap:wrap'>"
    for split_value in df[split_col].unique():
        df_split = df[df[split_col] == split_value].copy()
        fig_resid = crunched_residual_plot(
            df_split,
            actual_col,
            predicted_col,
            exposure_col,
            split_name=split_value,
            title=title,
            **residual_kwargs,
        )
        html_output += f"<div style='margin:10px'><h4>{split_value} Split</h4>{fig_to_base64_png(fig_resid)}</div>"
    html_output += "</div>"

    html_output += "<h2>Avg. and Std. of Deviance Residuals Plots</h2><div style='display:flex;flex-wrap:wrap'>"
    for split_value in df[split_col].unique():
        df_split = df[df[split_col] == split_value].copy()
        fig_avg_resid = plot_deviance_residuals(
            df_split,
            actual_col=actual_col,
            predicted_col=predicted_col,
            exposure_col=exposure_col,
            split_name=split_value,
            title=title,
            **dev_residual_kwargs,
        )
        html_output += f"<div style='margin:10px'><h4>{split_value} Split</h4>{fig_to_base64_png(fig_avg_resid)}</div>"
    html_output += "</div>"

    html_output += "</body></html>"

    with open(output_html, "w") as f:
        f.write(html_output)

    print(f"✅ Analysis report generated at {output_html}")
