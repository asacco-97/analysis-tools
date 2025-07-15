"""Reusable plotting utilities for model evaluation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def cumulative_gain_plot(model) -> pd.DataFrame:
    """Return cumulative gain DataFrame and plot cumulative gain curve."""
    importance = model.get_score(importance_type="gain")
    importance_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Gain"])

    total_gain = importance_df["Gain"].sum()
    importance_df["Percent_Gain"] = importance_df["Gain"] / total_gain * 100
    importance_df = importance_df.sort_values("Percent_Gain")
    importance_df["Cumulative_Gain"] = importance_df["Percent_Gain"].cumsum()

    plt.figure(figsize=(8, 4))
    plt.plot(importance_df["Feature"], importance_df["Cumulative_Gain"], marker="o")
    plt.axhline(1, linestyle="--", color="red")
    plt.xlabel("Features")
    plt.ylabel("Cumulative Gain (%)")
    plt.title("Cumulative Gain Contribution")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

    return importance_df


def percent_gain_plot(model) -> pd.DataFrame:
    """Return percent gain DataFrame and plot bar chart of percent gain."""
    importance = model.get_score(importance_type="gain")
    importance_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Gain"])

    total_gain = importance_df["Gain"].sum()
    importance_df["Percent_Gain"] = importance_df["Gain"] / total_gain * 100
    importance_df = importance_df.sort_values("Percent_Gain", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Percent_Gain"])
    plt.axvline(1, color="red", linestyle="--")
    plt.xlabel("Percent Gain (%)")
    plt.ylabel("Features")
    plt.title("Percent Gain by Feature")
    plt.grid(True)
    plt.show()

    return importance_df


def plot_auc_curves(
    y_true: Sequence[float],
    proba_dict: Dict[str, Sequence[float]] | Sequence[float],
    title: str = "ROC Curves",
    figsize: tuple[int, int] = (7, 4),
    linewidth: float = 2.0,
    cmap: str = "tab10",
) -> Dict[str, float]:
    """Plot ROC curves for one or multiple models and return AUC scores."""
    if not isinstance(proba_dict, dict):
        proba_dict = {"Model": proba_dict}

    colors = plt.get_cmap(cmap).colors

    plt.figure(figsize=figsize)
    auc_scores: Dict[str, float] = {}

    for idx, (name, y_proba) in enumerate(proba_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        score = auc(fpr, tpr)
        auc_scores[name] = score

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {score:.4f})",
            color=colors[idx % len(colors)],
            linewidth=linewidth,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return auc_scores


def plot_error_by_group(df, target_col, pred_col, group_col, bins=10, ax=None):
    """
    Plot prediction mean, target mean, and count by a grouping variable on a given axis.
    """
    data = df.copy()

    # Bin numeric group variable
    if pd.api.types.is_numeric_dtype(data[group_col]):
        binning = pd.qcut(data[group_col], q=bins, duplicates='drop')
        data["bin"] = binning
        bin_means = data.groupby("bin")[group_col].mean()
        label_map = {interval: f"{mean:.2f}" for interval, mean in bin_means.items()}
        data["bin_label"] = data["bin"].map(label_map)
        ordered_labels = [label_map[b] for b in bin_means.index]
        group_col_final = "bin_label"
    else:
        group_col_final = group_col
        ordered_labels = sorted(data[group_col].dropna().unique())

    # Group by label
    data.rename(columns={group_col_final: group_col + " Bin"}, inplace=True)
    group_col_final = group_col + " Bin"

    grouped = data.groupby(group_col_final).agg(
        actual_mean=(target_col, 'mean'),
        predicted_mean=(pred_col, 'mean'),
        count=(target_col, 'count')
    ).reindex(ordered_labels).reset_index()

    # Use provided axis or fallback to new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    ax1 = ax

    # Bar plot for count (primary y-axis)
    sns.barplot(
        data=grouped, x=group_col_final, y='count', alpha=0.3,
        ax=ax1, color='steelblue', order=ordered_labels
    )
    ax1.set_ylabel('Count', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Line plots for predicted and actual (secondary y-axis)
    ax2 = ax1.twinx()
    sns.lineplot(
        data=grouped, x=group_col_final, y='predicted_mean',
        ax=ax2, marker='o', label='Pred Mean'
    )
    sns.lineplot(
        data=grouped, x=group_col_final, y='actual_mean',
        ax=ax2, marker='s', label='Actual Mean'
    )
    ax2.set_ylabel('Predicted vs. Actual Mean', color='black')
    ax2.set_xlabel(f'{group_col}', color='black')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title(f'Error by {group_col}')

def plot_error_by_group_grid(df, target_col, pred_col, group_cols, bins=10, ncols=2, figsize=(6, 4)):
    """
    Plot prediction error and count by multiple group columns using shared grid layout.
    Calls `plot_error_by_group` for each plot.
    """
    n_rows = -(-len(group_cols) // ncols)  # ceiling division
    fig, axes = plt.subplots(n_rows, ncols, figsize=(figsize[0]*ncols, figsize[1]*n_rows))
    axes = np.array(axes).reshape(-1)  # flatten even if 1D

    for i, group_col in enumerate(group_cols):
        plot_error_by_group(df, target_col, pred_col, group_col, bins=bins, ax=axes[i])

    # Hide any extra axes
    for j in range(len(group_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_target_vs_predictors(
    df, target, predictors, bins: int = 10,
    weight_col: str = None, group_col: str = None
):
    """
    Plot target averages segmented by group_col, with count/weight bars at bottom.
    """
    n_preds = len(predictors)
    n_cols = 2 if n_preds <= 4 else 3
    n_rows = math.ceil(n_preds / n_cols)

    # Each predictor gets 3 rows (line, bar, spacer)
    total_rows = n_rows * 3

    # Set height ratios for each 3-row group
    row_heights = [7, 4, 6] * n_rows

    gs = gridspec.GridSpec(total_rows, n_cols, height_ratios=row_heights, hspace=0.1)

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    palette = sns.color_palette("tab10")

    for i, col in enumerate(predictors):
        block = i // n_cols
        row = block * 3
        col_pos = i % n_cols

        ax_line = fig.add_subplot(gs[row, col_pos])
        ax_bar = fig.add_subplot(gs[row + 1, col_pos], sharex=ax_line)

        df_temp = df[[col, target]].copy()
        if weight_col:
            df_temp[weight_col] = df[weight_col]
        if group_col:
            df_temp[group_col] = df[group_col]

        df_temp = df_temp.dropna()

        # Binning numeric features
        unique_vals = df[col].nunique()
        if pd.api.types.is_numeric_dtype(df[col]) and unique_vals > 30:
            df_temp["bin"] = pd.qcut(df_temp[col], bins, duplicates="drop")
        else:
            df_temp["bin"] = df_temp[col].astype(str)
            if unique_vals > bins:
                top_vals = df_temp["bin"].value_counts().nlargest(bins).index
                df_temp["bin"] = df_temp["bin"].apply(lambda x: x if x in top_vals else "Other")

        df_temp["bin"] = df_temp["bin"].astype(str)

        # Line: target average by group
        if group_col:
            for j, (group_val, sub_df) in enumerate(df_temp.groupby(group_col)):
                agg = sub_df.groupby("bin").agg(avg_target=(target, "mean")).reset_index()
                sns.lineplot(data=agg, x="bin", y="avg_target", marker="o", label=str(group_val), ax=ax_line, color=palette[j])
                
                # Ensure that legend has a title
                handles, labels = ax_line.get_legend_handles_labels()
                ax_line.legend(
                    handles=handles,
                    labels=labels,
                    title=group_col,  
                )
        else:
            agg = df_temp.groupby("bin").agg(avg_target=(target, "mean")).reset_index()
            sns.lineplot(data=agg, x="bin", y="avg_target", marker="o", ax=ax_line, color="black")

        ax_line.set_ylabel(f"Average {target}")
        ax_line.set_xlabel("")
        ax_line.set_title(f"{target} by {col}")
        ax_line.tick_params(axis='x', labelbottom=False)

        # Create grid lines and pad y limits
        ax_line.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax_line.set_ylim(ax_line.get_ylim()[0] * 0.9, ax_line.get_ylim()[1] * 1.1)  # add 10% padding on top and bottom
        ax_line.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

        # Bar: count or weight
        if weight_col:
            if group_col:
                bar_data = (
                    df_temp.groupby(["bin", group_col])[weight_col]
                    .sum()
                    .reset_index()
                )
                sns.barplot(
                    data=bar_data, x="bin", y=weight_col, hue=group_col,
                    ax=ax_bar, dodge=True, palette=palette
                )
                ax_bar.legend().remove()
            else:
                bar_data = (
                    df_temp.groupby("bin")[weight_col]
                    .sum()
                    .reset_index(name="weight")
                )
                sns.barplot(data=bar_data, x="bin", y="weight", ax=ax_bar, color="steelblue")
        else:
            if group_col:
                bar_data = (
                    df_temp.groupby(["bin", group_col])[target]
                    .count()
                    .reset_index(name="count")
                )
                sns.barplot(
                    data=bar_data, x="bin", y="count", hue=group_col,
                    ax=ax_bar, dodge=True, palette=palette
                )
                ax_bar.legend().remove()
            else:
                bar_data = (
                    df_temp.groupby("bin")[target]
                    .count()
                    .reset_index(name="count")
                )
                sns.barplot(
                    data=bar_data, x="bin", y="count",
                    ax=ax_bar, dodge=True, color="steelblue"
                )
                ax_bar.legend().remove()

        ax_bar.set_xlabel("")
        ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha="right")

        # Create grid lines and pad y limits
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.1)
        ax_bar.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

    plt.subplots_adjust(hspace=0.15, wspace=0.3)
    plt.show()

def plot_variable_distributions(df: pd.DataFrame, variables: Iterable[str], bins: int = 30) -> None:
    """Plot distributions for numeric and categorical variables."""
    n_vars = len(variables)
    n_cols = 2 if n_vars <= 4 else 3
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(variables):
        ax = axes[i]
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax, color="tab:blue")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        else:
            val_counts = df[col].value_counts().head(20)
            sns.barplot(x=val_counts.index.astype(str), y=val_counts.values, ax=ax, color="tab:orange")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

