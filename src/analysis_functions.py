import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import roc_curve, auc

def cumulative_gain_plot(model) -> pd.DataFrame:

    # Get feature importances based on gain
    importance = model.get_score(importance_type='gain')

    # Convert to DataFrame for easier handling
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])

    # Calculate total gain
    total_gain = importance_df['Gain'].sum()

    # Calculate percent contribution of each feature to the total gain
    importance_df['Percent_Gain'] = (importance_df['Gain'] / total_gain) * 100
    importance_df = importance_df.sort_values("Percent_Gain")

    # Calculate cumulative percentage of total gain
    importance_df['Cumulative_Gain'] = importance_df['Percent_Gain'].cumsum()

    # Plot the cumulative gain to see how much of the total gain is covered by the top features
    plt.figure(figsize=(8, 4))
    plt.plot(importance_df['Feature'], importance_df['Cumulative_Gain'], marker='o')
    plt.axhline(1, linestyle="--", color="red")
    plt.xlabel('Features')
    plt.ylabel('Cumulative Gain (%)')
    plt.title('Cumulative Gain Contribution')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

    return importance_df

def percent_gain_plot(model) -> pd.DataFrame:

    # Get feature importances based on gain
    importance = model.get_score(importance_type='gain')

    # Convert to DataFrame for easier handling
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])

    # Calculate total gain
    total_gain = importance_df['Gain'].sum()

    # Calculate percent contribution of each feature to the total gain
    importance_df['Percent_Gain'] = (importance_df['Gain'] / total_gain) * 100

    # Sort by percent gain for better visualization
    importance_df = importance_df.sort_values("Percent_Gain", ascending=True)

    # Plot percent gain by feature
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Percent_Gain'])
    plt.axvline(1, color="red", linestyle="--")
    plt.xlabel('Percent Gain (%)')
    plt.ylabel('Features')
    plt.title('Percent Gain by Feature')
    plt.grid(True)
    plt.show()

    return importance_df

def plot_auc_curves(
    y_true,
    proba_dict,
    title: str = "ROC Curves",
    figsize: tuple = (7, 4),
    linewidth: float = 2.0,
    cmap: str = "tab10",
):
    """Plot ROC curves for one or multiple models."""
    if not isinstance(proba_dict, dict):
        proba_dict = {"Model": proba_dict}

    colors = plt.get_cmap(cmap).colors

    plt.figure(figsize=figsize)
    auc_scores = {}

    for idx, (name, y_proba) in enumerate(proba_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        score = auc(fpr, tpr)
        auc_scores[name] = score

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {score:.4f})",
            color=colors[idx % len(colors)],
            linewidth=linewidth,
        )

    # Random chance on diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return auc_scores
    
def plot_error_by_group(df, target_col, pred_col, group_col, bins=10):
    """
    Plot prediction mean, target mean, and count by a grouping variable.
    Automatically bins numeric group columns and sorts by bin mean.
    """
    data = df.copy()

    # Bin numeric group variable and create labels based on bin mean
    if pd.api.types.is_numeric_dtype(data[group_col]):
        binning = pd.qcut(data[group_col], q=bins, duplicates='drop')
        data["bin"] = binning

        # Get mean of numeric variable per bin for label
        bin_means = data.groupby("bin")[group_col].mean()
        label_map = {interval: f"{mean:.2f}" for interval, mean in bin_means.items()}
        data["bin_label"] = data["bin"].map(label_map)

        # Use numeric sort order of bin means
        ordered_labels = [label_map[b] for b in bin_means.index]
        group_col_final = "bin_label"
    else:
        group_col_final = group_col
        ordered_labels = sorted(data[group_col].unique())

    # Group by label
    data.rename(columns={group_col_final: group_col + " Bin"}, inplace=True)
    group_col_final = group_col + " Bin"

    grouped = data.groupby(group_col_final).agg(
        actual_mean=(target_col, 'mean'),
        predicted_mean=(pred_col, 'mean'),
        count=(target_col, 'count')
    ).reindex(ordered_labels).reset_index()

    # Plot
    fig, ax1 = plt.subplots(figsize=(9, 4))

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
    ax2.set_xlabel(f'Avg. Binned Value or Level for {group_col}', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.title(f'Prediction Error and Count by {group_col}')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_target_vs_predictors(df, target, predictors):
    """Plot the average of the target vs. levels of predictor variables. If predictor is numeric, create bins."""
    n_preds = len(predictors)
    n_cols = 2 if n_preds <= 4 else 3
    n_rows = math.ceil(n_preds / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(predictors):
        ax = axes[i]
        unique_vals = df[col].nunique()
        df_temp = df[[col, target]].dropna().copy()

        # If a column is numeric with > 30 levels
        if pd.api.types.is_numeric_dtype(df[col]) and unique_vals > 30:
            df_temp["bin"] = pd.qcut(df_temp[col], 20, duplicates='drop')
            df_grouped = df_temp.groupby("bin")[target].mean().reset_index()
            df_grouped["bin"] = df_grouped["bin"].astype(str)
            sns.lineplot(data=df_grouped, x="bin", y=target, marker='o', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        else:
            # Categorical or low-cardinality numeric
            df_temp[col] = df_temp[col].astype(str)
            if unique_vals > 10:
                top_vals = df_temp[col].value_counts().nlargest(10).index
                df_temp[col] = df_temp[col].apply(lambda x: x if x in top_vals else "Other")
            df_grouped = df_temp.groupby(col)[target].mean().reset_index()
            sns.barplot(data=df_grouped, x=col, y=target, ax=ax, color='tab:orange')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        ax.set_title(f"{target} by {col}")
        ax.set_ylabel("Average Value of Target")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    
def plot_variable_distributions(df, variables, bins=30):
    """Create distribution plots for a list of numeric and/or categorical variables."""
    n_vars = len(variables)
    n_cols = 2 if n_vars <= 4 else 3
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(variables):
        ax = axes[i]
        if (pd.api.types.is_numeric_dtype(df[col])) & (df[col].nunique() > 10):
            sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax, color='tab:blue')
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        else:
            val_counts = df[col].value_counts().head(20)
            sns.barplot(x=val_counts.index.astype(str), y=val_counts.values, ax=ax, color='tab:orange')
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
