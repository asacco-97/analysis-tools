import pandas as pd
import numpy as np

def generate_and_save_tabulations(
    df,
    prediction_col,
    truth_col,
    group_vars,
    split_col,
    weights_col=None,
    n_bins=5,
    factor: bool = False,
    output_html="tabulations.html"
):
    html_sections = []

    for var in group_vars:
        df_copy = df.copy()

        # Bin numeric vars
        if np.issubdtype(df_copy[var].dtype, np.number) and df_copy[var].nunique() > n_bins:
            group_col = f"{var}_binned"
            df_copy[group_col] = pd.qcut(df_copy[var], q=n_bins, duplicates="drop")
        else:
            group_col = var

        # Compute intermediate fields
        df_copy["_w"] = df_copy[weights_col] if weights_col else 1.0
        df_copy["_pred_weighted"] = df_copy[prediction_col] * df_copy["_w"]
        df_copy["_truth_weighted"] = df_copy[truth_col] * df_copy["_w"]

        # If factor, set sum of predictions equal to the sum of ground truth
        if factor:
            # Compute group-wise factors
            fct_df = (
                df_copy.groupby(split_col)
                .agg(
                    truth_total=("_truth_weighted", "sum"),
                    pred_total=("_pred_weighted", "sum")
                )
                .assign(factor=lambda d: d["truth_total"] / d["pred_total"])
                .reset_index()
            )

            # Merge back the factor to each row in the main df_copy
            df_copy = df_copy.merge(fct_df[[split_col, "factor"]], on=split_col, how="left")

            # Apply the scaling factor per row based on the split
            df_copy["_pred_weighted"] *= df_copy["factor"]

        grouped = (
            df_copy.groupby([group_col, split_col])
            .agg(
                weight_sum=("_w", "sum"),
                pred_sum=("_pred_weighted", "sum"),
                truth_sum=("_truth_weighted", "sum"),
            )
            .reset_index()
        )

        grouped["prediction_mean"] = grouped["pred_sum"] / grouped["weight_sum"]
        grouped["truth_mean"] = grouped["truth_sum"] / grouped["weight_sum"]
        grouped["lr_error"] = grouped["truth_mean"] / grouped["prediction_mean"]

        # Round and format columns
        grouped["weight_sum"] = grouped["weight_sum"].round().astype(int)
        grouped["prediction_mean"] = grouped["prediction_mean"].round(3)
        grouped["truth_mean"] = grouped["truth_mean"].round(3)
        grouped["lr_error"] = grouped["lr_error"].round(3)

        # Pivot to wide format
        pivot = grouped.pivot(index=group_col, columns=split_col)
        pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
        pivot = pivot.reset_index()

        # Sort columns so that all split columns are together
        sort_cols = []
        for split in df_copy[split_col].unique():
            for col in pivot.columns:
                if col.endswith("_" + split):
                    sort_cols.append(col)
                else:
                    pass
        pivot = pivot[[group_col] + sort_cols]

        # Identify error columns
        error_cols = [col for col in pivot.columns if "lr_error" in col]

        # Apply color styling
        def highlight_errors(column):
            styles = []
            for val in column:
                if pd.isna(val):
                    styles.append("")
                elif 0.95 <= val <= 1.05:
                    # Soft neutral grey for ~accurate predictions
                    styles.append("color: #555555; font-weight: bold")
                elif 0.9 <= val <= 1.1:
                    # Soft yellow for slightly-inaccurate predictions
                    styles.append("color: #e0a723; font-weight: bold")  # muted gold
                elif val > 1.1:
                    # Soft red for underprediction
                    styles.append("color: #fd6060; font-weight: bold")  # soft coral red
                elif val < 0.9:
                    # Soft blue for overprediction
                    styles.append("color: #7c89f7; font-weight: bold")  # muted steel blue
                else:
                    # Neutral gray for others
                    styles.append("color: #555555; font-weight: bold")
            return styles

        styled = (
            pivot.style
            .apply(highlight_errors, subset=error_cols, axis=0)  # Apply by column
            .format(precision=3)
            .set_caption(f"<b style='font-size:16px'>{var} Tabulation</b>")
            .set_table_styles([
                {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "18px"), ("margin-bottom", "10px")]},
                {"selector": "th", "props": [
                    ("background-color", "#1f77b4"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("padding", "8px"),
                    ("border", "1px solid #ccc"),
                    ("text-align", "center")
                ]},
                {"selector": "td", "props": [
                    ("background-color", "#f0e8d6"),
                    ("padding", "8px"),
                    ("border", "1px solid #ddd"),
                    ("text-align", "center")
                ]},
                {"selector": "tr:nth-child(even) td", "props": [("background-color", "#e9eff3")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("margin", "auto"), ("width", "95%")]}
            ])
        )

        html_sections.append(styled.to_html())

    # Write final HTML file
    with open(output_html, "w") as f:
        f.write("<html><head><title>Model Tabulations</title></head><body style='font-family:sans-serif'>")
        for section in html_sections:
            f.write(section)
            f.write("<div style='height:50px;'></div>")  # Spacing between tables
        f.write("</body></html>")

    print(f"Tabulation report written to: {output_html}")


def generate_tabulations_html(
    df,
    prediction_col,
    truth_col,
    group_vars,
    split_col,
    weights_col=None,
    n_bins=5,
    factor: bool = False,
) -> str:
    """Return tabulations HTML string without writing to disk."""
    html_sections = []

    for var in group_vars:
        df_copy = df.copy()

        if np.issubdtype(df_copy[var].dtype, np.number) and df_copy[var].nunique() > n_bins:
            group_col = f"{var}_binned"
            df_copy[group_col] = pd.qcut(df_copy[var], q=n_bins, duplicates="drop")
        else:
            group_col = var

        df_copy["_w"] = df_copy[weights_col] if weights_col else 1.0
        df_copy["_pred_weighted"] = df_copy[prediction_col] * df_copy["_w"]
        df_copy["_truth_weighted"] = df_copy[truth_col] * df_copy["_w"]

        if factor:
            fct_df = (
                df_copy.groupby(split_col)
                .agg(
                    truth_total=("_truth_weighted", "sum"),
                    pred_total=("_pred_weighted", "sum"),
                )
                .assign(factor=lambda d: d["truth_total"] / d["pred_total"])
                .reset_index()
            )
            df_copy = df_copy.merge(fct_df[[split_col, "factor"]], on=split_col, how="left")
            df_copy["_pred_weighted"] *= df_copy["factor"]

        grouped = (
            df_copy.groupby([group_col, split_col])
            .agg(
                weight_sum=("_w", "sum"),
                pred_sum=("_pred_weighted", "sum"),
                truth_sum=("_truth_weighted", "sum"),
            )
            .reset_index()
        )

        grouped["prediction_mean"] = grouped["pred_sum"] / grouped["weight_sum"]
        grouped["truth_mean"] = grouped["truth_sum"] / grouped["weight_sum"]
        grouped["lr_error"] = grouped["truth_mean"] / grouped["prediction_mean"]

        grouped["weight_sum"] = grouped["weight_sum"].round().astype(int)
        grouped["prediction_mean"] = grouped["prediction_mean"].round(3)
        grouped["truth_mean"] = grouped["truth_mean"].round(3)
        grouped["lr_error"] = grouped["lr_error"].round(3)

        pivot = grouped.pivot(index=group_col, columns=split_col)
        pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
        pivot = pivot.reset_index()

        sort_cols = []
        for split in df_copy[split_col].unique():
            for col in pivot.columns:
                if col.endswith("_" + split):
                    sort_cols.append(col)
        pivot = pivot[[group_col] + sort_cols]

        error_cols = [col for col in pivot.columns if "lr_error" in col]

        def highlight_errors(column):
            styles = []
            for val in column:
                if pd.isna(val):
                    styles.append("")
                elif 0.95 <= val <= 1.05:
                    styles.append("color: #555555; font-weight: bold")
                elif 0.9 <= val <= 1.1:
                    styles.append("color: #e0a723; font-weight: bold")
                elif val > 1.1:
                    styles.append("color: #fd6060; font-weight: bold")
                elif val < 0.9:
                    styles.append("color: #7c89f7; font-weight: bold")
                else:
                    styles.append("color: #555555; font-weight: bold")
            return styles

        styled = (
            pivot.style
            .apply(highlight_errors, subset=error_cols, axis=0)
            .format(precision=3)
            .set_caption(f"<b style='font-size:16px'>{var} Tabulation</b>")
            .set_table_styles([
                {"selector": "caption", "props": [("caption-side", "top"), ("font-size", "18px"), ("margin-bottom", "10px")]},
                {"selector": "th", "props": [
                    ("background-color", "#1f77b4"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("padding", "8px"),
                    ("border", "1px solid #ccc"),
                    ("text-align", "center"),
                ]},
                {"selector": "td", "props": [
                    ("background-color", "#f0e8d6"),
                    ("padding", "8px"),
                    ("border", "1px solid #ddd"),
                    ("text-align", "center"),
                ]},
                {"selector": "tr:nth-child(even) td", "props": [("background-color", "#e9eff3")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("margin", "auto"), ("width", "95%")]},
            ])
        )

        html_sections.append(styled.to_html())

    html = "<div>" + "<div style='height:50px;'></div>".join(html_sections) + "</div>"
    return html


def generate_tabulations(
    df: pd.DataFrame,
    prediction_col: str,
    truth_col: str,
    group_vars: list,
    weights_col=None,
    split_col=None,
    n_bins=5,
):
    """
    Generate tabulations comparing prediction vs. ground truth across variables and splits.

    Args:
        df (pd.DataFrame): Input DataFrame.
        prediction_col (str): Name of the prediction column.
        truth_col (str): Name of the ground truth column.
        group_vars (list): List of variables (categorical or numeric) to group by.
        weights_col (str, optional): Name of weights column. Defaults to None.
        split_col (str, optional): Column name indicating data split (e.g. "split"). Defaults to None.
        n_bins (int): Number of bins for numeric variables. Defaults to 5.

    Returns:
        dict: Dictionary of tabulation DataFrames keyed by grouping variable name.
    """
    output = {}
    df = df.copy()

    # Handle weights
    df["_w"] = df[weights_col] if weights_col else 1.0

    # Precompute weighted values
    df["_pred_weighted"] = df[prediction_col] * df["_w"]
    df["_truth_weighted"] = df[truth_col] * df["_w"]

    for var in group_vars:
        # Bin numeric columns
        if np.issubdtype(df[var].dtype, np.number) and df[var].nunique() > n_bins:
            group_col = f"{var}_binned"
            df[group_col] = pd.qcut(df[var], q=n_bins, duplicates='drop')
        else:
            group_col = var

        # Grouping columns
        groupby_cols = [group_col]
        if split_col:
            groupby_cols.append(split_col)

        # Aggregation
        agg_dict = {
            "_w": "sum",
            "_pred_weighted": "sum",
            "_truth_weighted": "sum",
        }

        tab = df.groupby(groupby_cols).agg(agg_dict).reset_index()
        tab.rename(columns={"_w": "weight_sum"}, inplace=True)

        # Compute weighted means and log ratio error
        tab["prediction_mean"] = tab["_pred_weighted"] / tab["weight_sum"]
        tab["truth_mean"] = tab["_truth_weighted"] / tab["weight_sum"]
        tab["lr_error"] = tab["truth_mean"] / tab["prediction_mean"]

        # Final columns
        columns_to_keep = groupby_cols + ["weight_sum", "prediction_mean", "truth_mean", "lr_error"]
        output[var] = tab[columns_to_keep]

    return output

def pivot_tabulation_wide(tab_df, split_col="split", group_col="job"):
    """
    Convert a long-format tabulation DataFrame into a wide-format version,
    grouping columns by split (e.g., T and V).

    Args:
        tab_df (pd.DataFrame): Long-format tabulation output from `generate_tabulations`.
        split_col (str): Column name for data splits.
        group_col (str): Column name to group by (e.g., categorical variable like 'job').

    Returns:
        pd.DataFrame: Wide-format DataFrame with grouped columns by split.
    """
    # Identify metric columns (exclude group and split)
    value_cols = [col for col in tab_df.columns if col not in [group_col, split_col]]

    # Pivot table into wide format
    tab_wide = tab_df.pivot(index=group_col, columns=split_col, values=value_cols)

    # Reorder columns to group by split: e.g., T: all metrics, then V: all metrics
    new_cols = []
    for split in tab_df[split_col].unique():
        for metric in value_cols:
            new_cols.append((metric, split))

    # Flatten multi-index
    tab_wide = tab_wide.reindex(columns=pd.MultiIndex.from_tuples(new_cols))
    tab_wide.columns = [f"{split}_{metric}" for metric, split in tab_wide.columns]

    # Bring group_col back as a column
    tab_wide.reset_index(inplace=True)

    return tab_wide