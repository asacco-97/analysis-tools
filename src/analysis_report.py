import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def generate_model_analysis_report(
    df, actual_col, predicted_col,
    split_col='split', exposure_col=None,
    output_html='model_analysis.html',
    gain_kwargs=None, lift_kwargs=None, residual_kwargs=None, dev_residual_kwargs=None
):
    gain_kwargs = gain_kwargs or {}
    lift_kwargs = lift_kwargs or {}
    residual_kwargs = residual_kwargs or {}
    dev_residual_kwargs = dev_residual_kwargs or {}

    html_output = "<html><head><title>Model Analysis Report</title></head><body>"
    html_output += "<h1>Model Analysis Report</h1>"

    # Gain curves section
    html_output += "<h2>Gain Curve / Lorenz Curve with Gini</h2><div style='display:flex;flex-wrap:wrap'>"
    for split_value in df[split_col].unique():
        df_split = df[df[split_col]==split_value].copy()
        fig_gain = gain_curve_with_gini(df_split, actual_col, predicted_col, exposure_col, split_name=split_value, title=title, **gain_kwargs)
        html_output += f"<div style='margin:10px'><h4>{split_value} Split</h4>{fig_to_base64_png(fig_gain)}</div>"
    html_output += "</div>"

    # Lift charts section
    html_output += "<h2>Lift Charts</h2><div style='display:flex;flex-wrap:wrap'>"
    for split_value in df[split_col].unique():
        df_split = df[df[split_col]==split_value].copy()
        fig_lift = lift_chart(df_split, actual_col, predicted_col, exposure_col, split_name=split_value, title=title, **lift_kwargs)
        html_output += f"<div style='margin:10px'><h4>{split_value} Split</h4>{fig_to_base64_png(fig_lift)}</div>"
    html_output += "</div>"

    # Crunched residuals section
    html_output += "<h2>Crunched Residual Plots</h2><div style='display:flex;flex-wrap:wrap'>"
    for split_value in df[split_col].unique():
        df_split = df[df[split_col]==split_value].copy()
        fig_resid = crunched_residual_plot(df_split, actual_col, predicted_col, exposure_col, split_name=split_value, title=title, **residual_kwargs)
        html_output += f"<div style='margin:10px'><h4>{split_value} Split</h4>{fig_to_base64_png(fig_resid)}</div>"
    html_output += "</div>"

    # Mean and standard error of deviance residuals section
    html_output += "<h2>Avg. and Std. of Deviance Residuals Plots</h2><div style='display:flex;flex-wrap:wrap'>"
    for split_value in df[split_col].unique():
        df_split = df[df[split_col]==split_value].copy()
        fig_avg_resid = plot_deviance_residuals(
            df_split, actual_col=actual_col, predicted_col=predicted_col, exposure_col=exposure_col, split_name=split_value, title=title, **dev_residual_kwargs
        )
        html_output += f"<div style='margin:10px'><h4>{split_value} Split</h4>{fig_to_base64_png(fig_avg_resid)}</div>"
    html_output += "</div>"

    html_output += "</body></html>"

    with open(output_html, 'w') as f:
        f.write(html_output)

    print(f"✅ Analysis report generated at {output_html}")

def tweedie_deviance_residuals(y, mu, p):
    """
    Compute deviance residuals for Tweedie GLM.

    Parameters:
    -----------
    y : array-like
        Actual values.
    mu : array-like
        Fitted values (predicted means).
    p : float
        Power parameter of the Tweedie variance function.

    Returns:
    --------
    residuals : np.ndarray
        Deviance residuals.
    """
    y = np.asarray(y)
    mu = np.asarray(mu)
    
    if np.any(mu <= 0):
        raise ValueError("Fitted values mu must be strictly positive for Tweedie deviance residuals.")
    if np.any(y < 0):
        raise ValueError("Observed values y must be non-negative for Tweedie.")

    term1 = np.where(
        (y == 0),
        -mu**(2 - p) / ((1 - p) * (2 - p)),
        (y * (y ** (1 - p) - mu ** (1 - p))) / (1 - p)
        - (y ** (2 - p) - mu ** (2 - p)) / (2 - p)
    )
    
    deviance = 2 * term1
    residuals = np.sign(y - mu) * np.sqrt(np.abs(deviance))
    return residuals

def plot_deviance_residuals(
    df, actual_col, predicted_col, exposure_col=None,
    figsize=(8,6), split_name=None, p: float = 1.70, num_groups: int = 400, title=None
):
    """
    Plots crunched (binned) residual plots with mean residuals, ±1 std error bars,
    and mean absolute residual line for train vs validation side by side.
    Returns a matplotlib figure object.
    
    Expects 'split' column with values like 'T1', 'T2' for training and 'V' for validation.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    residual = tweedie_deviance_residuals(df[actual_col], df[predicted_col], p)
    df["residual"] = residual
    
    # Bin by quantiles on predictions
    df["group"] = pd.qcut(df[predicted_col], num_groups, duplicates='drop')

    # Calculate the normalized residual
    df['normalized_residual'] = (df[actual_col] - df[predicted_col]) / np.sqrt(df[predicted_col])

    # Aggregate by group
    df_grouped = df.groupby("group").agg(
        count=(predicted_col, "count"),
        avg_predicted=(predicted_col, "mean"),
        avg_residual=("normalized_residual", "mean"),
        std_residual=("normalized_residual", "std"),
        mean_abs_residual=("normalized_residual", lambda x: np.mean(np.abs(x)))
    ).reset_index()
    
    # Determine median group size to report
    group_size = int(np.nanpercentile(df_grouped["count"], 50))
    
    # Scatter mean residuals
    ax.scatter(df_grouped["avg_predicted"], df_grouped["avg_residual"], s=8, alpha=0.30, label="")
    
    # Reference line at 0
    ax.axhline(0, linestyle='--', color='gray', label="")
    
    # Plot error bars for ±1 std
    ax.errorbar(
        df_grouped["avg_predicted"], 
        df_grouped["avg_residual"],
        yerr=df_grouped["std_residual"],
        fmt='o', alpha=0.6, capsize=2, label="±1 SD"
    )
    
    # Plot mean absolute residual line
    ax.plot(df_grouped["avg_predicted"], df_grouped["mean_abs_residual"], 
            marker='x', linestyle='--', color='red', label="Mean |Residual|")

    # Set title
    effective_title = title or 'Avg. and Std. of Deviance Residuals vs Fitted'
    effective_title += f'\n(Group Size: {group_size})'
    if split_name:
        effective_title += f" ({split_name})"
   
    ax.set_title(effective_title, fontsize=14)
    ax.set_xlabel("Average Fitted Value")
    ax.set_ylabel('Average Residual')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig

def gain_curve_with_gini(df, actual_col, predicted_col, exposure_col=None,
                         figsize=(8,6), split_name=None, title=None):
    """
    Plots a cumulative gain (Lorenz) curve with Gini index and normalized Gini.
    Appends split name in title if provided.
    """
    df = df.copy()
    if exposure_col:
        df['weighted_actual'] = df[actual_col] * df[exposure_col]
        df['exposure'] = df[exposure_col]
    else:
        df['weighted_actual'] = df[actual_col]
        df['exposure'] = 1

    df.sort_values(predicted_col, ascending=False, inplace=True)
    df['cum_exposure'] = df['exposure'].cumsum() / df['exposure'].sum()
    df['cum_actual'] = df['weighted_actual'].cumsum() / df['weighted_actual'].sum()

    x = np.insert(df['cum_exposure'].values, 0, 0)
    y = np.insert(df['cum_actual'].values, 0, 0)

    df_perfect = df.sort_values('weighted_actual', ascending=False).copy()
    df_perfect['cum_exposure'] = df_perfect['exposure'].cumsum() / df_perfect['exposure'].sum()
    df_perfect['cum_actual'] = df_perfect['weighted_actual'].cumsum() / df_perfect['weighted_actual'].sum()
    perfect_x = np.insert(df_perfect['cum_exposure'].values, 0, 0)
    perfect_y = np.insert(df_perfect['cum_actual'].values, 0, 0)

    gini_model = 2 * (np.trapz(y, x) - 0.5)
    gini_perfect = 2 * (np.trapz(perfect_y, perfect_x) - 0.5)
    normalized_gini = gini_model / gini_perfect

    # Set title
    effective_title = title or "Gain Curve / Lorenz"
    if split_name:
        effective_title += f" ({split_name})"
    effective_title += f"\nGini={gini_model:.3f}, Norm Gini={normalized_gini:.3f}"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, label='Model', color='blue')
    ax.plot(perfect_x, perfect_y, color='green', label='Perfect')
    ax.plot([0,1], [0,1], '--', color='gray', label='Random')
    ax.fill_between(x, y, np.linspace(0,1,len(x)), alpha=0.3)
    ax.set_title(effective_title, fontsize=14)
    ax.set_xlabel('Cumulative % of Exposure')
    ax.set_ylabel('Cumulative % of Loss')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def lift_chart(df, actual_col, predicted_col, exposure_col=None, n_bins=10, figsize=(8,5), annotate_points=False, title=None, split_name=None):
    """
    Plots a lift chart comparing average actual vs predicted by deciles of prediction.

    Parameters
    ----------
    df : DataFrame
        The dataset containing actuals, predictions, and optional exposures.
    actual_col : str
        Column name for actual target variable.
    predicted_col : str
        Column name for predicted values.
    exposure_col : str or None
        Optional exposure column to use as weights.
    n_bins : int
        Number of quantiles to create (default 10 for deciles).
    figsize : tuple
        Size of the figure.
    annotate_points : bool
        If True, adds text labels to each point with average actual.
    title : str or None
        Custom title for the chart.
    """
    df = df.copy()
    if exposure_col:
        df['weighted_actual'] = df[actual_col] * df[exposure_col]
        df['exposure'] = df[exposure_col]
    else:
        df['weighted_actual'] = df[actual_col]
        df['exposure'] = 1

    # Create quantiles on predicted
    df['bin'] = pd.qcut(df[predicted_col], n_bins, duplicates='drop')

    # Compute weighted averages
    grouped = df.groupby('bin').apply(
        lambda g: pd.Series({
            'Avg Actual': np.average(g[actual_col], weights=g['exposure']),
            'Avg Predicted': np.average(g[predicted_col], weights=g['exposure']),
            'Total Exposure': g['exposure'].sum()
        })
    ).reset_index()

    # Build a context-aware title
    effective_title = title or "Lift Chart by Prediction Decile"
    if split_name:
        effective_title += f" ({split_name})"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    x_values = np.arange(1, len(grouped)+1)

    ax.plot(x_values, grouped['Avg Actual'], marker='o', linestyle='-', color='navy', label='Actual')
    ax.plot(x_values, grouped['Avg Predicted'], marker='x', linestyle='--', color='darkorange', label='Predicted')

    # Optionally annotate actual points
    if annotate_points:
        for x, y in zip(x_values, grouped['Avg Actual']):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    ax.set_title(effective_title, fontsize=14)
    ax.set_xlabel("Decile of Predicted Pure Premium", fontsize=12)
    ax.set_ylabel("Average Actual / Predicted Pure Premium", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig

def crunched_residual_plot(df, actual_col, predicted_col, exposure_col=None,
                           num_groups=400, figsize=(8,5), split_name=None, title=None):
    """
    Plots average residuals vs average fitted by grouped bins for zero-inflated data.
    Appends split name in title if provided.
    """
    df = df.copy()
    residual = df[actual_col] - df[predicted_col]
    df['residual'] = residual

    df['group'] = pd.qcut(df[predicted_col], num_groups, duplicates='drop')
    grouped = df.groupby('group').agg(
        count=(predicted_col, 'count'),
        avg_predicted=(predicted_col, 'mean'),
        avg_residual=('residual', 'mean')
    ).reset_index()

    # Get median group size from grouped plot
    group_size = int(np.nanpercentile(grouped["count"], 50))

    # Set title
    effective_title = title or 'Crunched Residuals vs Fitted'
    effective_title += f'\n(Group Size: {group_size})'
    if split_name:
        effective_title += f" ({split_name})"

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(grouped['avg_predicted'], grouped['avg_residual'], s=10, alpha=0.6, color='darkred')
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_title(effective_title, fontsize=14)
    ax.set_xlabel('Average Fitted Value')
    ax.set_ylabel('Average Residual')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def fig_to_base64_png(fig):
    """Renders a matplotlib figure as a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded}" />'
