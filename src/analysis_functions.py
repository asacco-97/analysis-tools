"""Backwards compatibility shim for plotting functions."""
from analysis import plots

cumulative_gain_plot = plots.cumulative_gain_plot
percent_gain_plot = plots.percent_gain_plot
plot_auc_curves = plots.plot_auc_curves
plot_error_by_group = plots.plot_error_by_group
plot_target_vs_predictors = plots.plot_target_vs_predictors
plot_variable_distributions = plots.plot_variable_distributions
