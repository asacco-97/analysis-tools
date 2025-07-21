"""Backwards compatibility shim for report utilities."""
from analysis import plots, report, utils

generate_model_analysis_report = report.generate_model_analysis_report

tweedie_deviance_residuals = utils.tweedie_deviance_residuals
plot_deviance_residuals = plots.plot_residual_fit
gain_curve_with_gini = plots.gain_curve_with_gini
lift_chart = plots.lift_chart
crunched_residual_plot = plots.crunched_residual_plot
fig_to_base64_png = utils.fig_to_base64_png
partial_gini_plot = plots.partial_gini_plot
