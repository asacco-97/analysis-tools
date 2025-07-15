"""Backwards compatibility shim for report utilities."""
from analysis import report

generate_model_analysis_report = report.generate_model_analysis_report

tweedie_deviance_residuals = report.tweedie_deviance_residuals
plot_deviance_residuals = report.plot_deviance_residuals
gain_curve_with_gini = report.gain_curve_with_gini
lift_chart = report.lift_chart
crunched_residual_plot = report.crunched_residual_plot
fig_to_base64_png = report.fig_to_base64_png
