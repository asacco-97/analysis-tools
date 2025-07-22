import matplotlib
matplotlib.use("Agg")
import pandas as pd

from analysis.report import ModelAnalysisReport
from analysis import plots


def test_repr_html_returns_string(monkeypatch):
    # Avoid showing any plots
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    df = pd.DataFrame({"actual": [1, 2], "pred": [1.1, 1.9], "split": ["train", "train"]})
    mar = ModelAnalysisReport(df, actual_col="actual", predicted_col="pred")
    mar.add_plot(plots.lift_chart, "Lift Chart")
    html = mar._repr_html_()
    assert isinstance(html, str)
    assert "Lift Chart" in html
