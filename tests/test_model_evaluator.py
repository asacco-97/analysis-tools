import pandas as pd
from analysis.report import ModelAnalysisReport


def test_metrics():
    df = pd.DataFrame({"actual": [1, 2, 3], "pred": [1, 2, 4]})
    rep = ModelAnalysisReport(df, actual_col="actual", predicted_col="pred")
    assert rep.mae() == 0.3333333333333333
    assert round(rep.mse(), 2) == 0.33


def test_partial_gini_plot_returns_figure(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    df = pd.DataFrame({"actual": [0, 1, 0, 1], "pred": [0, 1, 0, 1]})
    rep = ModelAnalysisReport(df, actual_col="actual", predicted_col="pred")
    fig = rep.plot_partial_gini(top_percent=50)
    assert hasattr(fig, "savefig")

def test_balanced_spread_returns_figure(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    df = pd.DataFrame({"actual": [1, 2, 3], "pred": [0.5, 2.5, 3.5]})
    rep = ModelAnalysisReport(df, actual_col="actual", predicted_col="pred")
    fig = rep.plot_balanced_spread()
    assert hasattr(fig, "savefig")
