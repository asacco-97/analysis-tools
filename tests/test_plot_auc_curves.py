import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

# Ensure src directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis_functions import plot_auc_curves


def test_perfect_predictions_auc_is_one(monkeypatch):
    # Avoid opening plot windows
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    y_true = [0, 1, 0, 1]
    y_scores = [0, 1, 0, 1]

    auc_scores = plot_auc_curves(y_true, y_scores)

    assert auc_scores["Model"] == 1.0

