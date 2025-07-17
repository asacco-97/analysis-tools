"""Utility helpers for analysis reports and plotting."""
from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np


def fig_to_base64_png(fig: plt.Figure) -> str:
    """Return a figure as a base64 encoded PNG for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded}" />'


def tweedie_deviance_residuals(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
    """Compute deviance residuals for a Tweedie GLM."""
    y = np.asarray(y)
    mu = np.asarray(mu)

    if np.any(mu <= 0):
        raise ValueError("Fitted values mu must be strictly positive for Tweedie deviance residuals.")
    if np.any(y < 0):
        raise ValueError("Observed values y must be non-negative for Tweedie.")

    term1 = np.where(
        (y == 0),
        -mu ** (2 - p) / ((1 - p) * (2 - p)),
        (y * (y ** (1 - p) - mu ** (1 - p))) / (1 - p) - (y ** (2 - p) - mu ** (2 - p)) / (2 - p),
    )
    deviance = 2 * term1
    residuals = np.sign(y - mu) * np.sqrt(np.abs(deviance))
    return residuals
