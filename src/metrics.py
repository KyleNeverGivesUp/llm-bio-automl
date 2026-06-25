"""Scoring — the single source of truth for how a prediction is judged.

The competition's primary metric is **RAE (Relative Absolute Error)**:

    RAE = sum|y - y_hat|  /  sum|y - mean(y)|

i.e. our total absolute error divided by the total absolute error of the trivial
"always predict the mean" model. RAE < 1 means we beat the mean; lower is better.
MAE and R² are reported alongside as human-readable context.

Centralizing these here (rather than re-deriving them per module) guarantees the
leaderboard, the runner, and the ensemble all score identically.
"""

from __future__ import annotations

import numpy as np


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def rae(y_true, y_pred, baseline: float | None = None) -> float:
    """Relative Absolute Error (primary metric, lower is better).

    ``baseline`` is the constant the trivial model would predict. By default it is
    the mean of ``y_true`` for the rows being scored. Pass an explicit baseline
    (e.g. the *training* mean) when you want the denominator fixed to a deployable
    naive predictor rather than the held-out set's own mean.
    """
    yt, yp = _as_array(y_true), _as_array(y_pred)
    if baseline is None:
        baseline = float(yt.mean())
    denominator = float(np.sum(np.abs(yt - baseline)))
    numerator = float(np.sum(np.abs(yt - yp)))
    return numerator / denominator if denominator > 0 else float("inf")


def mae(y_true, y_pred) -> float:
    yt, yp = _as_array(y_true), _as_array(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    yt, yp = _as_array(y_true), _as_array(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def score_all(y_true, y_pred, baseline: float | None = None) -> dict:
    """All metrics at once: ``{"rae": ..., "mae": ..., "r2": ...}``."""
    return {
        "rae": rae(y_true, y_pred, baseline=baseline),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
