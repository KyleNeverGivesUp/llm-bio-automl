"""Per-row training weights from each label's measurement uncertainty.

Every ``pEC50`` in the training data ships with a standard error
(``pEC50_std.error``, range ≈ 0.003–0.74). A noisier label deserves *less*
influence on the fit, so we can down-weight it. This is a free training signal
the organizers handed us — but whether it actually helps is an empirical question
answered by the **Set-1 judge**, not assumed (see ``src/analog_judge.py``).

Schemes (all return weights normalised to mean 1, so a model's effective sample
size and regularisation scale are unchanged — only the *relative* emphasis moves):
  - ``none``     : every row weight 1 (the control).
  - ``inv_var``  : w ∝ 1/se²  — inverse-variance, the MLE-optimal weighting under
                   Gaussian heteroscedastic noise. Strongest emphasis on precise rows.
  - ``inv_se``   : w ∝ 1/se   — a gentler version of the same idea.

``floor_q`` clips ``se`` from below at that quantile before inverting. Without it
a handful of ultra-precise rows (se≈0.003 → 1/se²≈10⁵) would swamp the fit; the
floor bounds the max-to-typical weight ratio. Set ``floor_q=0`` to disable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The std.error column name carries a unit suffix, e.g.
# "pEC50_std.error (-log10(molarity))" — match by prefix so a unit change won't break us.
STD_ERROR_PREFIX = "pEC50_std.error"


def find_std_error_column(df: pd.DataFrame) -> str | None:
    """Return the pEC50 standard-error column name, or None if absent."""
    for col in df.columns:
        if col.startswith(STD_ERROR_PREFIX):
            return col
    return None


def compute_sample_weights(
    df: pd.DataFrame,
    scheme: str = "none",
    *,
    floor_q: float = 0.05,
) -> np.ndarray | None:
    """Weights (mean 1) for the rows of ``df`` under ``scheme``; None if no weighting.

    Returns ``None`` for ``scheme="none"`` (so callers can treat None as "plain fit").
    Raises if a weighting scheme is requested but the std.error column is missing —
    silently falling back to uniform would make a weighting experiment a lie.
    """
    if scheme in (None, "none"):
        return None

    col = find_std_error_column(df)
    if col is None:
        raise ValueError(
            f"weight_scheme={scheme!r} requested but no '{STD_ERROR_PREFIX}*' column found"
        )

    se = df[col].to_numpy(dtype=float)
    # Guard non-finite / non-positive errors before they reach a reciprocal.
    se = np.where(np.isfinite(se) & (se > 0), se, np.nan)
    median = float(np.nanmedian(se))
    se = np.nan_to_num(se, nan=median)

    if floor_q and floor_q > 0:
        floor = float(np.quantile(se, floor_q))
        se = np.clip(se, floor, None)

    if scheme == "inv_var":
        w = 1.0 / (se ** 2)
    elif scheme == "inv_se":
        w = 1.0 / se
    else:
        raise ValueError(f"Unknown weight_scheme {scheme!r} (expected none|inv_var|inv_se|counter_sel)")

    return w / float(np.mean(w))  # normalise to mean 1


def counter_selectivity_weights(df: pd.DataFrame, center: float = 0.5, scale: float = 0.5) -> np.ndarray:
    """Down-weight likely PXR-null assay-interference compounds (M6 / competitor lever).

    Selectivity = primary pEC50 − counter pEC50. Compounds active in BOTH assays
    (low/negative selectivity) are probable false positives → low weight; cleanly
    selective ones → ~full weight. Molecules without counter data keep weight 1.
    Returns weights normalised to mean 1.
    """
    if "counter_pEC50" not in df.columns:
        raise ValueError("counter_selectivity_weights needs a 'counter_pEC50' column")
    sel = df["pEC50"].to_numpy(float) - df["counter_pEC50"].to_numpy(float)
    w = np.ones(len(df))
    have = np.isfinite(sel)
    w[have] = 2.0 / (1.0 + np.exp(-(sel[have] - center) / scale))  # sigmoid in (0,2)
    return w / float(np.mean(w))
