"""The Set-1 judge — score broad-trained predictions against the real analog labels.

Why this module exists (decision 2026-06-21, "the judge can't join the competition"):
the 513-compound test set is an *analog* set (close analogs of 63 hits, activity
cliffs), a narrower distribution than the broad ~4,139 training set. Broad
scaffold-CV is NOT a reliable proxy for analog-test RAE. **Analog Set 1** (253
now-public labels) IS that distribution, so we use it as our private leaderboard
*judge*: score ourselves on it exactly as the official board would.

The one rule: the judge **never joins the competition**. Set 1 is only ever read
here to *score* predictions — it is never folded into training (that is the
deprecated ``scripts/run_analog.py`` path) and never consumed as a one-shot test.

Mechanics: the 253 Set-1 ``Molecule Name``s are a subset of the 513 test names.
So any broad-trained submission (513 rows: ``Molecule Name`` + ``pEC50``) is
judged by matching those 253 names and scoring RAE/MAE/R² against their labels —
the same RAE convention (denominator = Set-1's own mean) the official board uses.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics import score_all

DATA_DIR = Path("data/pxr_activity")
SET1_PATH = DATA_DIR / "phase1_unblinded.csv"
NAME_COL = "Molecule Name"
TARGET_COL = "pEC50"


def load_set1(path: str | Path = SET1_PATH) -> pd.DataFrame:
    """Load the 253-row Analog Set 1 judge labels: ``Molecule Name`` + ``pEC50``."""
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns or NAME_COL not in df.columns:
        raise ValueError(f"{path} missing required columns {NAME_COL!r}/{TARGET_COL!r}")
    return df[[NAME_COL, TARGET_COL]].dropna(subset=[TARGET_COL]).reset_index(drop=True)


def judge_predictions(
    pred_df: pd.DataFrame,
    set1: pd.DataFrame | None = None,
    *,
    pred_col: str = TARGET_COL,
) -> dict:
    """Score a 513-row prediction frame against Set 1 by matching ``Molecule Name``.

    ``pred_df`` must carry ``Molecule Name`` and a prediction column (default
    ``pEC50``). Returns RAE/MAE/R² over the matched analog labels plus coverage
    counts. Raises if the prediction file does not cover every Set-1 compound —
    a silent partial match would quietly understate or overstate the judge score.
    """
    set1 = load_set1() if set1 is None else set1
    if NAME_COL not in pred_df.columns or pred_col not in pred_df.columns:
        raise ValueError(f"prediction frame needs {NAME_COL!r} and {pred_col!r} columns")

    merged = set1.merge(
        pred_df[[NAME_COL, pred_col]].rename(columns={pred_col: "_pred"}),
        on=NAME_COL,
        how="left",
    )
    n_missing = int(merged["_pred"].isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing}/{len(set1)} Set-1 compounds have no prediction "
            f"(submission must cover all 513 test rows; the 253 Set-1 names are a subset)."
        )

    y_true = merged[TARGET_COL].to_numpy(float)
    y_pred = merged["_pred"].to_numpy(float)
    scores = score_all(y_true, y_pred)  # baseline = Set-1's own mean (official RAE convention)
    scores["n_judged"] = int(len(merged))
    scores["n_set1"] = int(len(set1))
    return scores


def judge_csv(path: str | Path, *, pred_col: str = TARGET_COL) -> dict:
    """Judge a submission / test_predictions CSV file on disk against Set 1."""
    return judge_predictions(pd.read_csv(path), pred_col=pred_col)
