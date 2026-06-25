"""The Aggregator (M2): combine base learners into an ensemble — honestly.

Why this is where the leaderboard is won: diverse base models make *different*
mistakes, so a good combination of them beats any single one. We already have
leak-free out-of-fold (OOF) predictions for every base plan; this module learns
how to combine them.

The one rule that must not be broken — **the combiner is itself cross-validated
on the same frozen scaffold folds**. Each base model's OOF prediction for a row
is already leak-free (it came from a base model that never trained on that row).
But if we fit a meta-model on *all* OOF rows and then score it on those same
rows, the meta-model overfits and we'd report a fantasy number. So to estimate
the ensemble's honest score we do, for each fold k: fit the combiner on the OOF
rows outside fold k, predict the rows inside fold k. The row-level guarantee
holds — no row's label ever informs its own ensemble prediction.

Combiners offered (all evaluated under the same honest protocol):
  - ``mean``  : uniform average (the dumb-but-strong baseline)
  - ``nnls``  : non-negative least-squares weights (robust convex blend)
  - ``ridge`` : Ridge meta-model on the OOF predictions (classic stacking)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import score_all

TARGET_COL = "pEC50"
NAME_COL = "Molecule Name"
SMILES_COL = "SMILES"


# --------------------------------------------------------------------------- #
# Loading & aligning base predictions
# --------------------------------------------------------------------------- #
def load_base_oof(plan_dirs: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Stack each plan's OOF predictions into a matrix aligned by ``row_id``.

    Returns ``(names, P[n_train, M], y[n_train], fold_of_row[n_train])`` and
    asserts every plan shares the same targets and the same fold assignment —
    the precondition that makes cross-validated stacking valid.
    """
    names: list[str] = []
    cols: list[np.ndarray] = []
    y_ref: np.ndarray | None = None
    fold_ref: np.ndarray | None = None

    for d in plan_dirs:
        df = pd.read_csv(Path(d) / "oof_predictions.csv").sort_values("row_id").reset_index(drop=True)
        y = df["y_true"].to_numpy(float)
        fold = df["fold"].to_numpy(int)
        if y_ref is None:
            y_ref, fold_ref = y, fold
        else:
            if not np.array_equal(fold, fold_ref):
                raise ValueError(f"{d} uses different folds than the first plan — cannot stack.")
            if not np.allclose(y, y_ref):
                raise ValueError(f"{d} has different y_true than the first plan — misaligned rows.")
        names.append(Path(d).name)
        cols.append(df["y_pred"].to_numpy(float))

    return names, np.column_stack(cols), y_ref, fold_ref


def load_base_test(plan_dirs: list[Path]) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Stack each plan's test predictions into ``P_test[n_test, M]``, aligned by molecule.

    Returns ``(names, P_test, meta)`` where ``meta`` carries SMILES + Molecule Name
    in a fixed order. Every plan is reindexed onto the first plan's molecule order.
    """
    names: list[str] = []
    cols: list[np.ndarray] = []
    meta: pd.DataFrame | None = None
    order: list | None = None

    for d in plan_dirs:
        df = pd.read_csv(Path(d) / "test_predictions.csv")
        if meta is None:
            meta = df[[SMILES_COL, NAME_COL]].copy()
            order = df[NAME_COL].tolist()
        df = df.set_index(NAME_COL).reindex(order).reset_index()
        names.append(Path(d).name)
        cols.append(df[TARGET_COL].to_numpy(float))

    return names, np.column_stack(cols), meta


# --------------------------------------------------------------------------- #
# Combiners — each: fit(P, y) -> self ; predict(P) -> np.ndarray
# --------------------------------------------------------------------------- #
class MeanCombiner:
    def fit(self, P, y):
        self.weights_ = np.full(P.shape[1], 1.0 / P.shape[1])
        return self

    def predict(self, P):
        return P @ self.weights_


class NNLSCombiner:
    """Non-negative least-squares weights. Robust: can't assign negative weight to
    a base model, which prevents the wild extrapolation Ridge stacking can do when
    base predictions are highly correlated."""

    def fit(self, P, y):
        from scipy.optimize import nnls

        w, _ = nnls(P, y)
        self.weights_ = w
        return self

    def predict(self, P):
        return P @ self.weights_


class RidgeStackCombiner:
    """Ridge meta-model on the base OOF predictions (classic stacking). ``positive``
    keeps weights non-negative for stability."""

    def __init__(self, alpha: float = 1.0, positive: bool = True):
        self.alpha = alpha
        self.positive = positive

    def fit(self, P, y):
        from sklearn.linear_model import Ridge

        self.model_ = Ridge(alpha=self.alpha, positive=self.positive).fit(P, y)
        return self

    def predict(self, P):
        return self.model_.predict(P)


COMBINERS = {
    "mean": lambda: MeanCombiner(),
    "nnls": lambda: NNLSCombiner(),
    "ridge": lambda: RidgeStackCombiner(alpha=1.0, positive=True),
}


# --------------------------------------------------------------------------- #
# Honest cross-validated evaluation of a combiner
# --------------------------------------------------------------------------- #
def cross_val_combine(P, y, fold_of_row, make_combiner) -> np.ndarray:
    """Stacked OOF: for each fold, fit the combiner on rows *outside* the fold and
    predict the rows *inside* it. The returned vector is a leak-free ensemble
    prediction for every training row."""
    stacked = np.full(len(y), np.nan)
    for k in sorted(set(int(f) for f in fold_of_row)):
        val = fold_of_row == k
        tr = ~val
        combiner = make_combiner().fit(P[tr], y[tr])
        stacked[val] = combiner.predict(P[val])
    if np.isnan(stacked).any():
        raise RuntimeError("Combiner left some rows unpredicted (fold coverage bug).")
    return stacked


def aggregate(
    plan_dirs: list[Path],
    run_dir: Path,
    methods: list[str] | None = None,
) -> dict:
    """Evaluate every combiner honestly, pick the best by OOF RAE, and write the
    ensemble's OOF + test predictions and a report. Returns the report dict."""
    methods = methods or list(COMBINERS)
    run_dir = Path(run_dir)
    ens_dir = run_dir / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    names, P_oof, y, fold_of_row = load_base_oof(plan_dirs)
    test_names, P_test, test_meta = load_base_test(plan_dirs)
    assert names == test_names, "OOF and test plan ordering diverged."

    # Best single base model (the bar the ensemble must clear by >=3%).
    base_scores = {n: score_all(y, P_oof[:, j])["rae"] for j, n in enumerate(names)}
    best_single_name = min(base_scores, key=base_scores.get)
    best_single_rae = base_scores[best_single_name]

    results = {}
    for m in methods:
        stacked_oof = cross_val_combine(P_oof, y, fold_of_row, COMBINERS[m])
        results[m] = {"oof": score_all(y, stacked_oof), "stacked_oof": stacked_oof}

    best_method = min(results, key=lambda m: results[m]["oof"]["rae"])
    best_rae = results[best_method]["oof"]["rae"]
    improvement = (best_single_rae - best_rae) / best_single_rae  # fraction; >0 means better

    # Final ensemble: fit the winning combiner on ALL OOF rows, apply to base test preds.
    final = COMBINERS[best_method]().fit(P_oof, y)
    ens_test = final.predict(P_test)
    weights = getattr(final, "weights_", None)

    # Write artifacts -------------------------------------------------------
    pd.DataFrame(
        {SMILES_COL: test_meta[SMILES_COL], NAME_COL: test_meta[NAME_COL], TARGET_COL: ens_test}
    ).to_csv(ens_dir / "test_predictions.csv", index=False)

    pd.DataFrame(
        {"row_id": np.arange(len(y)), "y_true": y, "y_pred": results[best_method]["stacked_oof"]}
    ).to_csv(ens_dir / "oof_predictions.csv", index=False)

    report = {
        "n_base_models": len(names),
        "base_models": names,
        "base_oof_rae": base_scores,
        "best_single": {"name": best_single_name, "rae": best_single_rae},
        "methods": {m: results[m]["oof"] for m in methods},
        "best_method": best_method,
        "ensemble_rae": best_rae,
        "improvement_vs_best_single": improvement,
        "beats_best_single_by_3pct": improvement >= 0.03,
        "final_weights": (
            {n: float(w) for n, w in zip(names, weights)} if weights is not None else "ridge_meta"
        ),
        "test_predictions_path": str(ens_dir / "test_predictions.csv"),
    }
    (run_dir / "ensemble_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
