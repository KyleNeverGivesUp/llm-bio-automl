"""The Approach-1 "Coder": run one modeling recipe across the frozen scaffold folds.

This is the single function that turns *any* registered (featurizer, model, params)
into honest, leak-free scores — no code generation, no per-model special-casing.

What "honest" means here, concretely:
  - We use the **frozen scaffold folds** from ``folds.json`` (NOT a random split).
    Molecules sharing a core skeleton stay in one fold, so the score reflects
    performance on *unseen chemistry* — the way the hidden test set will judge us.
  - Each training row's out-of-fold (OOF) prediction comes only from models that
    never saw that row. OOF is what we later stack/blend on (M2) and is our most
    trustworthy local estimate.
  - Test predictions are averaged across the per-fold (and per-seed) models.

Leakage guards live in three places: stateless featurization (``featurizers.py``),
per-fold-fit learned transforms (``models.py``), and the fold discipline below.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.featurizers import featurize
from src.metrics import score_all
from src.models import fit_model, make_model
from src.sample_weights import compute_sample_weights
from src.schemas import FoldSpec, MenuPlan

SMILES_COL = "SMILES"
NAME_COL = "Molecule Name"
TARGET_COL = "pEC50"


def _summarize(per_fold: list[dict], key: str) -> dict:
    vals = np.array([m[key] for m in per_fold], dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "per_fold": [float(v) for v in vals],
    }


def run_plan_cv(
    plan: MenuPlan,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds: FoldSpec,
    out_dir: str | Path,
    cache_dir: str | Path | None = None,
    refit_full: bool = False,
    aux_train_df: pd.DataFrame | None = None,
    aux_weight: float = 1.0,
    sample_weight_override: np.ndarray | None = None,
) -> dict:
    """Run ``plan`` through scaffold cross-validation and write its artifacts.

    Produces, under ``out_dir``: ``metrics.json``, ``oof_predictions.csv``,
    ``test_predictions.csv``. Returns the metrics dict.

    ``refit_full``: how the *test* predictions are produced (the OOF score is
    unaffected either way). False (default) averages the per-fold models' test
    predictions; True retrains on 100% of the training data (per seed) and predicts
    test from that — the standard final-submission move, usually a touch better and
    more stable. The CV/OOF numbers always come from the held-out folds.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # --- alignment contract: folds index train rows 0..n-1 in CSV order -------
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    fold_of_row = np.asarray(folds.fold_of_row, dtype=int)
    if len(fold_of_row) != len(train_df):
        raise ValueError(
            f"Fold/row mismatch: folds cover {len(fold_of_row)} rows but train_df "
            f"has {len(train_df)}. folds.json must be built from this exact train.csv."
        )

    smiles_train = train_df[SMILES_COL].tolist()
    smiles_test = test_df[SMILES_COL].tolist()
    y = train_df[TARGET_COL].to_numpy(dtype=float)

    # --- per-row sample weights from measurement uncertainty (optional lever) --
    # Driven by the plan: params["weight_scheme"] in {none,inv_var,inv_se}. None
    # means a plain unweighted fit. Computed once over all rows, then sliced by fold.
    weight_scheme = plan.params.get("weight_scheme", "none")
    if sample_weight_override is not None:
        sample_weight = np.asarray(sample_weight_override, dtype=float)
        weight_scheme = "override"
    else:
        sample_weight = compute_sample_weights(
            train_df, weight_scheme, floor_q=float(plan.params.get("weight_floor_q", 0.05))
        )

    # --- featurize once over all rows (stateless -> safe to compute together) --
    X_train = featurize(plan.featurizer, smiles_train, plan.params, cache_dir=cache_dir)
    X_test = featurize(plan.featurizer, smiles_test, plan.params, cache_dir=cache_dir)

    # --- optional auxiliary training data (M5): ALWAYS in training, NEVER held out --
    # Extra rows (e.g. crude/semi-pure HT-chem pEC50) augment every fold's training
    # set but never become a validation row, so OOF + the Set-1 judge stay measured on
    # the broad rows only. ``aux_weight`` globally scales the aux rows (down-weight
    # low-fidelity data). Set 1 is NOT aux data — the caller dedupes it out upstream.
    X_aux = y_aux = w_aux_base = None
    if aux_train_df is not None and len(aux_train_df) > 0:
        aux_train_df = aux_train_df.reset_index(drop=True)
        X_aux = featurize(plan.featurizer, aux_train_df[SMILES_COL].tolist(), plan.params, cache_dir=cache_dir)
        y_aux = aux_train_df[TARGET_COL].to_numpy(dtype=float)
        sw_aux = compute_sample_weights(aux_train_df, weight_scheme,
                                        floor_q=float(plan.params.get("weight_floor_q", 0.05)))
        w_aux_base = (sw_aux if sw_aux is not None else np.ones(len(y_aux))) * float(aux_weight)

    def _combine_weights(sw_tr_broad, n_broad):
        """Concatenate broad + aux per-row weights (or None if neither is weighted)."""
        if sw_tr_broad is None and w_aux_base is None:
            return None
        wb = sw_tr_broad if sw_tr_broad is not None else np.ones(n_broad)
        return np.concatenate([wb, w_aux_base]) if w_aux_base is not None else wb

    seeds = plan.seeds or [42]
    oof = np.full(len(train_df), np.nan, dtype=float)
    test_fold_preds: list[np.ndarray] = []
    fold_ids = sorted(set(int(f) for f in fold_of_row))
    weight_applied = False  # set True if any fit actually consumed the sample weights

    for k in fold_ids:
        val_mask = fold_of_row == k
        tr_mask = ~val_mask
        X_tr, y_tr = X_train[tr_mask], y[tr_mask]
        X_va = X_train[val_mask]
        sw_tr = sample_weight[tr_mask] if sample_weight is not None else None
        # append aux rows to this fold's TRAINING set (never to validation)
        if X_aux is not None:
            sw_tr = _combine_weights(sw_tr, X_tr.shape[0])
            X_tr = np.vstack([X_tr, X_aux])
            y_tr = np.concatenate([y_tr, y_aux])

        # average across seeds (multi-seed is a no-op for deterministic models)
        seed_val, seed_test = [], []
        for seed in seeds:
            model = make_model(plan.model, plan.params, seed=seed)
            _, used = fit_model(model, X_tr, y_tr, sample_weight=sw_tr)
            weight_applied = weight_applied or used
            seed_val.append(model.predict(X_va))
            seed_test.append(model.predict(X_test))
        oof[val_mask] = np.mean(seed_val, axis=0)
        test_fold_preds.append(np.mean(seed_test, axis=0))

    # --- correctness guards ---------------------------------------------------
    if np.isnan(oof).any():
        missing = int(np.isnan(oof).sum())
        raise RuntimeError(f"{missing} train rows received no OOF prediction (fold coverage bug)")

    # Test predictions: full-data refit (train on 100%) or average the fold models.
    if refit_full:
        X_full, y_full = X_train, y
        sw_full = sample_weight
        if X_aux is not None:
            sw_full = _combine_weights(sample_weight, X_train.shape[0])
            X_full = np.vstack([X_train, X_aux])
            y_full = np.concatenate([y, y_aux])
        seed_full = []
        for seed in seeds:
            model = make_model(plan.model, plan.params, seed=seed)
            _, used = fit_model(model, X_full, y_full, sample_weight=sw_full)
            weight_applied = weight_applied or used
            seed_full.append(model.predict(X_test))
        test_pred = np.mean(seed_full, axis=0)
        test_pred_method = "full_refit"
    else:
        test_pred = np.mean(np.vstack(test_fold_preds), axis=0)
        test_pred_method = "fold_avg"

    # --- metrics: per-fold (each scored vs its own mean) + pooled OOF ---------
    per_fold = [score_all(y[fold_of_row == k], oof[fold_of_row == k]) for k in fold_ids]
    pooled = score_all(y, oof)  # baseline = global train mean -> comparable to a real submission

    metrics = {
        "plan_id": plan.plan_id,
        "name": plan.name,
        "featurizer": plan.featurizer,
        "model": plan.model,
        "params": plan.params,
        "seeds": seeds,
        "weight_scheme": weight_scheme,
        "sample_weight_applied": bool(weight_applied),
        "n_aux_rows": int(len(y_aux)) if y_aux is not None else 0,
        "aux_weight": float(aux_weight) if X_aux is not None else None,
        "test_pred_method": test_pred_method,
        "skill_ref": plan.skill_ref,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_features": int(X_train.shape[1]),
        "cv": {
            "rae": _summarize(per_fold, "rae"),
            "mae": _summarize(per_fold, "mae"),
            "r2": _summarize(per_fold, "r2"),
        },
        "oof": pooled,                 # pooled OOF metrics (headline)
        "primary_metric": "rae",
        "score": pooled["rae"],        # convenience for the leaderboard sort (lower is better)
        "oof_path": str(out_dir / "oof_predictions.csv"),
        "test_path": str(out_dir / "test_predictions.csv"),
        "runtime_sec": round(time.time() - started, 2),
        "status": "ok",
        "error": None,
    }

    # --- write artifacts ------------------------------------------------------
    pd.DataFrame(
        {
            "row_id": np.arange(len(train_df)),
            "fold": fold_of_row,
            NAME_COL: train_df[NAME_COL],
            SMILES_COL: train_df[SMILES_COL],
            "y_true": y,
            "y_pred": oof,
        }
    ).to_csv(out_dir / "oof_predictions.csv", index=False)

    pd.DataFrame(
        {
            SMILES_COL: test_df[SMILES_COL],
            NAME_COL: test_df[NAME_COL],
            TARGET_COL: test_pred,
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
