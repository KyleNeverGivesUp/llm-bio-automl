"""Produce the final broad-only submission, selected on the calibrated folds.

Pulls together everything from this phase:
  - the completed menu (drops `molformer` per its judge verdict; keeps `mlp_head`),
  - the calibrated cluster folds (`folds_calibrated.json`) for judge-faithful OOF,
  - honest stacking + Set-1-judge scoring.

It re-runs the strong base set on the calibrated folds (refit_full -> clean test
predictions), stacks them, judges the ensemble on Set 1, compares against the
current scaffold-fold best (baseline+mlp = 0.632), and writes the better one as a
valid 513-row submission.

Usage:
    uv run python -m scripts.finalize
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from pathlib import Path

import pandas as pd

from scripts.run_menu import build_menu
from src.aggregator import aggregate
from src.analog_judge import judge_csv
from src.cv_runner import run_plan_cv
from src.schemas import FoldSpec

DATA_DIR = Path("data/pxr_activity")
CACHE_DIR = Path("data/featurizer_cache")
RUN_DIR = Path("outputs/final")
FOLDS = DATA_DIR / "folds_calibrated.json"

# Drop molformer (single-model RAE ~1.0 on analogs, hurts the ensemble — RESULTS.md §4).
REPS = ["morgan", "maccs", "avalon", "rdkit_descriptors", "chemberta", "chemberta100m", "fusion_desc_cb"]
DENSE_REPS = ["rdkit_descriptors", "chemberta", "chemberta100m", "fusion_desc_cb"]
GBDT_LINEAR = ["ridge", "random_forest", "xgboost", "lightgbm", "catboost"]  # no elastic_net (catastrophic on analogs)


def write_submission(ens_test_csv: Path, out_csv: Path) -> int:
    """Map an ensemble's 513 predictions onto the official submission columns."""
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    preds = pd.read_csv(ens_test_csv)
    pred_map = dict(zip(preds["Molecule Name"], preds["pEC50"]))
    sub = sample.copy()
    sub["pEC50"] = sub["Molecule Name"].map(pred_map)
    sub = sub[list(sample.columns)]
    assert sub["pEC50"].isna().sum() == 0 and len(sub) == 513, "submission incomplete/invalid"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return len(sub)


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    folds = FoldSpec.from_json(FOLDS)
    plans_dir = RUN_DIR / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    plans = build_menu(REPS, GBDT_LINEAR, n_seeds=1) + build_menu(DENSE_REPS, ["mlp_head"], n_seeds=1)
    print(f"Calibrated folds: {folds.strategy} (cutoff {json.loads(FOLDS.read_text()).get('cluster_cutoff')}) | "
          f"{len(plans)} base plans | train {len(train_df)} test {len(test_df)}\n")

    ok_dirs = []
    for i, plan in enumerate(plans, 1):
        try:
            run_plan_cv(plan, train_df, test_df, folds, out_dir=plans_dir / plan.plan_id,
                        cache_dir=CACHE_DIR, refit_full=True)
            ok_dirs.append(plans_dir / plan.plan_id)
            print(f"[{i}/{len(plans)}] {plan.plan_id} ok")
        except Exception as e:
            print(f"[{i}/{len(plans)}] {plan.plan_id} FAILED: {type(e).__name__}: {e}")

    # Stack on the calibrated OOF, judge the ensemble on Set 1.
    rep = aggregate(ok_dirs, RUN_DIR)
    cal = judge_csv(RUN_DIR / "ensemble" / "test_predictions.csv")
    print(f"\nCalibrated-fold ensemble ({rep['best_method']}): "
          f"CV={rep['ensemble_rae']:.4f}  JUDGE={cal['rae']:.4f}")

    # Compare against the current scaffold-fold best (baseline + mlp).
    prev_csv = Path("outputs/ensemble_compare/baseline+mlp/ensemble/test_predictions.csv")
    prev = judge_csv(prev_csv)["rae"] if prev_csv.exists() else None
    if prev is not None:
        print(f"Scaffold-fold best (baseline+mlp) JUDGE={prev:.4f}")

    winner_csv, winner_tag, winner_rae = (
        (RUN_DIR / "ensemble" / "test_predictions.csv", "calibrated", cal["rae"])
        if prev is None or cal["rae"] <= prev
        else (prev_csv, "scaffold_baseline_mlp", prev))

    n = write_submission(winner_csv, RUN_DIR / "submission.csv")
    print(f"\nWINNER: {winner_tag} (judge {winner_rae:.4f}) -> {RUN_DIR}/submission.csv ({n} rows)")
    manifest = {"winner": winner_tag, "judge_rae": winner_rae,
                "calibrated_ensemble_judge_rae": cal["rae"], "scaffold_baseline_mlp_judge_rae": prev,
                "folds": str(FOLDS), "n_bases": len(ok_dirs), "combiner": rep["best_method"],
                "interim_lb_top5_reference": 0.538}
    (RUN_DIR / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {RUN_DIR}/submission_manifest.json")


if __name__ == "__main__":
    main()
