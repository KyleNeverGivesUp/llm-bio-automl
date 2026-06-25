"""Broad-train + Set-1-judge experiment harness — the missing eval loop.

Replaces the deprecated ``scripts/run_analog.py`` (which folded Set 1 into training
— "the judge can't join the competition"). Here every plan is trained on the
**broad 4,139 only**, predicts all 513, and is then *scored* on the 253 Set-1
labels. We report two numbers side by side:

  - **scaffold-CV RAE** — our internal honest-but-miscalibrated proxy.
  - **Set-1 judge RAE** — the real read on the analog distribution.

Use it to (1) decide which new menu modules to keep (keep only what lowers judge
RAE) and (2) measure how badly scaffold-CV ranks models vs the judge (Spearman) —
the calibration signal for analog-faithful folds.

Usage:
    uv run python -m scripts.run_judge                 # full experiment matrix
    uv run python -m scripts.run_judge --group weights # one group only
"""

from __future__ import annotations

import os

# Set before numpy/torch import (macOS OpenMP deadlock — see scripts/run_menu.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregator import aggregate
from src.analog_judge import judge_csv
from src.cv_runner import run_plan_cv
from src.menu_config import CHEMBERTA, MOLFORMER
from src.schemas import FoldSpec, MenuPlan

DATA_DIR = Path("data/pxr_activity")
RUN_DIR = Path("outputs/judge")
CACHE_DIR = Path("data/featurizer_cache")

FUSION_CB = {"components": ["rdkit_descriptors", "chemberta_embedding"], "skill_ref": CHEMBERTA}
FUSION_MF = {"components": ["rdkit_descriptors", "molformer_embedding"], "skill_ref": MOLFORMER}
CB = {"skill_ref": CHEMBERTA}
MF = {"skill_ref": MOLFORMER}


def _plan(pid, featurizer, model, params, *, group, seeds=(42,)):
    """Build a MenuPlan. Single-seed by default: this is a *screen* for which modules
    lower the judge RAE, so speed beats the small variance-reduction of multi-seed —
    we multi-seed only the final chosen models."""
    p = MenuPlan(plan_id=pid, name=pid, featurizer=featurizer, model=model,
                 params=dict(params), seeds=list(seeds), skill_ref=params.get("skill_ref"))
    p.group = group  # type: ignore[attr-defined]
    return p


def build_experiments() -> list:
    """Curated matrix that isolates the three new menu modules against strong
    reference bases. Each plan trains broad-only; all judged on Set 1."""
    plans: list = []

    # --- reference bases (anchor the comparison; known-strong on the analog set) ---
    plans += [
        _plan("ref__desc_lightgbm", "rdkit_descriptors", "lightgbm", {}, group="ref"),
        _plan("ref__desc_xgboost", "rdkit_descriptors", "xgboost", {}, group="ref"),
        _plan("ref__fusion_cb_xgboost", "fusion", "xgboost", FUSION_CB, group="ref"),
        _plan("ref__cb_ridge", "chemberta_embedding", "ridge", {**CB, "scale": True}, group="ref"),
    ]

    # --- A. sample weights: does down-weighting noisy labels help the judge? ---
    for wid, ws in [("none", "none"), ("invse", "inv_se"), ("invvar", "inv_var")]:
        plans += [
            _plan(f"wt__desc_lightgbm__{wid}", "rdkit_descriptors", "lightgbm",
                  {"weight_scheme": ws}, group="weights"),
            _plan(f"wt__desc_xgboost__{wid}", "rdkit_descriptors", "xgboost",
                  {"weight_scheme": ws}, group="weights"),
            _plan(f"wt__fusion_cb_lightgbm__{wid}", "fusion", "lightgbm",
                  {**FUSION_CB, "weight_scheme": ws}, group="weights"),
        ]

    # --- B. mlp_head: the non-tree family, on dense reps where it works ---
    plans += [
        _plan("mlp__cb", "chemberta_embedding", "mlp_head", {**CB, "scale": True}, group="mlp"),
        _plan("mlp__molformer", "molformer_embedding", "mlp_head", {**MF, "scale": True}, group="mlp"),
        _plan("mlp__fusion_cb", "fusion", "mlp_head", {**FUSION_CB, "scale": True}, group="mlp"),
        _plan("mlp__desc", "rdkit_descriptors", "mlp_head", {"scale": True}, group="mlp"),
    ]

    # --- C. molformer: a second, different CLM — alone and fused ---
    plans += [
        _plan("mf__ridge", "molformer_embedding", "ridge", {**MF, "scale": True}, group="molformer"),
        _plan("mf__lightgbm", "molformer_embedding", "lightgbm", MF, group="molformer"),
        _plan("mf__xgboost", "molformer_embedding", "xgboost", MF, group="molformer"),
        _plan("mf__fusion_lightgbm", "fusion", "lightgbm", FUSION_MF, group="molformer"),
        _plan("mf__fusion_ridge", "fusion", "ridge", {**FUSION_MF, "scale": True}, group="molformer"),
    ]
    return plans


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation without scipy (rank both, Pearson on ranks)."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Broad-train, judge on Set 1.")
    ap.add_argument("--group", default=None, help="run only one group: ref|weights|mlp|molformer")
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--no-ensemble", action="store_true", help="skip stacking the bases")
    args = ap.parse_args()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    folds = FoldSpec.from_json(DATA_DIR / "folds.json")
    plans_dir = args.run_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    plans = [p for p in build_experiments() if args.group is None or p.group == args.group]
    print(f"Broad train: {len(train_df)} | test: {len(test_df)} | folds: {folds.n_folds} | "
          f"plans: {len(plans)}\n")
    print(f"{'plan':<34} {'scaffCV':>8} {'judge':>7} {'gap':>7} {'wt?':>4}")

    rows = []
    for plan in plans:
        try:
            m = run_plan_cv(plan, train_df, test_df, folds,
                            out_dir=plans_dir / plan.plan_id, cache_dir=CACHE_DIR, refit_full=True)
            j = judge_csv(plans_dir / plan.plan_id / "test_predictions.csv")
            scaff, judge = m["score"], j["rae"]
            rows.append({"plan_id": plan.plan_id, "group": plan.group, "scaffold_cv_rae": scaff,
                         "judge_rae": judge, "judge_mae": j["mae"], "judge_r2": j["r2"],
                         "gap": judge - scaff, "weight_applied": m["sample_weight_applied"]})
            print(f"{plan.plan_id:<34} {scaff:8.4f} {judge:7.4f} {judge-scaff:+7.4f} "
                  f"{'yes' if m['sample_weight_applied'] else '-':>4}")
        except Exception as e:
            print(f"{plan.plan_id:<34} FAILED: {type(e).__name__}: {e}")

    rows.sort(key=lambda r: r["judge_rae"])
    print("\n=== RANKED BY SET-1 JUDGE RAE (lower is better) ===")
    print(f"{'rank':>4} {'plan':<34} {'judge':>7} {'scaffCV':>8} {'gap':>7} {'group':>9}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>4} {r['plan_id']:<34} {r['judge_rae']:7.4f} {r['scaffold_cv_rae']:8.4f} "
              f"{r['gap']:+7.4f} {r['group']:>9}")

    # Calibration signal: how well does scaffold-CV rank models vs the judge?
    rho = spearman([r["scaffold_cv_rae"] for r in rows], [r["judge_rae"] for r in rows])
    print(f"\nScaffold-CV vs judge rank correlation (Spearman): {rho:.3f}  "
          f"(1.0 = scaffold-CV ranks models exactly like the judge)")
    print(f"Mean gap (judge - scaffoldCV): {np.mean([r['gap'] for r in rows]):+.4f}")

    report = {"n_plans": len(rows), "ranked": rows, "spearman_scaffold_vs_judge": rho,
              "mean_gap": float(np.mean([r["gap"] for r in rows])),
              "broad_ensemble_reference_judge_rae": 0.633}

    # Stack ALL successful bases (broad OOF) and judge the ensemble on Set 1.
    if not args.no_ensemble and len(rows) >= 2:
        ok_dirs = [plans_dir / r["plan_id"] for r in rows]
        agg = aggregate(ok_dirs, args.run_dir)
        ens_judge = judge_csv(args.run_dir / "ensemble" / "test_predictions.csv")
        report["ensemble"] = {"method": agg["best_method"], "scaffold_cv_rae": agg["ensemble_rae"],
                              "judge_rae": ens_judge["rae"], "judge_mae": ens_judge["mae"],
                              "judge_r2": ens_judge["r2"]}
        print(f"\nEnsemble of all {len(rows)} bases ({agg['best_method']}): "
              f"scaffoldCV={agg['ensemble_rae']:.4f}  JUDGE={ens_judge['rae']:.4f}  "
              f"(broad ensemble reference was 0.633)")

    (args.run_dir / "judge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nArtifacts: {args.run_dir}/judge_report.json, {args.run_dir}/plans/<id>/")


if __name__ == "__main__":
    main()
