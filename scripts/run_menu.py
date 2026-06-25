"""M1 end-to-end: run a menu of modeling recipes through scaffold CV and rank them.

This is the first place the project produces an **honest local score** — every
plan is evaluated on the frozen scaffold folds (unseen chemistry), emits OOF
predictions, and lands on a leaderboard sorted by RAE (lower is better).

Usage:
    uv run python -m scripts.run_menu                      # cheap matrix (fingerprints/descriptors)
    uv run python -m scripts.run_menu --with-embedding     # also run ChemBERTa embeddings (slow, cached)
    uv run python -m scripts.run_menu --featurizers morgan --models ridge xgboost
"""

from __future__ import annotations

import os

# Must be set BEFORE numpy/torch load: multiple OpenMP runtimes (torch's libomp +
# numpy/sklearn's) in one process deadlock on macOS — the ChemBERTa forward pass
# hangs indefinitely at 0% CPU. Pinning to one OMP thread avoids the oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path

import pandas as pd

from src.cv_runner import run_plan_cv
from src.menu_config import (
    ALL_MODELS,
    CHEAP_REPS,
    EMBED_REPS,
    MODEL_DEFAULTS,
    REPRESENTATIONS,
    SCALED_MODELS,
    SEED_POOL,
    STOCHASTIC_MODELS,
)
from src.schemas import FoldSpec, MenuPlan

DATA_DIR = Path("data/pxr_activity")
DEFAULT_RUN_DIR = Path("outputs/m1_menu")
CACHE_DIR = Path("data/featurizer_cache")


def build_menu(reps: list[str], models: list[str], n_seeds: int = 1) -> list[MenuPlan]:
    plans: list[MenuPlan] = []
    for rep_label in reps:
        spec = REPRESENTATIONS[rep_label]
        for model in models:
            params = dict(spec["params"])               # featurizer params (skill_ref, components, n_bits)
            params.update(MODEL_DEFAULTS.get(model, {}))  # then model hyperparameters
            # Scale-sensitive models (linear + MLP): standardize dense reps, NOT
            # binary fingerprints (models._maybe_scale).
            if model in SCALED_MODELS:
                params["scale"] = not spec["binary"]
            # Multi-seed only helps stochastic models; linear ones stay single-seed.
            seeds = SEED_POOL[:n_seeds] if model in STOCHASTIC_MODELS else [42]
            plans.append(
                MenuPlan(
                    plan_id=f"{rep_label}__{model}",
                    name=f"{rep_label} + {model}",
                    featurizer=spec["featurizer"],
                    model=model,
                    params=params,
                    seeds=seeds,
                    skill_ref=spec["params"].get("skill_ref"),
                )
            )
    return plans


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the M1 modeling menu through scaffold CV.")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--with-embedding", action="store_true", help="include embedding/fusion reps (slow)")
    ap.add_argument("--representations", nargs="+", default=None, help=f"override rep list {list(REPRESENTATIONS)}")
    ap.add_argument("--models", nargs="+", default=None, help="override model list")
    ap.add_argument("--seeds", type=int, default=1, help="seeds to average for stochastic models")
    ap.add_argument("--refit-full", action="store_true", help="final test preds via full-data refit")
    args = ap.parse_args()

    reps = args.representations or (CHEAP_REPS + EMBED_REPS if args.with_embedding else CHEAP_REPS)
    models = args.models or ALL_MODELS

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    folds = FoldSpec.from_json(DATA_DIR / "folds.json")

    run_dir = args.run_dir
    plans_dir = run_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)} | "
          f"Folds: {folds.n_folds} (scaffold) | Reps: {reps} | Models: {models}\n")

    leaderboard: list[dict] = []
    menu = build_menu(reps, models, n_seeds=args.seeds)
    for i, plan in enumerate(menu, 1):
        print(f"[{i}/{len(menu)}] {plan.name} ...", end=" ", flush=True)
        try:
            metrics = run_plan_cv(
                plan, train_df, test_df, folds,
                out_dir=plans_dir / plan.plan_id,
                cache_dir=CACHE_DIR,
                refit_full=args.refit_full,
            )
            rae_mean = metrics["cv"]["rae"]["mean"]
            rae_std = metrics["cv"]["rae"]["std"]
            print(f"OOF RAE={metrics['score']:.4f}  (per-fold {rae_mean:.4f}±{rae_std:.4f}, "
                  f"MAE={metrics['oof']['mae']:.4f}, R²={metrics['oof']['r2']:.3f}, "
                  f"{metrics['runtime_sec']}s)")
            leaderboard.append({
                "plan_id": plan.plan_id,
                "name": plan.name,
                "featurizer": plan.featurizer,
                "model": plan.model,
                "rae_oof": metrics["score"],
                "rae_fold_mean": rae_mean,
                "rae_fold_std": rae_std,
                "mae_oof": metrics["oof"]["mae"],
                "r2_oof": metrics["oof"]["r2"],
                "runtime_sec": metrics["runtime_sec"],
                "status": "ok",
            })
        except Exception as e:  # one bad plan must not kill the sweep
            print(f"FAILED: {type(e).__name__}: {e}")
            leaderboard.append({"plan_id": plan.plan_id, "name": plan.name, "status": "error",
                                "error": f"{type(e).__name__}: {e}"})

    ok = [r for r in leaderboard if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["rae_oof"])

    print("\n=== LEADERBOARD (scaffold-CV OOF RAE, lower is better) ===")
    for rank, r in enumerate(ok, 1):
        print(f"{rank:>2}. {r['name']:<34} RAE={r['rae_oof']:.4f}  "
              f"MAE={r['mae_oof']:.4f}  R²={r['r2_oof']:.3f}")
    failed = [r for r in leaderboard if r.get("status") != "ok"]
    for r in failed:
        print(f"  x  {r['name']:<34} {r['error']}")

    (run_dir / "leaderboard.json").write_text(json.dumps(ok + failed, indent=2), encoding="utf-8")
    if ok:
        best = ok[0]
        (run_dir / "best_plan.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"\nFirst honest scaffold-CV score recorded: "
              f"{best['name']} -> RAE {best['rae_oof']:.4f}")
        print(f"Artifacts: {run_dir}/  (leaderboard.json, best_plan.json, plans/<id>/)")


if __name__ == "__main__":
    main()
