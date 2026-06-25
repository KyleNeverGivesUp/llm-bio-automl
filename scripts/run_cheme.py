"""Does the CheMeleon graph foundation model help (the leaders' main lever, PRD §7.1)?

Trains CheMeleon-based bases on the calibrated folds, judges each on Set 1, then
stacks them onto the frozen 39-base pool (`outputs/final/plans`, judge 0.6266) and
re-judges the ensemble. If CheMeleon is the edge it was for the leaders, both the
single models and the ensemble should improve.

Usage:  uv run python -m scripts.run_cheme
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from pathlib import Path

import pandas as pd

from src.aggregator import aggregate
from src.analog_judge import judge_csv
from src.cv_runner import run_plan_cv
from src.schemas import FoldSpec, MenuPlan

DATA = Path("data/pxr_activity")
CACHE = Path("data/featurizer_cache")
RUN_DIR = Path("outputs/cheme")
FOLDS = DATA / "folds_calibrated.json"
FINAL_PLANS = Path("outputs/final/plans")

CHEME = {}
FUSE = {"components": ["rdkit_descriptors", "chemeleon_embedding"]}
BASES = [
    ("cheme_ridge", "chemeleon_embedding", "ridge", {"scale": True}),
    ("cheme_lightgbm", "chemeleon_embedding", "lightgbm", {}),
    ("cheme_xgboost", "chemeleon_embedding", "xgboost", {}),
    ("cheme_catboost", "chemeleon_embedding", "catboost", {}),
    ("fuse_cheme_lightgbm", "fusion", "lightgbm", FUSE),
    ("fuse_cheme_ridge", "fusion", "ridge", {**FUSE, "scale": True}),
]


def main() -> None:
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    folds = FoldSpec.from_json(FOLDS)
    plans_dir = RUN_DIR / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    print("=== CheMeleon single models (judged on Set 1) ===")
    cheme_dirs = []
    for pid, feat, model, params in BASES:
        plan = MenuPlan(plan_id=pid, name=pid, featurizer=feat, model=model,
                        params=dict(params), seeds=[42, 1])
        m = run_plan_cv(plan, train_df, test_df, folds, out_dir=plans_dir / pid,
                        cache_dir=CACHE, refit_full=True)
        j = judge_csv(plans_dir / pid / "test_predictions.csv")["rae"]
        cheme_dirs.append(plans_dir / pid)
        print(f"  {pid:<22} judge {j:.4f}  (scaffCV {m['score']:.4f})")

    # Frozen strong pool (the 0.6266 ensemble) — exclude elastic_net/molformer-junk.
    pool = [d for d in sorted(FINAL_PLANS.iterdir())
            if (d / "oof_predictions.csv").exists() and "elastic_net" not in d.name]

    base_rep = aggregate(pool, RUN_DIR / "baseline")
    base_j = judge_csv(RUN_DIR / "baseline" / "ensemble" / "test_predictions.csv")["rae"]
    full_rep = aggregate(pool + cheme_dirs, RUN_DIR / "with_cheme")
    full_j = judge_csv(RUN_DIR / "with_cheme" / "ensemble" / "test_predictions.csv")["rae"]

    print("\n=== ensemble (judge RAE) ===")
    print(f"  frozen pool ({len(pool)} bases)          : {base_j:.4f}")
    print(f"  + CheMeleon ({len(cheme_dirs)} bases)        : {full_j:.4f}  ({full_j-base_j:+.4f})")
    print(f"\n{'HELPS' if full_j < base_j-1e-4 else 'no help / hurts'} — leaders reached ~0.54 with CheMeleon as the core.")
    report = {"cheme_singles": {p.name: judge_csv(p/'test_predictions.csv')['rae'] for p in cheme_dirs},
              "baseline_ensemble": base_j, "with_cheme_ensemble": full_j, "delta": full_j-base_j,
              "n_pool": len(pool), "n_cheme": len(cheme_dirs)}
    (RUN_DIR / "cheme_report.json").write_text(json.dumps(report, indent=2))
    print(f"Artifacts: {RUN_DIR}/cheme_report.json")


if __name__ == "__main__":
    main()
