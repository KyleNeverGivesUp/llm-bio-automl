"""Calibrate the internal CV against the Set-1 judge — close the 0.543->0.633 gap.

The problem: broad scaffold-CV ranks models differently than the analog judge and
reads ~0.09 too low. We can't fix that by touching the judge (it can't join the
competition). Instead we choose a *broad-only* fold design whose per-model ranking
matches the judge's and whose RAE level lands near the judge's — then we select
models with that ruler and let the judge merely confirm.

Protocol (leakage-safe — folds partition ONLY the broad 4,139; Set-1 labels are
never trained on, only used to grade the fold designs at the meta level):
  1. Take a diverse probe set of models (reuse their broad-trained predictions
     from ``scripts.run_judge`` for the fold-independent **judge RAE**).
  2. For each candidate fold design (scaffold + Tanimoto-cluster at several
     cutoffs), recompute each probe's **CV RAE** under those folds.
  3. Score each design by Spearman(CV-rank, judge-rank) and how close its mean CV
     RAE sits to the judge. Pick the best; save its ``folds.json``.

Usage:
    uv run python -m scripts.calibrate_folds                  # after scripts.run_judge
    uv run python -m scripts.calibrate_folds --cutoffs 0.5 0.6 0.7
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analog_judge import judge_csv
from src.cv_runner import run_plan_cv
from src.cv_split import assign_cluster_folds
from src.menu_config import CHEMBERTA, MOLFORMER
from src.schemas import FoldSpec, MenuPlan

DATA_DIR = Path("data/pxr_activity")
JUDGE_DIR = Path("outputs/judge")
CACHE_DIR = Path("data/featurizer_cache")
OUT_DIR = Path("outputs/calibration")
CB = {"skill_ref": CHEMBERTA}
MF = {"skill_ref": MOLFORMER}
FUSION_CB = {"components": ["rdkit_descriptors", "chemberta_embedding"], "skill_ref": CHEMBERTA}

# Probe models (plan_id reused from scripts.run_judge for the judge RAE) chosen to
# span the judge-RAE range and several representations — what we want the ruler to
# rank correctly. Kept to fast models so calibration stays cheap.
# Fast, diverse probes spanning the judge-RAE range (0.70–0.99). Deliberately no
# MLP probe: an MLP trains per fold per design and made the sweep ~5× slower for a
# single mid-range point — the rank signal is already well-determined without it.
PROBES = [
    ("ref__desc_lightgbm", "rdkit_descriptors", "lightgbm", {}),          # judge ~0.697
    ("ref__desc_xgboost", "rdkit_descriptors", "xgboost", {}),            # judge ~0.722
    ("ref__cb_ridge", "chemberta_embedding", "ridge", {**CB, "scale": True}),  # ~0.740
    ("mf__lightgbm", "molformer_embedding", "lightgbm", MF),              # ~0.993 (weak anchor)
]


def spearman(a, b) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cv_rae_under(folds: FoldSpec, plans, train_df, test_df, tag: str) -> list[float]:
    """OOF RAE for each probe under one fold design (no refit — we only need the rank)."""
    out = []
    for pid, feat, model, params in plans:
        m = run_plan_cv(
            MenuPlan(plan_id=f"{tag}__{pid}", name=pid, featurizer=feat, model=model,
                     params=dict(params), seeds=[42], skill_ref=params.get("skill_ref")),
            train_df, test_df, folds, out_dir=OUT_DIR / "plans" / f"{tag}__{pid}",
            cache_dir=CACHE_DIR, refit_full=False)
        out.append(m["score"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate CV folds to the Set-1 judge.")
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7],
                    help="Butina Tanimoto-distance cutoffs to try for cluster folds")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    smiles = train_df["SMILES"].tolist()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Judge RAE (fold-independent) + scaffold CV RAE, both reused from scripts.run_judge.
    judge_rae, scaffold_cv = [], []
    for pid, *_ in PROBES:
        pdir = JUDGE_DIR / "plans" / pid
        if not (pdir / "test_predictions.csv").exists():
            raise SystemExit(f"Missing {pdir} — run `uv run python -m scripts.run_judge` first.")
        judge_rae.append(judge_csv(pdir / "test_predictions.csv")["rae"])
        scaffold_cv.append(json.loads((pdir / "metrics.json").read_text())["score"])

    judge_mean = float(np.mean(judge_rae))
    print(f"Probes ({len(PROBES)}) | judge mean RAE = {judge_mean:.4f}\n")
    print(f"{'design':<16} {'meanCV':>7} {'gap':>7} {'spearman_vs_judge':>18}")

    designs = []
    # Baseline: the existing scaffold design (no recompute needed).
    rho = spearman(scaffold_cv, judge_rae)
    designs.append({"design": "scaffold", "cutoff": None, "mean_cv": float(np.mean(scaffold_cv)),
                    "cv_rae": scaffold_cv, "spearman_vs_judge": rho})
    print(f"{'scaffold':<16} {np.mean(scaffold_cv):7.4f} {np.mean(scaffold_cv)-judge_mean:+7.4f} "
          f"{rho:>18.3f}")

    # Candidate cluster designs.
    for cutoff in args.cutoffs:
        fold_of_row, diag = assign_cluster_folds(smiles, n_folds=args.n_folds, cutoff=cutoff)
        folds = FoldSpec(strategy="cluster", n_folds=args.n_folds, seed=42, fold_of_row=fold_of_row)
        cv = cv_rae_under(folds, PROBES, train_df, test_df, tag=f"clust{cutoff}")
        rho = spearman(cv, judge_rae)
        designs.append({"design": f"cluster@{cutoff}", "cutoff": cutoff, "mean_cv": float(np.mean(cv)),
                        "cv_rae": cv, "spearman_vs_judge": rho, "n_clusters": diag["n_clusters"],
                        "fold_sizes": diag["fold_sizes"], "fold_of_row": fold_of_row})
        print(f"{'cluster@'+str(cutoff):<16} {np.mean(cv):7.4f} {np.mean(cv)-judge_mean:+7.4f} "
              f"{rho:>18.3f}  ({diag['n_clusters']} clusters)")

    # Pick: best rank-agreement first; among ties prefer BALANCED folds (a coarse
    # cutoff can hit the same rank with a few mega-clusters and lumpy folds — that's
    # a worse ruler), then the closest RAE level to the judge.
    def _imbalance(d):
        fs = d.get("fold_sizes")
        return max(fs) / min(fs) if fs else 1.0

    best = max(designs, key=lambda d: (round(d["spearman_vs_judge"], 3),
                                       -round(_imbalance(d), 2),
                                       -abs(d["mean_cv"] - judge_mean)))
    print(f"\nBest design: {best['design']}  "
          f"(spearman={best['spearman_vs_judge']:.3f}, meanCV={best['mean_cv']:.4f} vs judge {judge_mean:.4f})")

    # Persist the winning folds (if it's a cluster design) for the rest of the pipeline.
    if best.get("fold_of_row"):
        from src.cv_split import save_folds
        diag = {"strategy": "cluster", "n_molecules": len(smiles), "n_folds": args.n_folds,
                "seed": 42, "cluster_cutoff": best["cutoff"], "fold_sizes": best["fold_sizes"],
                "calibrated_to": "set1_judge", "spearman_vs_judge": best["spearman_vs_judge"]}
        save_folds(best["fold_of_row"], diag, DATA_DIR / "folds_calibrated.json")
        print(f"Saved winning folds -> {DATA_DIR}/folds_calibrated.json")

    report = {"judge_mean_rae": judge_mean, "probes": [p[0] for p in PROBES],
              "judge_rae": judge_rae, "designs": [{k: v for k, v in d.items() if k != "fold_of_row"}
                                                  for d in designs], "best": best["design"]}
    (OUT_DIR / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Artifacts: {OUT_DIR}/calibration_report.json")


if __name__ == "__main__":
    main()
