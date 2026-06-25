"""Decide which new modules to keep — by their effect on the *ensemble* judge RAE.

As single models the new modules all score worse than the existing GBDT bases, but
the submission is an ensemble, where a model earns its place through *diversity*,
not standalone score. So the real keep/drop test is: does adding a module to the
strong existing base pool lower the **Set-1 judge** RAE of the stacked ensemble?

We stack on the frozen scaffold-CV OOF (every base shares folds.json, so they're
stackable), apply the combiner to each base's 513 test predictions, and judge the
result on Set 1. Baseline = the existing m1_menu bases; then add mlp / molformer.

Usage:
    uv run python -m scripts.ensemble_compare
"""

from __future__ import annotations

import json
from pathlib import Path

from src.aggregator import aggregate
from src.analog_judge import judge_csv

M1 = Path("outputs/m1_menu/plans")
JUDGE = Path("outputs/judge/plans")
OUT = Path("outputs/ensemble_compare")


def m1_strong_bases() -> list[Path]:
    """Existing m1_menu bases with OOF, minus the known-bad ones: elastic_net is
    catastrophic on analogs (RAE>1), and the tune_* dirs are redundant re-runs."""
    dirs = []
    for d in sorted(M1.iterdir()):
        if not (d / "oof_predictions.csv").exists():
            continue
        if "elastic_net" in d.name or d.name.startswith("tune_"):
            continue
        dirs.append(d)
    return dirs


def judged_ensemble(plan_dirs: list[Path], tag: str) -> dict:
    rep = aggregate(plan_dirs, OUT / tag)
    j = judge_csv(OUT / tag / "ensemble" / "test_predictions.csv")
    return {"tag": tag, "n_bases": len(plan_dirs), "method": rep["best_method"],
            "scaffold_cv_rae": rep["ensemble_rae"], "judge_rae": j["rae"],
            "judge_mae": j["mae"], "judge_r2": j["r2"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = m1_strong_bases()
    mlp = [JUDGE / p for p in ["mlp__desc", "mlp__cb", "mlp__fusion_cb", "mlp__molformer"]]
    mf = [JUDGE / p for p in ["mf__ridge", "mf__lightgbm", "mf__xgboost",
                              "mf__fusion_lightgbm", "mf__fusion_ridge"]]
    mlp = [p for p in mlp if (p / "oof_predictions.csv").exists()]
    mf = [p for p in mf if (p / "oof_predictions.csv").exists()]

    combos = {
        "baseline": base,
        "baseline+mlp": base + mlp,
        "baseline+molformer": base + mf,
        "baseline+mlp+molformer": base + mlp + mf,
    }

    print(f"Strong m1 bases: {len(base)} | mlp add-ons: {len(mlp)} | molformer add-ons: {len(mf)}\n")
    print(f"{'combo':<24} {'bases':>5} {'judge':>7} {'scaffCV':>8}  {'vs baseline':>12}")
    results, base_judge = [], None
    for tag, dirs in combos.items():
        r = judged_ensemble(dirs, tag)
        if tag == "baseline":
            base_judge = r["judge_rae"]
        delta = r["judge_rae"] - base_judge
        results.append({**r, "delta_vs_baseline": delta})
        print(f"{tag:<24} {r['n_bases']:>5} {r['judge_rae']:7.4f} {r['scaffold_cv_rae']:8.4f}  "
              f"{delta:+12.4f}")

    print("\nKeep a module only if its combo's judge RAE is BELOW baseline "
          f"({base_judge:.4f}). Interim LB Top-5 = 0.538 (a moving reference).")
    (OUT / "compare_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Artifacts: {OUT}/compare_report.json")


if __name__ == "__main__":
    main()
