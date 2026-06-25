"""M2 end-to-end: stack/blend the menu's base learners into an honest ensemble.

Reads every plan under ``<run-dir>/plans/`` that produced OOF predictions, runs
the Aggregator's cross-validated combiners, and reports whether the ensemble
beats the best single model by the M2 target of >=3%.

Usage:
    uv run python -m scripts.run_ensemble
    uv run python -m scripts.run_ensemble --run-dir outputs/m1_menu --top-k 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.aggregator import aggregate


def discover_plan_dirs(run_dir: Path, top_k: int | None) -> list[Path]:
    """Return plan dirs that have OOF, optionally limited to the top-k by OOF RAE
    (read from the leaderboard if present — fewer, stronger, more diverse bases
    often stack better than throwing everything in)."""
    plans_root = run_dir / "plans"
    have_oof = {p.name for p in plans_root.iterdir() if (p / "oof_predictions.csv").exists()}

    lb_path = run_dir / "leaderboard.json"
    if lb_path.exists():
        lb = json.loads(lb_path.read_text())
        ordered = [r["plan_id"] for r in lb if r.get("status") == "ok" and r["plan_id"] in have_oof]
    else:
        ordered = sorted(have_oof)

    if top_k:
        ordered = ordered[:top_k]
    return [plans_root / name for name in ordered]


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate base learners into an ensemble.")
    ap.add_argument("--run-dir", type=Path, default=Path("outputs/m1_menu"))
    ap.add_argument("--top-k", type=int, default=None, help="use only the top-k base models")
    args = ap.parse_args()

    plan_dirs = discover_plan_dirs(args.run_dir, args.top_k)
    if len(plan_dirs) < 2:
        raise SystemExit(f"Need >=2 base plans with OOF; found {len(plan_dirs)} in {args.run_dir}/plans")

    print(f"Base models ({len(plan_dirs)}): {[p.name for p in plan_dirs]}\n")
    report = aggregate(plan_dirs, args.run_dir)

    print("=== ENSEMBLE REPORT (scaffold-CV OOF RAE, lower is better) ===")
    print(f"Best single model : {report['best_single']['name']}  "
          f"RAE={report['best_single']['rae']:.4f}")
    print("Combiners:")
    for m, s in report["methods"].items():
        mark = "  <-- best" if m == report["best_method"] else ""
        print(f"  {m:<6} RAE={s['rae']:.4f}  MAE={s['mae']:.4f}  R²={s['r2']:.3f}{mark}")
    imp = report["improvement_vs_best_single"] * 100
    verdict = "YES" if report["beats_best_single_by_3pct"] else "no"
    print(f"\nEnsemble ({report['best_method']}) RAE={report['ensemble_rae']:.4f}  "
          f"| improvement vs best single: {imp:+.2f}%  | beats by >=3%: {verdict}")
    if isinstance(report["final_weights"], dict):
        print("Final blend weights:")
        for n, w in sorted(report["final_weights"].items(), key=lambda kv: -kv[1]):
            if abs(w) > 1e-6:
                print(f"  {w:6.3f}  {n}")
    print(f"\nArtifacts: {args.run_dir}/ensemble_report.json, "
          f"{args.run_dir}/ensemble/test_predictions.csv")


if __name__ == "__main__":
    main()
