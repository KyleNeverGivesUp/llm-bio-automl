"""Run a fine-tune PLAN end-to-end — the C step that makes the pipeline actually fine-tune.

This is the "tuner" wired up: take a fine-tune plan (what the LLM designer would propose),
build the GPU command (B), LAUNCH it on the A5000 (C), collect the OOF/test into a plan dir (D),
then optionally stack with other members + judge. Run this ON DSMLP in a GPU pod.

  # one plan, end-to-end (actually trains on GPU, ~1-2h):
  python scripts/run_finetune_plan.py --backbone chemeleon --epochs 50
  python scripts/run_finetune_plan.py --backbone unimol --epochs 15

  # skip training, just collect an already-finished run + judge (fast, for wiring checks):
  python scripts/run_finetune_plan.py --backbone chemeleon --collect-only --out-dir ~/cheme_mt5_out

Stacks against `predictions/` if those exist, so you see the member's marginal effect on the judge.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analog_judge import judge_csv                         # noqa: E402
from src.aggregator import aggregate                            # noqa: E402
from src.finetune_runner import FineTunePlan, build_command, collect_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a fine-tune plan end-to-end (build -> GPU train -> collect -> judge).")
    ap.add_argument("--backbone", required=True, choices=["chemeleon", "unimol"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/pxr_activity")
    ap.add_argument("--out-dir", type=Path, default=None, help="where the template writes its CSVs (default: /tmp/<plan_id>)")
    ap.add_argument("--collect-only", action="store_true", help="skip GPU training; collect an existing run")
    args = ap.parse_args()

    plan = FineTunePlan(backbone=args.backbone, epochs=args.epochs, tta=args.tta, label=args.label)
    out_dir = args.out_dir or Path("/tmp") / plan.plan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plan] {plan.plan_id}: backbone={plan.backbone} epochs={plan.epochs} tta={plan.tta}")

    # B + C: build the GPU command and (unless collect-only) actually run the training
    cmd = build_command(plan, repo_dir=REPO, data_dir=args.data_dir, out_dir=out_dir)
    print(f"[coder] command: {' '.join(cmd)}")
    if not args.collect_only:
        print("[tuner] launching GPU training...", flush=True)
        subprocess.run(cmd, check=True)

    # D: collect into an aggregator plan dir
    plan_dir = collect_results(
        plan, out_dir=out_dir, plans_root=REPO / "outputs/finetune_plans",
        folds_json=args.data_dir / "folds_calibrated.json", train_csv=args.data_dir / "train.csv",
    )
    print(f"[collect] -> {plan_dir}")
    print(f"[judge] {plan.plan_id} single: RAE {judge_csv(plan_dir / 'test_predictions.csv')['rae']:.4f}")

    # optional: stack with whatever other fine-tune members are already collected, judge the ensemble
    members = [d for d in (REPO / "outputs/finetune_plans").iterdir() if (d / "oof_predictions.csv").exists()]
    if len(members) > 1:
        ens = aggregate(members, REPO / "outputs/finetune_ens")
        rae = judge_csv(REPO / "outputs/finetune_ens/ensemble/test_predictions.csv")["rae"]
        print(f"[judge] STACK of {len(members)} fine-tune members: RAE {rae:.4f}")


if __name__ == "__main__":
    main()
