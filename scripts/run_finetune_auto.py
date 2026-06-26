"""End-to-end LLM-orchestrated fine-tuning — the autonomous loop (prof-approved).

The full thing, no hand-typed plans:
  1. FineTuneDesigner (LLM) decides WHICH backbones to fine-tune + epochs (picks decorrelated families).
  2. For each plan: build the GPU command (B) and run it (C) — or --collect-only to reuse finished runs.
  3. Collect each into a plan dir (D), stack, judge.

  python scripts/run_finetune_auto.py                          # LLM designs -> GPU trains -> stack -> judge
  python scripts/run_finetune_auto.py --collect-only           # LLM designs -> reuse predictions/ -> stack (no GPU)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.agent.finetune_designer import FineTuneDesigner          # noqa: E402
from src.aggregator import aggregate                               # noqa: E402
from src.analog_judge import judge_csv                             # noqa: E402
from src.finetune_runner import build_command, collect_results, TEMPLATES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-orchestrated fine-tuning, end-to-end.")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/pxr_activity")
    ap.add_argument("--collect-only", action="store_true", help="reuse existing predictions/ instead of GPU training")
    ap.add_argument("--reuse-dir", type=Path, default=REPO / "predictions", help="where finished OOF/test CSVs live (collect-only)")
    args = ap.parse_args()

    # 1) LLM decides what to fine-tune
    designer = FineTuneDesigner()
    plans = designer.propose(prior_results=[])
    print("[designer] LLM proposed:", ", ".join(f"{p.backbone}(e{p.epochs})" for p in plans))

    plans_root = REPO / "outputs/finetune_auto/plans"
    plan_dirs = []
    for p in plans:
        out_dir = (args.reuse_dir if args.collect_only else Path("/tmp") / p.plan_id)
        if not args.collect_only:
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(p, repo_dir=REPO, data_dir=args.data_dir, out_dir=out_dir)
            print(f"[tuner] {p.plan_id}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)              # C: real GPU training
        # D: collect into a plan dir
        try:
            pd_ = collect_results(p, out_dir=out_dir, plans_root=plans_root,
                                  folds_json=args.data_dir / "folds_calibrated.json",
                                  train_csv=args.data_dir / "train.csv")
            plan_dirs.append(pd_)
            print(f"[judge] {p.plan_id} single: RAE {judge_csv(pd_ / 'test_predictions.csv')['rae']:.4f}")
        except FileNotFoundError as e:
            print(f"[skip] {p.plan_id}: no results to collect ({e})")

    if len(plan_dirs) >= 2:
        aggregate(plan_dirs, REPO / "outputs/finetune_auto/ens")
        rae = judge_csv(REPO / "outputs/finetune_auto/ens/ensemble/test_predictions.csv")["rae"]
        print(f"\n[RESULT] LLM-orchestrated fine-tune STACK ({len(plan_dirs)} members): RAE {rae:.4f}")


if __name__ == "__main__":
    main()
