"""Run the skill-driven LLM manager (architecture B) end-to-end.

  python scripts/run_skill_manager.py                 # GPU fine-tuning (DSMLP)
  python scripts/run_skill_manager.py --collect-only  # reuse predictions/ (Mac, no GPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent.skill_manager import SkillManager, Ctx   # noqa: E402
from src.schemas import FoldSpec                          # noqa: E402

DATA = Path("data/pxr_activity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    run_dir = Path("outputs/skill_manager")
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(data_dir=DATA, run_dir=run_dir, folds_json=DATA / "folds_calibrated.json",
              train_df=pd.read_csv(DATA / "train.csv"), test_df=pd.read_csv(DATA / "test.csv"),
              folds=FoldSpec.from_json(DATA / "folds_calibrated.json"), collect_only=args.collect_only)
    SkillManager().run(ctx, max_steps=args.max_steps)
    print(f"\n[FINAL] best stacked judge RAE = {ctx.state['best']}")
    (run_dir / "manager_log.json").write_text(json.dumps(ctx.state["log"], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
