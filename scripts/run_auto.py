"""M3 Manager — the judge-in-the-loop auto-design loop (one command, budgeted, resumable).

This is the automation layer the PRD called for, built on the deterministic pieces
from this phase. Each round:

  Designer proposes N candidates  ->  run on the calibrated folds  ->  judge on
  Set 1  ->  add to the pool  ->  stack the pool  ->  judge the ensemble  ->
  Manager decides continue / stop (budget, patience).

Two safety properties:
  - **Never regress.** The pool is *seeded* with the frozen strong menu (the 0.6266
    submission), and we only ever overwrite `submission.csv` when a round's ensemble
    beats the best seen. A bad LLM proposal scores poorly, gets ~0 stacking weight,
    and is simply ignored — it cannot make the submission worse.
  - **Resumable.** `run_state.json` records every evaluated candidate (+ its judge
    RAE) and the best ensemble; a re-run skips work already on disk.

The objective is the **Set-1 judge**, not scaffold-CV — so the loop optimizes the
number that actually predicts the leaderboard.

Usage:
    uv run python -m scripts.run_auto                          # 3 rounds x 6 candidates
    uv run python -m scripts.run_auto --rounds 4 --candidates 8 --no-llm
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path

import subprocess

import pandas as pd

from src.agent.menu_designer import MenuDesigner
from src.agent.menu_tuner import MenuTuner
from src.agent.finetune_designer import FineTuneDesigner
from src.agent.hf_retrieval import discover_models
from src.aggregator import aggregate
from src.analog_judge import judge_csv
from src.cv_runner import run_plan_cv
from src.finetune_runner import build_command, collect_results
from src.schemas import FoldSpec, MenuPlan

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path("data/pxr_activity")
CACHE_DIR = Path("data/featurizer_cache")
RUN_DIR = Path("outputs/auto")
FOLDS = DATA_DIR / "folds_calibrated.json"
SEED_PLANS = Path("outputs/final/plans")   # the frozen strong menu (calibrated folds) we anchor on
STACK_RAE_CEILING = 0.95                    # exclude near-mean junk (RAE>=this) from the stack


def _judge_dir(plan_dir: Path) -> float | None:
    tp = plan_dir / "test_predictions.csv"
    return judge_csv(tp)["rae"] if tp.exists() else None


def seed_pool(state: dict) -> None:
    """Anchor the pool on the frozen strong menu so the loop starts at ~0.6266."""
    if not SEED_PLANS.exists():
        print("  [seed] outputs/final/plans missing — run `scripts.finalize` first for the anchor.")
        return
    for d in sorted(SEED_PLANS.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        pid = m["plan_id"]
        if pid in state["plans"]:
            continue
        state["plans"][pid] = {"dir": str(d), "judge_rae": _judge_dir(d),
                               "featurizer": m["featurizer"], "model": m["model"], "params": m.get("params", {})}


def stack_and_judge(state: dict, out_dir: Path) -> dict:
    """Stack all non-junk pool members on the calibrated OOF, judge on Set 1."""
    members = [(pid, p) for pid, p in state["plans"].items()
               if p["judge_rae"] is not None and p["judge_rae"] < STACK_RAE_CEILING]
    dirs = [Path(p["dir"]) for _, p in members]
    rep = aggregate(dirs, out_dir)
    j = judge_csv(out_dir / "ensemble" / "test_predictions.csv")
    return {"judge_rae": j["rae"], "cv_rae": rep["ensemble_rae"], "method": rep["best_method"],
            "n_members": len(dirs), "members": [pid for pid, _ in members]}


def _round_dir(best_round) -> str:
    """Map a best-round tag (int | 'seed' | 'tuned') to its ensemble subdir name."""
    return f"round_{best_round}"


def run_candidate(plan, state, train_df, test_df, folds, plans_dir) -> bool:
    """Run one candidate on the calibrated folds, judge it, add to the pool. ✓/✗ isolated."""
    try:
        m = run_plan_cv(plan, train_df, test_df, folds, out_dir=plans_dir / plan.plan_id,
                        cache_dir=CACHE_DIR, refit_full=True)
        jr = _judge_dir(plans_dir / plan.plan_id)
        state["plans"][plan.plan_id] = {"dir": str(plans_dir / plan.plan_id), "judge_rae": jr,
                                        "featurizer": plan.featurizer, "model": plan.model, "params": plan.params}
        print(f"    ✓ {plan.plan_id}: judge {jr:.4f} (scaffCV {m['score']:.4f})")
        return True
    except Exception as e:
        print(f"    ✗ {plan.plan_id}: {type(e).__name__}: {str(e)[:100]}")
        return False


def run_finetune_candidate(plan, state, plans_dir, collect_only) -> bool:
    """Coder routing for FINE-TUNE plans: GPU-train via template (or reuse), judge, add to the SAME pool."""
    try:
        out_dir = (REPO / "predictions") if collect_only else (Path("/tmp") / plan.plan_id)
        if not collect_only:
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(plan, repo_dir=REPO, data_dir=DATA_DIR, out_dir=out_dir)
            print(f"    [GPU] {plan.plan_id}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
        plan_dir = collect_results(plan, out_dir=out_dir, plans_root=plans_dir,
                                   folds_json=DATA_DIR / "folds_calibrated.json", train_csv=DATA_DIR / "train.csv")
        jr = _judge_dir(plan_dir)
        state["plans"][plan.plan_id] = {"dir": str(plan_dir), "judge_rae": jr,
                                        "featurizer": f"finetune:{plan.backbone}", "model": "finetune",
                                        "params": {"epochs": plan.epochs}}
        print(f"    ✓ {plan.plan_id}: judge {jr:.4f}")
        return True
    except Exception as e:
        print(f"    ✗ {plan.plan_id}: {type(e).__name__}: {str(e)[:120]}")
        return False


def finetune_phase(state, plans_dir, collect_only) -> int:
    """LLM (FineTuneDesigner) picks decorrelated backbones to fine-tune; run each into the pool."""
    prior = [{"plan_id": pid, "judge_rae": p["judge_rae"]}
             for pid, p in state["plans"].items() if p.get("judge_rae") is not None]
    plans = FineTuneDesigner().propose(prior_results=prior)
    print(f"  FineTuneDesigner proposed: {[f'{p.backbone}(e{p.epochs})' for p in plans]}")
    return sum(run_finetune_candidate(p, state, plans_dir, collect_only) for p in plans)


def tune_phase(state, tuner, train_df, test_df, folds, plans_dir, top, n_cand, use_llm) -> int:
    """Refine the hyperparameters of the top-``top`` single bases, judged on Set 1."""
    singles = sorted([(pid, p) for pid, p in state["plans"].items() if p["judge_rae"] is not None],
                     key=lambda kv: kv[1]["judge_rae"])[:top]
    n_new = 0
    for pid, base in singles:
        feat, model, bparams = base["featurizer"], base["model"], base["params"]
        prior = [{"params": p["params"], "judge_rae": p["judge_rae"]} for p in state["plans"].values()
                 if p["featurizer"] == feat and p["model"] == model and p["judge_rae"] is not None]
        print(f"  Tuning {feat}+{model} (base judge {base['judge_rae']:.4f})")
        proposals = tuner.propose(feat, model, bparams, prior, n_cand, exclude=set(state["plans"]),
                                  use_llm=use_llm, log_path=RUN_DIR / "llm_logs" / f"tune_{feat}_{model}.json")
        for plan in proposals:
            n_new += run_candidate(plan, state, train_df, test_df, folds, plans_dir)
    return n_new


def write_submission(ens_test_csv: Path, out_csv: Path) -> int:
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    preds = pd.read_csv(ens_test_csv)
    pred_map = dict(zip(preds["Molecule Name"], preds["pEC50"]))
    sub = sample.copy()
    sub["pEC50"] = sub["Molecule Name"].map(pred_map)
    sub = sub[list(sample.columns)]
    assert sub["pEC50"].isna().sum() == 0 and len(sub) == 513
    sub.to_csv(out_csv, index=False)
    return len(sub)


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 judge-in-the-loop auto-design.")
    ap.add_argument("--rounds", type=int, default=3, help="max design rounds (budget)")
    ap.add_argument("--candidates", type=int, default=6, help="candidates proposed per round")
    ap.add_argument("--patience", type=int, default=2, help="stop after this many rounds with no improvement")
    ap.add_argument("--tune-top", type=int, default=0, help="after design rounds, tune the top-K single bases (0=off)")
    ap.add_argument("--tune-candidates", type=int, default=4, help="hyperparameter sets per tuned base")
    ap.add_argument("--no-llm", action="store_true", help="use the deterministic fallback Designer/Tuner only")
    ap.add_argument("--finetune", action="store_true", help="add the LLM-orchestrated fine-tune phase (GPU)")
    ap.add_argument("--ft-collect-only", action="store_true", help="fine-tune phase reuses predictions/ instead of GPU training")
    args = ap.parse_args()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    folds = FoldSpec.from_json(FOLDS)
    plans_dir = RUN_DIR / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    state_path = RUN_DIR / "run_state.json"

    # Resume if a prior state exists, else seed from the frozen menu.
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"Resuming from round {state['round']} with {len(state['plans'])} evaluated candidates.")
    else:
        state = {"round": 0, "plans": {}, "best": None, "history": []}
        seed_pool(state)
        print(f"Seeded pool with {len(state['plans'])} frozen-menu bases.")

    designer = MenuDesigner()

    # Live model retrieval (replaces the static 7-ChemBERTa manifest): discover + log a
    # family-classified candidate pool the designer/finetune-designer draw on.
    try:
        cands = discover_models(top_k=15)
        (RUN_DIR / "candidates_live.json").write_text(
            json.dumps([c.to_dict() for c in cands], indent=2), encoding="utf-8")
        fams = sorted({c.family for c in cands})
        print(f"Live retrieval: {len(cands)} HF candidates, families={fams}")
    except Exception as e:
        print(f"Live retrieval skipped ({e})")

    # Baseline ensemble from the seed pool (the bar to beat).
    if state["best"] is None:
        base = stack_and_judge(state, RUN_DIR / "round_seed")
        state["best"] = base
        state["history"].append({"round": 0, "n_new": 0, "ensemble_judge_rae": base["judge_rae"]})
        print(f"Seed ensemble: judge RAE {base['judge_rae']:.4f} ({base['n_members']} members)\n")
        state_path.write_text(json.dumps(state, indent=2))

    stale = 0
    while state["round"] < args.rounds and stale < args.patience:
        state["round"] += 1
        rnd = state["round"]
        print(f"=== Round {rnd}/{args.rounds} ===")

        prior = [{"plan_id": pid, "featurizer": p["featurizer"], "model": p["model"],
                  "params": p["params"], "judge_rae": p["judge_rae"]} for pid, p in state["plans"].items()]
        proposals = designer.propose(args.candidates, prior, exclude=set(state["plans"]),
                                     use_llm=not args.no_llm, log_path=RUN_DIR / "llm_logs" / f"round_{rnd}.json")
        print(f"  Designer proposed {len(proposals)}: {[p.plan_id for p in proposals]}")

        n_new = sum(run_candidate(plan, state, train_df, test_df, folds, plans_dir) for plan in proposals)

        ens = stack_and_judge(state, RUN_DIR / f"round_{rnd}")
        improved = ens["judge_rae"] < state["best"]["judge_rae"] - 1e-5
        mark = "IMPROVED" if improved else "no gain"
        print(f"  Round {rnd} ensemble: judge {ens['judge_rae']:.4f} vs best "
              f"{state['best']['judge_rae']:.4f}  -> {mark}\n")
        state["history"].append({"round": rnd, "n_new": n_new, "ensemble_judge_rae": ens["judge_rae"]})
        if improved:
            state["best"] = {**ens, "round": rnd}
            stale = 0
        else:
            stale += 1
        state_path.write_text(json.dumps(state, indent=2))

    # Optional tuning phase: refine the best bases' hyperparameters (judged on Set 1).
    if args.tune_top > 0:
        print(f"=== Tuning top {args.tune_top} bases ===")
        n_tuned = tune_phase(state, MenuTuner(), train_df, test_df, folds, plans_dir,
                             args.tune_top, args.tune_candidates, not args.no_llm)
        ens = stack_and_judge(state, RUN_DIR / "round_tuned")
        improved = ens["judge_rae"] < state["best"]["judge_rae"] - 1e-5
        print(f"  Tuned ensemble ({n_tuned} new): judge {ens['judge_rae']:.4f} vs best "
              f"{state['best']['judge_rae']:.4f}  -> {'IMPROVED' if improved else 'no gain'}\n")
        state["history"].append({"round": "tune", "n_new": n_tuned, "ensemble_judge_rae": ens["judge_rae"]})
        if improved:
            state["best"] = {**ens, "round": "tuned"}
        state_path.write_text(json.dumps(state, indent=2))

    # LLM-orchestrated fine-tune phase: designer picks decorrelated backbones to fine-tune;
    # they join the SAME pool and get stacked/judged with the menu members (the rank 84->20 lever).
    if args.finetune:
        print("=== Fine-tune phase (LLM-orchestrated) ===")
        n_ft = finetune_phase(state, plans_dir, args.ft_collect_only)
        ens = stack_and_judge(state, RUN_DIR / "round_finetune")
        improved = ens["judge_rae"] < state["best"]["judge_rae"] - 1e-5
        print(f"  Fine-tune ensemble ({n_ft} new): judge {ens['judge_rae']:.4f} vs best "
              f"{state['best']['judge_rae']:.4f}  -> {'IMPROVED' if improved else 'no gain'}\n")
        state["history"].append({"round": "finetune", "n_new": n_ft, "ensemble_judge_rae": ens["judge_rae"]})
        if improved:
            state["best"] = {**ens, "round": "finetune"}
        state_path.write_text(json.dumps(state, indent=2))

    # Final submission from the best-ever ensemble (re-stack its members for the test preds).
    best_round = state["best"].get("round", "seed")
    best_ens_csv = RUN_DIR / _round_dir(best_round) / "ensemble" / "test_predictions.csv"
    if best_ens_csv.exists():
        n = write_submission(best_ens_csv, RUN_DIR / "submission.csv")
        print(f"BEST ensemble: judge RAE {state['best']['judge_rae']:.4f} "
              f"({state['best']['n_members']} members, round {best_round}) -> {RUN_DIR}/submission.csv ({n} rows)")
    manifest = {"best_judge_rae": state["best"]["judge_rae"], "best_round": best_round,
                "n_candidates_evaluated": len(state["plans"]), "rounds_run": state["round"],
                "history": state["history"], "frozen_menu_anchor": 0.6266, "objective": "set1_judge"}
    (RUN_DIR / "auto_report.json").write_text(json.dumps(manifest, indent=2))
    print(f"Report: {RUN_DIR}/auto_report.json")


if __name__ == "__main__":
    main()
