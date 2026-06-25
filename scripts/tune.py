"""Tune the top (representation, model) pairs from the current leaderboard.

Each tuned plan is written under ``plans/`` and the leaderboard is rebuilt from
*all* plan metrics (menu + tuned), so the tuned winners drop straight into the
next ``run_ensemble`` call.

    uv run python -m scripts.tune --top-k 3 --n-trials 8
    uv run python -m scripts.tune --pairs rdkit_descriptors:xgboost fusion_desc_cb:catboost
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path

import pandas as pd

from src.menu_config import REPRESENTATIONS
from src.schemas import FoldSpec
from src.tuner_search import tune_pair

DATA_DIR = Path("data/pxr_activity")
CACHE_DIR = Path("data/featurizer_cache")


def rebuild_leaderboard(run_dir: Path) -> list[dict]:
    """Rebuild leaderboard.json from every plan's metrics.json (menu + tuned)."""
    rows = []
    for d in sorted((run_dir / "plans").iterdir()):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        if m.get("status") != "ok":
            continue
        rows.append({
            "plan_id": d.name, "name": m.get("name", d.name),
            "featurizer": m.get("featurizer"), "model": m.get("model"),
            "rae_oof": m.get("score"),
            "mae_oof": m.get("oof", {}).get("mae"), "r2_oof": m.get("oof", {}).get("r2"),
            "status": "ok",
        })
    rows.sort(key=lambda r: r["rae_oof"])
    (run_dir / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _split_pair(plan_id: str) -> tuple[str, str] | None:
    """'fusion_desc_cb__xgboost' -> ('fusion_desc_cb', 'xgboost'); None if not a known rep."""
    if "__" not in plan_id:
        return None
    rep, model = plan_id.rsplit("__", 1)
    return (rep, model) if rep in REPRESENTATIONS else None


def select_pairs(run_dir: Path, top_k: int, explicit: list[str] | None) -> list[tuple[str, str]]:
    if explicit:
        return [tuple(p.split(":", 1)) for p in explicit]
    lb = json.loads((run_dir / "leaderboard.json").read_text())
    pairs: list[tuple[str, str]] = []
    for r in lb:
        if r.get("status") != "ok" or r["plan_id"].startswith("tune_"):
            continue
        pair = _split_pair(r["plan_id"])
        if pair and pair not in pairs:
            pairs.append(pair)
        if len(pairs) >= top_k:
            break
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune the top (rep, model) pairs.")
    ap.add_argument("--run-dir", type=Path, default=Path("outputs/m1_menu"))
    ap.add_argument("--top-k", type=int, default=3, help="how many leaderboard pairs to tune")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1, help="seeds per trial (stochastic models)")
    ap.add_argument("--pairs", nargs="+", default=None, help="explicit rep:model pairs to tune")
    args = ap.parse_args()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    folds = FoldSpec.from_json(DATA_DIR / "folds.json")
    seeds = tuple([42, 1, 2, 7, 13][: args.seeds])

    pairs = select_pairs(args.run_dir, args.top_k, args.pairs)
    print(f"Tuning {len(pairs)} pair(s): {pairs}  ({args.n_trials} trials each, seeds={seeds})\n")

    for rep_label, model in pairs:
        print(f"== {rep_label} + {model} ==")
        results = tune_pair(rep_label, model, train_df, test_df, folds,
                            out_root=args.run_dir / "plans", cache_dir=CACHE_DIR,
                            n_trials=args.n_trials, seeds=seeds)
        best = results[0]
        print(f"   best: RAE={best['rae']:.4f}  {best['params'] or '(defaults)'}\n")

    lb = rebuild_leaderboard(args.run_dir)
    print("=== LEADERBOARD after tuning (top 12) ===")
    for rank, r in enumerate(lb[:12], 1):
        print(f"{rank:>2}. {r['plan_id']:<34} RAE={r['rae_oof']:.4f}  R²={r['r2_oof']:.3f}")
    print(f"\n{len(lb)} plans total. Re-run: uv run python -m scripts.run_ensemble")


if __name__ == "__main__":
    main()
