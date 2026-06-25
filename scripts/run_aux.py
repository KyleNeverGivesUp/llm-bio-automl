"""M5(a): does adding the clean auxiliary HT-chem pEC50 data help the Set-1 judge?

Trains strong base configs with vs. without the auxiliary rows (crude + semi-pure
PXR pEC50, deduped against broad/test/Set-1 — see `data/pxr_activity/aux_train.csv`),
on the calibrated folds, and judges each ensemble on Set 1. Aux rows are ALWAYS in
training, never held out, so OOF + the judge stay on the broad rows. **Set 1 is never
training data** (it's only the judge, and it was deduped out of the aux set).

Usage:
    uv run python -m scripts.run_aux
    uv run python -m scripts.run_aux --aux-weight 0.5
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path

import pandas as pd

from src.aggregator import aggregate
from src.analog_judge import judge_csv
from src.cv_runner import run_plan_cv
from src.menu_config import CHEMBERTA
from src.schemas import FoldSpec, MenuPlan

DATA = Path("data/pxr_activity")
CACHE = Path("data/featurizer_cache")
RUN_DIR = Path("outputs/aux")
FOLDS = DATA / "folds_calibrated.json"
FUSION_CB = {"components": ["rdkit_descriptors", "chemberta_embedding"], "skill_ref": CHEMBERTA}

# A compact strong, diverse base set (the ones that carry the broad ensemble).
BASES = [
    ("desc_lightgbm", "rdkit_descriptors", "lightgbm", {}),
    ("desc_xgboost", "rdkit_descriptors", "xgboost", {}),
    ("desc_catboost", "rdkit_descriptors", "catboost", {}),
    ("fusion_cb_lightgbm", "fusion", "lightgbm", FUSION_CB),
    ("fusion_cb_ridge", "fusion", "ridge", {**FUSION_CB, "scale": True}),
    ("cb_ridge", "chemberta_embedding", "ridge", {"skill_ref": CHEMBERTA, "scale": True}),
    ("morgan_xgboost", "morgan", "xgboost", {}),
]


def _run_set(aux_df, aux_weight, tag, train_df, test_df, folds) -> dict:
    plans_dir = RUN_DIR / tag / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    dirs = []
    for pid, feat, model, params in BASES:
        plan = MenuPlan(plan_id=pid, name=pid, featurizer=feat, model=model,
                        params=dict(params), seeds=[42, 1], skill_ref=params.get("skill_ref"))
        run_plan_cv(plan, train_df, test_df, folds, out_dir=plans_dir / pid, cache_dir=CACHE,
                    refit_full=True, aux_train_df=aux_df, aux_weight=aux_weight)
        dirs.append(plans_dir / pid)
    rep = aggregate(dirs, RUN_DIR / tag)
    j = judge_csv(RUN_DIR / tag / "ensemble" / "test_predictions.csv")
    return {"tag": tag, "ensemble_judge_rae": j["rae"], "ensemble_cv_rae": rep["ensemble_rae"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Auxiliary-data ablation on the Set-1 judge.")
    ap.add_argument("--aux-weight", type=float, default=1.0, help="global weight on aux rows")
    args = ap.parse_args()

    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    aux_df = pd.read_csv(DATA / "aux_train.csv")
    folds = FoldSpec.from_json(FOLDS)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"broad {len(train_df)} | aux {len(aux_df)} (weight {args.aux_weight}) | bases {len(BASES)}\n")

    no_aux = _run_set(None, 1.0, "no_aux", train_df, test_df, folds)
    with_aux = _run_set(aux_df, args.aux_weight, f"with_aux_w{args.aux_weight}", train_df, test_df, folds)

    delta = with_aux["ensemble_judge_rae"] - no_aux["ensemble_judge_rae"]
    print(f"{'setting':<22}{'judge RAE':>10}{'cv RAE':>9}")
    print(f"{'no aux':<22}{no_aux['ensemble_judge_rae']:>10.4f}{no_aux['ensemble_cv_rae']:>9.4f}")
    print(f"{'+ aux ('+str(len(aux_df))+')':<22}{with_aux['ensemble_judge_rae']:>10.4f}{with_aux['ensemble_cv_rae']:>9.4f}")
    print(f"\nDelta (aux - no_aux) on judge: {delta:+.4f}  -> {'HELPS' if delta < -1e-4 else 'no help / hurts'}")
    report = {"no_aux": no_aux, "with_aux": with_aux, "delta_judge": delta,
              "aux_weight": args.aux_weight, "n_aux": len(aux_df)}
    (RUN_DIR / "aux_report.json").write_text(json.dumps(report, indent=2))
    print(f"Artifacts: {RUN_DIR}/aux_report.json")


if __name__ == "__main__":
    main()
