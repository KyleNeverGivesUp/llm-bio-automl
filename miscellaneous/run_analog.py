"""Analog-set pipeline — the strategy after CHALLENGE_BRIEF §8.1.

The 513-compound test set is an *analog* set (close analogs of 63 hits, activity
cliffs), a narrower distribution than the broad ~4,139 training set. Broad
scaffold-CV does NOT predict analog-test RAE (we measured 0.543 scaffold-CV vs
0.633 on the real analog labels). So here we:

  1. Use **Analog Set 1 (253 now-public labels)** as the true validation set.
  2. Validate every base model by **analog-CV**: 5-fold over Set 1, where each
     fold trains on (broad 4,139 + the other Set-1 folds) and predicts the held-out
     Set-1 fold. This is the realistic "train on broad+Set1, predict unseen analogs"
     scenario and gives leak-free analog OOF for stacking.
  3. **Stack** the base models' analog OOF (ridge / nnls), selecting weights on the
     analog distribution.
  4. For the **submission**, retrain every base on (broad + ALL of Set 1) and
     predict the 513 test compounds; combine with the stacked weights.

Run:  uv run python -m scripts.run_analog
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.aggregator import NNLSCombiner, RidgeStackCombiner
from src.featurizers import featurize
from src.menu_config import CHEMBERTA
from src.metrics import score_all
from src.models import make_model

DATA = Path("data/pxr_activity")
CACHE = Path("data/featurizer_cache")
RUN_DIR = Path("outputs/analog")
N_FOLDS = 5
SEED = 42

# Diverse base configs (rep featurizer, model, params). Dropped elastic_net — it is
# catastrophic on the analog distribution (RAE > 1.0). "scale" flags linear models
# on dense reps; "binary" reps (morgan/maccs/avalon) are never scaled.
FUSION = {"components": ["rdkit_descriptors", "chemberta_embedding"], "skill_ref": CHEMBERTA}
CONFIGS = [
    ("desc",     "rdkit_descriptors",   {}, "lightgbm", {}),
    ("desc",     "rdkit_descriptors",   {}, "xgboost", {}),
    ("desc",     "rdkit_descriptors",   {}, "catboost", {}),
    ("desc",     "rdkit_descriptors",   {}, "random_forest", {}),
    ("desc",     "rdkit_descriptors",   {}, "ridge", {"scale": True}),
    ("fusion",   "fusion",              FUSION, "xgboost", {}),
    ("fusion",   "fusion",              FUSION, "lightgbm", {}),
    ("fusion",   "fusion",              FUSION, "ridge", {"scale": True}),
    ("cb",       "chemberta_embedding", {"skill_ref": CHEMBERTA}, "ridge", {"scale": True}),
    ("cb",       "chemberta_embedding", {"skill_ref": CHEMBERTA}, "lightgbm", {}),
    ("morgan",   "morgan",              {}, "xgboost", {}),
    ("avalon",   "avalon",              {"n_bits": 1024}, "lightgbm", {}),
    ("maccs",    "maccs",               {}, "xgboost", {}),
]


def _feat_key(featurizer: str, fparams: dict) -> tuple:
    return (featurizer, json.dumps(fparams, sort_keys=True))


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    broad = pd.read_csv(DATA / "train.csv")
    s1 = pd.read_csv(DATA / "phase1_unblinded.csv")
    test = pd.read_csv(DATA / "test.csv")
    yb = broad["pEC50"].to_numpy(float)
    ys = s1["pEC50"].to_numpy(float)

    # Featurize each distinct representation once for broad / Set1 / test.
    feat_cache: dict[tuple, tuple] = {}
    for _, featurizer, fparams, _, _ in CONFIGS:
        k = _feat_key(featurizer, fparams)
        if k not in feat_cache:
            feat_cache[k] = (
                featurize(featurizer, broad["SMILES"].tolist(), fparams, cache_dir=CACHE),
                featurize(featurizer, s1["SMILES"].tolist(), fparams, cache_dir=CACHE),
                featurize(featurizer, test["SMILES"].tolist(), fparams, cache_dir=CACHE),
            )

    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(ys))

    names, OOF, TEST = [], [], []
    print(f"Base models: {len(CONFIGS)} | Set1={len(ys)} | broad={len(yb)} | test={len(test)}\n")
    print("analog-CV RAE per base (fold-in: broad + other Set1 folds -> held-out Set1 fold):")
    for label, featurizer, fparams, model, mparams in CONFIGS:
        Xb, Xs, Xt = feat_cache[_feat_key(featurizer, fparams)]
        params = {**fparams, **mparams}
        # analog OOF on Set1 (leak-free)
        oof = np.zeros(len(ys))
        for tr, va in folds:
            m = make_model(model, params, seed=SEED)
            m.fit(np.vstack([Xb, Xs[tr]]), np.concatenate([yb, ys[tr]]))
            oof[va] = m.predict(Xs[va])
        # test preds: train on broad + ALL Set1
        m_full = make_model(model, params, seed=SEED)
        m_full.fit(np.vstack([Xb, Xs]), np.concatenate([yb, ys]))
        tpred = m_full.predict(Xt)

        nm = f"{label}__{model}"
        names.append(nm); OOF.append(oof); TEST.append(tpred)
        print(f"  {score_all(ys, oof)['rae']:.4f}  {nm}")

    OOF = np.column_stack(OOF); TEST = np.column_stack(TEST)

    # Honest stack: CV the combiner over the same Set1 folds (weights fit on
    # out-of-fold rows, predict the held-out fold) so the ensemble RAE isn't optimistic.
    print("\nensemble (analog-CV stacked OOF RAE):")
    best = None
    for cname, factory in [("nnls", lambda: NNLSCombiner()),
                           ("ridge", lambda: RidgeStackCombiner(alpha=1.0, positive=True))]:
        stacked = np.zeros(len(ys))
        for tr, va in folds:
            stacked[va] = factory().fit(OOF[tr], ys[tr]).predict(OOF[va])
        s = score_all(ys, stacked)
        print(f"  {cname:<5} RAE={s['rae']:.4f}  MAE={s['mae']:.4f}  R²={s['r2']:.3f}")
        if best is None or s["rae"] < best[1]:
            best = (cname, s["rae"], factory)

    # Final ensemble: fit chosen combiner on ALL Set1 OOF, apply to test base preds.
    cname, best_rae, factory = best
    final = factory().fit(OOF, ys)
    ens_test = final.predict(TEST)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    pred_map = dict(zip(test["Molecule Name"], ens_test))
    sub = sample.copy(); sub["pEC50"] = sub["Molecule Name"].map(pred_map)
    sub = sub[list(sample.columns)]
    assert sub["pEC50"].isna().sum() == 0
    sub.to_csv(RUN_DIR / "submission.csv", index=False)

    report = {
        "strategy": "analog_foldin_stack",
        "base_models": names,
        "best_combiner": cname,
        "analog_cv_ensemble_rae": best_rae,
        "n_set1": len(ys), "n_broad": len(yb), "n_test": len(test),
    }
    (RUN_DIR / "analog_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBEST analog-CV ensemble: {cname} RAE={best_rae:.4f}  "
          f"(broad-only ensemble was 0.633 | LB Top-5=0.538)")
    print(f"submission -> {RUN_DIR/'submission.csv'} (trained on broad + all Set1, {len(sub)} rows)")


if __name__ == "__main__":
    main()
