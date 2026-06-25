"""Hyperparameter search — the deterministic content-Tuner for Approach 1.

Given a (representation, model) pair, it evaluates a set of hyperparameter trials
on the frozen scaffold folds via ``cv_runner`` and keeps the best by OOF RAE.
Reproducible: trials are sampled from a fixed grid with a fixed seed.

This is the *score-lever* tuner (pure code). The M3 LLM Tuner will later propose
trials on top of this same machinery — it reuses ``run_plan_cv`` exactly as here,
so a tuned plan is scored identically to a menu plan and drops straight into the
ensemble.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.cv_runner import run_plan_cv
from src.menu_config import MODEL_DEFAULTS, REPRESENTATIONS, rep_base_params
from src.schemas import FoldSpec, MenuPlan

# Search space per model. Sampled (not full-grid) to bound cost — each trial is a
# full 5-fold run.
PARAM_GRID: dict[str, dict[str, list]] = {
    "ridge": {"alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]},
    "elastic_net": {"alpha": [0.01, 0.05, 0.1, 0.3, 1.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "random_forest": {
        "n_estimators": [400, 800], "max_depth": [None, 16, 24],
        "max_features": ["sqrt", 0.3, 0.5], "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [400, 700, 1000], "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
        "reg_lambda": [1.0, 3.0, 5.0],
    },
    "lightgbm": {
        "n_estimators": [400, 700, 1000], "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0], "min_child_samples": [10, 20, 40],
        "reg_lambda": [0.0, 1.0, 3.0],
    },
    "catboost": {
        "n_estimators": [600, 1000], "max_depth": [4, 6, 8],
        "learning_rate": [0.02, 0.05, 0.1], "reg_lambda": [1.0, 3.0, 9.0],
    },
}


def _sample(model: str, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_GRID.get(model, {}).items()}


def tune_pair(
    rep_label: str,
    model: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds: FoldSpec,
    out_root: Path,
    cache_dir: Path,
    n_trials: int = 8,
    seeds: tuple[int, ...] = (42,),
    refit_full: bool = True,
    search_seed: int = 0,
) -> list[dict]:
    """Run up to ``n_trials`` hyperparameter trials for one (rep, model) pair.

    Trial 0 is always the current defaults (so tuning can never do worse than the
    menu); the rest are distinct random samples from the grid. Each trial writes a
    ``tune_<rep>__<model>__<i>`` plan dir. Returns trial summaries sorted best-first.
    """
    rng = random.Random(search_seed)
    spec = REPRESENTATIONS[rep_label]
    base = rep_base_params(rep_label, model)

    trials = [dict(MODEL_DEFAULTS.get(model, {}))]      # trial 0 = defaults
    seen = {tuple(sorted(trials[0].items()))}
    attempts = 0
    while len(trials) < n_trials and attempts < n_trials * 30:
        attempts += 1
        cand = _sample(model, rng)
        key = tuple(sorted(cand.items()))
        if key not in seen:
            seen.add(key)
            trials.append(cand)

    results = []
    for i, trial_params in enumerate(trials):
        params = dict(base)
        params.update(trial_params)
        plan = MenuPlan(
            plan_id=f"tune_{rep_label}__{model}__{i}",
            name=f"{rep_label}+{model} tune#{i}",
            featurizer=spec["featurizer"],
            model=model,
            params=params,
            seeds=list(seeds),
            skill_ref=spec["params"].get("skill_ref"),
        )
        m = run_plan_cv(plan, train_df, test_df, folds, out_dir=out_root / plan.plan_id,
                        cache_dir=cache_dir, refit_full=refit_full)
        results.append({"plan_id": plan.plan_id, "rae": m["score"], "params": trial_params})
        print(f"   trial {i}: RAE={m['score']:.4f}  {trial_params or '(defaults)'}")

    results.sort(key=lambda r: r["rae"])
    return results
