import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.constants import RUN_ID
from src.runner import run_plan
from src.schemas import PlanSpec
from src.selector import sort_metrics


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: dict, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_best_plan(best_plan_path: str | Path) -> dict:
    return load_json(best_plan_path)


def load_design_plans(design_plans_path: str | Path) -> list[PlanSpec]:
    payload = load_json(design_plans_path)
    return [PlanSpec(**plan_dict) for plan_dict in payload["plans"]]


def find_base_plan(best_plan: dict, design_plans: list[PlanSpec]) -> PlanSpec:
    best_plan_id = best_plan["plan_id"]
    for plan in design_plans:
        if plan.plan_id == best_plan_id:
            return plan
    raise ValueError(f"Could not find base plan for best plan_id={best_plan_id}")


def build_tuning_trials(base_plan: PlanSpec) -> list[PlanSpec]:
    trials: list[PlanSpec] = []

    if base_plan.model_type == "ridge":
        if base_plan.feature_type in {"skill_embedding", "skill_embedding_plus_morgan"}:
            trial_grid = [
                {"alpha": 0.5, "pooling": "cls", "max_length": 256},
                {"alpha": 0.75, "pooling": "cls", "max_length": 256},
                {"alpha": 1.0, "pooling": "cls", "max_length": 192},
                {"alpha": 1.0, "pooling": "cls", "max_length": 256},
                {"alpha": 1.0, "pooling": "cls", "max_length": 384},
                {"alpha": 1.25, "pooling": "cls", "max_length": 256},
                {"alpha": 1.5, "pooling": "cls", "max_length": 256},
                {"alpha": 1.0, "pooling": "mean", "max_length": 256},
            ]
            for idx, updates in enumerate(trial_grid, start=1):
                params = dict(base_plan.params)
                params.update(updates)
                trials.append(
                    PlanSpec(
                        plan_id=f"{base_plan.plan_id}_tune_{idx}",
                        name=f"{base_plan.name} Tune {idx}",
                        feature_type=base_plan.feature_type,
                        model_type=base_plan.model_type,
                        params=params,
                        skill_ref=base_plan.skill_ref,
                        skill_path=base_plan.skill_path,
                        notes=(
                            "Tuning trial: "
                            f"alpha={params['alpha']}, "
                            f"pooling={params['pooling']}, "
                            f"max_length={params['max_length']}"
                        ),
                    )
                )
        else:
            alphas = [0.03, 0.1, 0.3, 1.0, 3.0]
            for idx, alpha in enumerate(alphas, start=1):
                params = dict(base_plan.params)
                params["alpha"] = alpha
                trials.append(
                    PlanSpec(
                        plan_id=f"{base_plan.plan_id}_tune_{idx}",
                        name=f"{base_plan.name} Tune {idx}",
                        feature_type=base_plan.feature_type,
                        model_type=base_plan.model_type,
                        params=params,
                        skill_ref=base_plan.skill_ref,
                        skill_path=base_plan.skill_path,
                        notes=f"Tuning trial: alpha={alpha}",
                    )
                    )
        return trials

    if base_plan.model_type == "elastic_net":
        alphas = [0.01, 0.03, 0.1, 0.3]
        l1_ratios = [0.2, 0.5, 0.8]
        if base_plan.feature_type in {"skill_embedding", "skill_embedding_plus_morgan"}:
            poolings = ["mean", "cls"]
            idx = 1
            for pooling in poolings:
                for alpha in alphas:
                    for l1_ratio in l1_ratios:
                        params = dict(base_plan.params)
                        params["alpha"] = alpha
                        params["l1_ratio"] = l1_ratio
                        params["pooling"] = pooling
                        trials.append(
                            PlanSpec(
                                plan_id=f"{base_plan.plan_id}_tune_{idx}",
                                name=f"{base_plan.name} Tune {idx}",
                                feature_type=base_plan.feature_type,
                                model_type=base_plan.model_type,
                                params=params,
                                skill_ref=base_plan.skill_ref,
                                skill_path=base_plan.skill_path,
                                notes=(
                                    f"Tuning trial: alpha={alpha}, l1_ratio={l1_ratio}, "
                                    f"pooling={pooling}"
                                ),
                            )
                        )
                        idx += 1
        else:
            idx = 1
            for alpha in alphas:
                for l1_ratio in l1_ratios:
                    params = dict(base_plan.params)
                    params["alpha"] = alpha
                    params["l1_ratio"] = l1_ratio
                    trials.append(
                        PlanSpec(
                            plan_id=f"{base_plan.plan_id}_tune_{idx}",
                            name=f"{base_plan.name} Tune {idx}",
                            feature_type=base_plan.feature_type,
                            model_type=base_plan.model_type,
                            params=params,
                            skill_ref=base_plan.skill_ref,
                            skill_path=base_plan.skill_path,
                            notes=f"Tuning trial: alpha={alpha}, l1_ratio={l1_ratio}",
                        )
                    )
                    idx += 1
        return trials

    if base_plan.model_type == "random_forest":
        trial_grid = [
            {"n_estimators": 100, "max_depth": 4},
            {"n_estimators": 300, "max_depth": 6},
            {"n_estimators": 500, "max_depth": 8},
        ]
        for idx, updates in enumerate(trial_grid, start=1):
            params = dict(base_plan.params)
            params.update(updates)
            trials.append(
                PlanSpec(
                    plan_id=f"{base_plan.plan_id}_tune_{idx}",
                    name=f"{base_plan.name} Tune {idx}",
                    feature_type=base_plan.feature_type,
                    model_type=base_plan.model_type,
                    params=params,
                    skill_ref=base_plan.skill_ref,
                    skill_path=base_plan.skill_path,
                    notes=(
                        "Tuning trial: "
                        f"n_estimators={params['n_estimators']}, "
                        f"max_depth={params['max_depth']}"
                    ),
                )
            )
        return trials

    if base_plan.model_type == "xgboost":
        trial_grid = [
            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
            {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03},
        ]
        if base_plan.feature_type in {"skill_embedding", "skill_embedding_plus_morgan"}:
            poolings = ["mean", "cls"]
            idx = 1
            for pooling in poolings:
                for updates in trial_grid:
                    params = dict(base_plan.params)
                    params.update(updates)
                    params["pooling"] = pooling
                    trials.append(
                        PlanSpec(
                            plan_id=f"{base_plan.plan_id}_tune_{idx}",
                            name=f"{base_plan.name} Tune {idx}",
                            feature_type=base_plan.feature_type,
                            model_type=base_plan.model_type,
                            params=params,
                            skill_ref=base_plan.skill_ref,
                            skill_path=base_plan.skill_path,
                            notes=(
                                "Tuning trial: "
                                f"n_estimators={params['n_estimators']}, "
                                f"max_depth={params['max_depth']}, "
                                f"learning_rate={params['learning_rate']}, "
                                f"pooling={pooling}"
                            ),
                        )
                    )
                    idx += 1
        else:
            for idx, updates in enumerate(trial_grid, start=1):
                params = dict(base_plan.params)
                params.update(updates)
                trials.append(
                    PlanSpec(
                        plan_id=f"{base_plan.plan_id}_tune_{idx}",
                        name=f"{base_plan.name} Tune {idx}",
                        feature_type=base_plan.feature_type,
                        model_type=base_plan.model_type,
                        params=params,
                        skill_ref=base_plan.skill_ref,
                        skill_path=base_plan.skill_path,
                        notes=(
                            "Tuning trial: "
                            f"n_estimators={params['n_estimators']}, "
                            f"max_depth={params['max_depth']}, "
                            f"learning_rate={params['learning_rate']}"
                        ),
                    )
                )
        return trials

    raise ValueError(f"Unsupported model_type for tuning: {base_plan.model_type}")


def run_tuning_trials(
    trials: list[PlanSpec],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: str | Path,
) -> list[dict]:
    run_path = Path(run_dir)
    trial_results: list[dict] = []
    total_trials = len(trials)

    for idx, trial in enumerate(trials, start=1):
        output_dir = run_path / "plans" / trial.plan_id
        print(
            (
                f"[tuner] starting trial {idx}/{total_trials}: "
                f"{trial.plan_id} | name={trial.name} | "
                f"feature_type={trial.feature_type} | model_type={trial.model_type} | "
                f"params={json.dumps(trial.params, ensure_ascii=False, sort_keys=True)}"
            ),
            flush=True,
        )
        metrics = run_plan(trial, train_df, test_df, str(output_dir))
        metrics["trial_plan_id"] = trial.plan_id
        metrics["trial_name"] = trial.name
        metrics["trial_params"] = trial.params
        metrics["metrics_path"] = str(output_dir / "metrics.json")
        trial_results.append(metrics)
        print(
            (
                f"[tuner] finished trial {idx}/{total_trials}: "
                f"{trial.plan_id} | mae={metrics['mae']:.6f} | "
                f"rae={metrics['rae']:.6f} | r2={metrics['r2']:.6f} | "
                f"remaining={total_trials - idx}"
            ),
            flush=True,
        )

    return trial_results


def write_tuning_trials(trials: list[PlanSpec], output_path: str | Path) -> None:
    payload = {"trials": [asdict(trial) for trial in trials]}
    write_json(payload, output_path)


def write_tuning_summary(
    base_plan: PlanSpec,
    best_baseline: dict,
    tuned_results: list[dict],
    output_path: str | Path,
    primary_metric: str = "RAE",
) -> None:
    ranked = sort_metrics(tuned_results, primary_metric=primary_metric)
    if not ranked:
        raise ValueError("No tuned results found.")

    best_tuned = ranked[0]

    summary = {
        "base_plan_id": base_plan.plan_id,
        "base_plan_name": base_plan.name,
        "ranking_metric_requested": primary_metric,
        "ranking_metric_used": "rae" if primary_metric == "RAE" else primary_metric,
        "best_baseline": best_baseline,
        "best_tuned": best_tuned,
        "n_tuning_trials": len(tuned_results),
        "all_tuned_results": ranked,
    }
    write_json(summary, output_path)


def run_tuner(
    run_id: str,
    data_dir: str = "data/pxr_activity",
    primary_metric: str = "RAE",
) -> dict:
    run_dir = Path("outputs") / run_id
    best_plan_path = run_dir / "best_plan.json"
    design_plans_path = run_dir / "design_plans.json"

    best_baseline = load_best_plan(best_plan_path)
    design_plans = load_design_plans(design_plans_path)
    base_plan = find_base_plan(best_baseline, design_plans)

    trials = build_tuning_trials(base_plan)
    write_tuning_trials(trials, run_dir / "tuning_trials.json")

    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    test_df = pd.read_csv(Path(data_dir) / "test.csv")

    tuned_results = run_tuning_trials(trials, train_df, test_df, run_dir)
    write_tuning_summary(
        base_plan=base_plan,
        best_baseline=best_baseline,
        tuned_results=tuned_results,
        output_path=run_dir / "tuning_summary.json",
        primary_metric=primary_metric,
    )

    ranked = sort_metrics(tuned_results, primary_metric=primary_metric)
    return ranked[0]


if __name__ == "__main__":
    best_tuned = run_tuner(run_id=RUN_ID)
    print("best tuned trial:")
    print(f"  plan_id: {best_tuned['plan_id']}")
    print(f"  mae: {best_tuned['mae']}")
    print(f"  r2: {best_tuned['r2']}")
