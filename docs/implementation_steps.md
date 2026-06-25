# Implementation Steps

This document is the execution checklist for building the MVP.

The target architecture is `7 agents`, but the implementation order is still a `single-manager MVP`.

That means:

- We design modules around the future 7-agent split
- We implement one component at a time
- We run everything first through one orchestrator
- We do not build the full multi-agent runtime first

## Target 7-Agent Architecture

The intended future agents are:

1. `Manager`
2. `Setup`
3. `Retrieval`
4. `Designer`
5. `Coder`
6. `Tuner`
7. `Exporter/Aggregator`

For the MVP, these map to local modules:

- `Manager` -> `manager.py`
- `Setup` -> `src/schemas.py` and `src/data_utils.py`
- `Retrieval` -> `src/retrieval.py`
- `Designer` -> `src/planner.py`
- `Coder` -> `src/runner.py`
- `Tuner` -> later `src/tuner.py` or a tuning section split out from runner logic
- `Exporter/Aggregator` -> `src/selector.py` and `src/exporter.py`

## Working Principle

Follow these rules:

1. Build one component at a time
2. Make each component produce explicit files
3. Make the next component consume those files directly
4. Delay real multi-agent execution until the single-manager path is stable

## Step 1: Create the Minimal Project Layout

Create these directories first:

- `src/`
- `data/pxr_activity/`
- `outputs/`
- `registry/`

Define one `run_id` format, for example:

- `pxr_activity_YYYYMMDD_HHMMSS`

Target result:

- Every run writes only to `outputs/<run_id>/`

Do this manually first. Do not automate too early.

## Step 2: Implement the Setup Agent Contract

Start with the future `Setup` agent responsibilities.

Create `src/schemas.py` and define:

- `TaskSpec`
- `PlanSpec`
- `RunState`

### `TaskSpec` minimum fields

- `challenge_name`
- `track`
- `task_type`
- `target_column`
- `primary_metric`
- `submission_columns`
- `data_dir`

### `PlanSpec` minimum fields

- `plan_id`
- `name`
- `feature_type`
- `model_type`
- `params`
- `notes`

### `RunState` minimum fields

- `run_id`
- `task_spec_path`
- `current_stage`
- `plan_paths`
- `result_paths`
- `best_plan_path`

After that, verify that these classes can be imported and instantiated cleanly.

## Step 3: Implement the Data Validation Part of Setup

Create `src/data_utils.py`.

Only do two things here:

1. Load data
2. Validate data

Implement at least:

- `load_activity_train(data_dir)`
- `load_activity_test(data_dir)`
- `validate_activity_dataset(data_dir)`
- `write_dataset_report(report, output_path)`

Validation should check:

- required files exist
- train file is readable
- test file is readable
- required columns exist
- row counts are greater than zero
- `SMILES` is not empty
- `Molecule Name` is not empty

Expected output:

- `outputs/<run_id>/dataset_report.json`

## Step 4: Implement a Fixed TaskSpec Builder

Do not parse natural language yet.

Write one function such as:

- `build_pxr_activity_task_spec(data_dir)`

Hardcode:

- challenge = `openadmet/pxr-challenge`
- track = `activity`
- task type = `small_molecule_regression`
- target = `pEC50`
- metric = `RAE`
- submission columns = `["SMILES", "Molecule Name", "pEC50"]`

Write the result to:

- `outputs/<run_id>/task_spec.json`

At this point you should have:

- `task_spec.json`
- `dataset_report.json`

## Step 5: Implement the Designer as a Template Planner

Create `src/planner.py`.

This is the future `Designer` agent, but for MVP it is fully template-based.

Return exactly 3 plans:

1. `morgan_ridge`
2. `morgan_gbdt`
3. `rdkit_rf`

Each plan should define:

- feature type
- model type
- rough params
- output directory

Implement:

- `build_mvp_plans(task_spec, run_id)`
- `write_plans(plans, output_path)`

Expected output:

- `outputs/<run_id>/design_plans.json`

## Step 6: Implement the Coder Agent as the Experiment Runner

Create `src/runner.py`.

This is the future `Coder` agent in MVP form.

Start small. Support:

- fingerprint features
- descriptor features later if needed
- 2 or 3 simple models

Recommended rollout:

Version 1:

- Morgan fingerprint
- Ridge
- RandomForest

Version 2:

- RDKit descriptors
- GBDT or CatBoost

Suggested functions:

- `featurize_morgan(smiles_list)`
- `featurize_rdkit_descriptors(smiles_list)`
- `run_plan(plan, train_df, test_df, output_dir)`
- `save_plan_metrics(metrics, output_path)`

Minimum outputs per plan:

- `metrics.json`
- `oof_predictions.csv`
- `test_predictions.csv`

Do not optimize too early. The goal is a stable artifact flow.

## Step 7: Add a Separate Tuner Component

Because the target architecture includes a dedicated `Tuner`, reserve that responsibility explicitly.

For the first MVP pass, `Tuner` can be minimal.

Create `src/tuner.py` after the baseline runner works.

Its job is:

- read the best baseline result
- generate a small set of new parameter trials
- rerun training using those new configs
- compare tuned vs untuned results

Suggested functions:

- `load_best_plan(run_dir)`
- `build_tuning_trials(best_plan, metrics)`
- `run_tuning_trials(...)`
- `write_tuning_summary(...)`

For the first version, keep tuning tiny:

- Ridge alpha sweep
- RandomForest `n_estimators` and `max_depth`
- one small GBDT search if applicable

Expected outputs:

- `outputs/<run_id>/tuning_trials.json`
- `outputs/<run_id>/tuning_summary.json`
- tuned plan result files

Important:

- `Tuner` decides what new configs to try
- `Coder/Runner` still executes the actual training

## Step 8: Implement Selector Logic

Create `src/selector.py`.

This component reads all plan results and picks the best one.

Implement:

- `load_all_metrics(run_dir)`
- `select_best_plan(metrics_list, primary_metric)`
- `write_leaderboard(metrics_list, output_path)`

Expected outputs:

- `leaderboard.json`
- `best_plan.json`

If tuned results exist, they should be included in ranking.

## Step 9: Implement Exporter/Aggregator

Create `src/exporter.py`.

This is the future `Exporter/Aggregator` agent.

Responsibilities:

1. Read the best plan's test predictions
2. Combine them with test metadata
3. Produce the final submission format
4. Write a final report

Implement:

- `build_submission(test_df, pred_df)`
- `write_submission(submission_df, output_path)`
- `write_final_report(report, output_path)`

Expected outputs:

- `submission.csv`
- `final_report.json`

Before writing, verify:

- exact column names
- row count matches test set
- no null values in required columns

## Step 10: Wire the Single Manager MVP

Only after the components above are stable, update `manager.py`.

This is the future `Manager` agent, but the MVP version is just a sequential orchestrator.

Execution order:

1. create `run_id`
2. build `TaskSpec`
3. validate dataset
4. build plans
5. run baseline plans
6. select current best
7. run tuning on the best plan
8. select best overall result
9. export submission

You only need one minimal entrypoint, for example:

```python
def run_pxr_activity_mvp(data_dir: str) -> None:
    ...
```

Do not add full resume logic or complex CLI yet.

## Step 11: Add Retrieval Later

The target architecture includes a `Retrieval` agent, but it is not on the MVP critical path.

After the main execution chain is stable, create `src/retrieval.py`.

Its job will be:

- load the skill registry
- find relevant model skills if any exist
- fall back to chemistry baselines otherwise

Why later:

- your current bottleneck is execution, not retrieval
- retrieval only becomes useful once the baseline pipeline already works

## Self-Check After Every Step

After each step, verify:

1. Did it create explicit output files?
2. Can the next component consume those files directly?
3. If you rerun it, do outputs stay organized under one `run_id`?

## Recommended Build Order

Follow this order strictly:

1. directories and `run_id`
2. `schemas.py`
3. `data_utils.py`
4. fixed `TaskSpec`
5. `planner.py`
6. `runner.py`
7. `selector.py`
8. `exporter.py`
9. `manager.py`
10. `tuner.py`
11. `retrieval.py`

If you want to reserve the `Tuner` interface earlier, do that in design, but do not block baseline execution on tuning.

## How We Work Together

You write the code.

I will:

- tell you exactly what to implement next
- review what you wrote
- tighten the interface contracts
- help debug issues
- give the smallest viable correction when you get stuck

Best working loop:

1. you finish one step
2. you show me the relevant file
3. I review it and tell you what to fix before the next step
