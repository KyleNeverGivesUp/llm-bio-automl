# llm-bio-automl

This repository currently contains two PXR activity modeling pipelines:

- `ML pipeline`: a deterministic baseline pipeline
- `AutoML pipeline`: an LLM-orchestrated pipeline built on top of the deterministic execution layer

The original bio-model skill generation code is still present in the repository, but it is not the main workflow documented here.

## Overview

The project targets the `openadmet/pxr-challenge` activity regression task using the dataset in [data/pxr_activity](/Users/kyle/Projects/bio-model-skills-creator/data/pxr_activity).

The two active versions are:

1. `ML version`
   - Deterministic
   - Reproducible
   - Uses a fixed, supported set of feature/model combinations
   - Good for a stable baseline and submission generation

2. `AutoML version`
   - Uses LLM agents to guide retrieval, design, selection, and tuning
   - Uses the deterministic execution layer for actual model training, scoring, and export
   - Current implementation is constrained AutoML, not fully open-ended autonomous model discovery

## Current Status

As of April 17, 2026:

- The deterministic `ML pipeline` runs end to end.
- The `AutoML pipeline` also runs end to end.
- OpenRouter integration is working.
- The current AutoML system is best described as `LLM-assisted constrained AutoML`.

One validated run produced:

- best plan: `morgan_rf_t3`
- `mae = 0.6723264856692197`
- `r2 = 0.37605752261143266`

Example artifacts:

- Best plan: [best_plan.json](/Users/kyle/Projects/bio-model-skills-creator/outputs/pxr_activity_20260417_191914/best_plan.json)
- Final report: [final_report.json](/Users/kyle/Projects/bio-model-skills-creator/outputs/pxr_activity_20260417_191914/final_report.json)
- Smoke test report: [smoketest_report.json](/Users/kyle/Projects/bio-model-skills-creator/outputs/openrouter_smoketest_20260417_191700/smoketest_report.json)

## Version 1: ML Pipeline

The deterministic ML pipeline lives in [src/pxr_manager.py](/Users/kyle/Projects/bio-model-skills-creator/src/pxr_manager.py).

### What it does

It performs the following stages:

1. Load and validate the PXR dataset
2. Build a task specification
3. Build baseline plans
4. Execute supported baseline models
5. Rank baseline results
6. Tune the current best baseline
7. Rank all runs again
8. Export the final submission

### Supported model combinations

The currently supported execution paths are:

- `morgan_fingerprint + ridge`
- `morgan_fingerprint + random_forest`

There are additional plan definitions in the planner, but the runner currently supports only the combinations above.

### Main files

- [src/pxr_manager.py](/Users/kyle/Projects/bio-model-skills-creator/src/pxr_manager.py)
- [src/planner.py](/Users/kyle/Projects/bio-model-skills-creator/src/planner.py)
- [src/runner.py](/Users/kyle/Projects/bio-model-skills-creator/src/runner.py)
- [src/selector.py](/Users/kyle/Projects/bio-model-skills-creator/src/selector.py)
- [src/tuner.py](/Users/kyle/Projects/bio-model-skills-creator/src/tuner.py)
- [src/exporter.py](/Users/kyle/Projects/bio-model-skills-creator/src/exporter.py)

### How to run

```bash
python -m src.pxr_manager
```

### Outputs

Each run writes artifacts to:

```bash
outputs/<run_id>/
```

Typical files include:

- `run_state.json`
- `dataset_report.json`
- `task_spec.json`
- `design_plans.json`
- `leaderboard.json`
- `best_plan.json`
- `submission.csv`
- `final_report.json`

## Version 2: AutoML Pipeline

The AutoML pipeline lives in [src/agent/](/Users/kyle/Projects/bio-model-skills-creator/src/agent).

Its main entrypoint is [src/agent/run_agent_pipeline.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/run_agent_pipeline.py).

### What it does

The AutoML version adds an LLM orchestration layer on top of the deterministic modeling layer.

The current agent manager coordinates:

1. setup
2. retrieval
3. design
4. execution
5. baseline selection
6. tuning
7. final selection
8. export

### What “AutoML” means here

This is not yet a fully open-ended autonomous system.

Current behavior:

- The LLM participates in design, selection, and tuning decisions
- The deterministic execution layer still owns model training and evaluation
- The search space is constrained by the currently supported plan schema and runner capabilities

So the most accurate description is:

- `constrained AutoML`
- `LLM-assisted AutoML`

not:

- fully autonomous unrestricted AutoML

### Main files

- [src/agent/manager_agent.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/manager_agent.py)
- [src/agent/setup_agent.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/setup_agent.py)
- [src/agent/retrieval_agent.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/retrieval_agent.py)
- [src/agent/designer_agent.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/designer_agent.py)
- [src/agent/coder_agent.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/coder_agent.py)
- [src/agent/selector_agent.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/selector_agent.py)
- [src/agent/tuner_agent.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/tuner_agent.py)
- [src/agent/exporter_agent.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/exporter_agent.py)
- [src/agent/base.py](/Users/kyle/Projects/bio-model-skills-creator/src/agent/base.py)

### LLM configuration

The agent pipeline reads OpenRouter configuration from `.env`.

Example variables:

```env
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
OPENROUTER_MODEL=openrouter/free
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=llm-bio-automl
OPENROUTER_MAX_TOKENS=2000
OPENROUTER_TEMPERATURE=0
```

A template is available at [.env.agent.example](/Users/kyle/Projects/bio-model-skills-creator/.env.agent.example).

### How to run

First, verify the LLM connection:

```bash
python -m src.agent.test_openrouter
```

Then run the AutoML pipeline:

```bash
python -m src.agent.run_agent_pipeline
```

### Outputs

Like the deterministic version, each run writes to:

```bash
outputs/<run_id>/
```

Typical AutoML-specific files include:

- `retrieval_result.json`
- `designer_report.json`
- `baseline_selection_report.json`
- `tuner_report.json`
- `overall_selection_report.json`
- `llm_logs/*.json`

## Setup

This project uses `uv` and [pyproject.toml](/Users/kyle/Projects/bio-model-skills-creator/pyproject.toml).

Python requirement:

- `>= 3.13`

Install dependencies with:

```bash
uv sync
```

If you already use the local virtual environment:

```bash
source .venv/bin/activate
uv sync
```

## Dataset

The active dataset for these pipelines is:

- [data/pxr_activity/train.csv](/Users/kyle/Projects/bio-model-skills-creator/data/pxr_activity/train.csv)
- [data/pxr_activity/test.csv](/Users/kyle/Projects/bio-model-skills-creator/data/pxr_activity/test.csv)

Older top-level dataset files in `data/` are legacy and are not the main inputs for the PXR workflow.

## Legacy Components

These files still exist, but they are not the primary workflow described in this README:

- [manager.py](/Users/kyle/Projects/bio-model-skills-creator/manager.py)
- [scripts/model_search.py](/Users/kyle/Projects/bio-model-skills-creator/scripts/model_search.py)

## Recommended Run Order

For the cleanest workflow:

1. Configure `.env`
2. Run the OpenRouter smoke test
3. Run the deterministic ML pipeline if you want a stable baseline
4. Run the AutoML pipeline if you want the LLM-assisted version

Commands:

```bash
python -m src.agent.test_openrouter
python -m src.pxr_manager
python -m src.agent.run_agent_pipeline
```
