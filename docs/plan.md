# AutoML Automation Plan

## Goal

Build an end-to-end automated pipeline that can:

1. Accept a biological ML task in natural language
2. Discover the relevant model skills already collected from Hugging Face
3. Select feature/model strategies automatically
4. Generate runnable training code
5. Execute evaluation and tuning loops
6. Aggregate results and emit final artifacts
7. Persist enough state and metadata to make the pipeline reproducible

## Current Repo Reality

What already exists:

- `manager.py` with a simple manager loop
- Top-level skills: `setup`, `designer`, `coder`, `tuner`, `aggregator`
- A generated model-skill library under `skills/models/...`
- A model manifest at `skills/models/manifest.json`

What is still missing for real AutoML:

- Recursive loading and indexing of model skills
- Task parsing into a structured problem spec
- Retrieval and ranking of relevant model skills for a given task
- A typed orchestration/state machine instead of a loose prompt loop
- Code generation templates and execution contracts
- Artifact tracking, retries, and reproducibility controls
- Validation that generated plans actually improve downstream results

## Delivery Strategy

Implement in two milestones:

### Milestone 1: Minimal Closed Loop

Objective: one command can go from task description to evaluated candidate models and a final ranked result.

### Milestone 2: Robust AutoML System

Objective: the loop becomes reproducible, debuggable, extensible, and safe enough to run repeatedly on new biological tasks.

## Work Plan

### Phase 0: Stabilize Project Structure

- [ ] Create canonical directories and make every pipeline stage write only into them
- [ ] Standardize:
  - [ ] `data/` for user-provided task data
  - [ ] `outputs/` for run artifacts
  - [ ] `logs/` for structured execution logs
  - [ ] `registry/` for normalized skill indexes
  - [ ] `cache/` for downloaded model metadata or embeddings
- [ ] Define one run ID format so every generated file can be tied back to a specific execution
- [ ] Add one config file for global defaults:
  - [ ] model provider
  - [ ] score thresholds
  - [ ] max candidate plans
  - [ ] tuning budget
  - [ ] retry policy

Exit criteria:

- Every pipeline artifact has a deterministic path
- Re-running does not overwrite unrelated prior runs

### Phase 1: Build a Skill Registry Layer

- [ ] Replace top-level-only skill discovery in `manager.py`
- [ ] Recursively scan `skills/**/SKILL.md`
- [ ] Parse frontmatter for every skill
- [ ] Normalize metadata into a single registry JSON
- [ ] Merge registry data with `skills/models/manifest.json`
- [ ] Add capability fields for each model skill:
  - [ ] domain
  - [ ] modality
  - [ ] input type
  - [ ] output type
  - [ ] task type
  - [ ] framework
  - [ ] source model ref
- [ ] Add validation for malformed or incomplete skills
- [ ] Emit `registry/skills_registry.json`

Subtasks:

- [ ] Define the normalized schema
- [ ] Write registry builder code
- [ ] Add a CLI entrypoint to refresh the registry
- [ ] Add tests for frontmatter parsing and recursive loading

Exit criteria:

- The manager can “see” all collected model skills
- Skill lookup is data-driven, not based on folder assumptions

### Phase 2: Add Task Intake and Problem Structuring

- [ ] Convert free-form user task text into a structured problem spec
- [ ] Extract:
  - [ ] task name
  - [ ] domain
  - [ ] objective type
  - [ ] input modality
  - [ ] labels/targets
  - [ ] metric
  - [ ] split strategy
  - [ ] constraints
  - [ ] expected deliverables
- [ ] Save structured spec as `outputs/<run_id>/task_spec.json`
- [ ] Detect ambiguity and mark fields as `unknown` instead of hallucinating
- [ ] Support task classes like:
  - [ ] sequence classification
  - [ ] sequence regression
  - [ ] token labeling
  - [ ] embedding extraction
  - [ ] multimodal bio tasks later

Subtasks:

- [ ] Define a `TaskSpec` schema
- [ ] Add a parser prompt and validation layer
- [ ] Add fallback heuristics if the LLM output is incomplete

Exit criteria:

- Every run starts from a typed task spec, not raw text only

### Phase 3: Implement Model Skill Retrieval and Ranking

- [ ] Use the structured task spec to retrieve candidate model skills
- [ ] Rank model skills by relevance instead of exposing all skills to the manager
- [ ] Support multiple retrieval axes:
  - [ ] domain match
  - [ ] task-type match
  - [ ] input/output compatibility
  - [ ] model size or runtime feasibility
  - [ ] prior quality score from manifest
- [ ] Return top-K model skills with reasons
- [ ] Persist retrieval results to `outputs/<run_id>/retrieved_models.json`

Subtasks:

- [ ] Define a capability taxonomy for the current model-skill library
- [ ] Add a scoring function for retrieval
- [ ] Add optional LLM reranking for top candidates
- [ ] Mark “feature extractor” vs “direct predictor” vs “NER tool” usage modes

Exit criteria:

- For a genomics/protein task, the system picks a sensible shortlist automatically

### Phase 4: Redesign the Planner Layer

- [ ] Update `designer` so it plans against retrieved model skills, not just generic ML baselines
- [ ] Make every design plan explicit about:
  - [ ] selected skill IDs
  - [ ] feature extraction path
  - [ ] downstream model type
  - [ ] preprocessing
  - [ ] validation strategy
  - [ ] expected risks
- [ ] Generate both classical ML baselines and model-skill-based plans
- [ ] Add plan diversity rules so all plans are not minor variants of one idea
- [ ] Save to `outputs/<run_id>/design_plans.json`

Subtasks:

- [ ] Define `PlanSpec`
- [ ] Add template families:
  - [ ] embedding + linear model
  - [ ] embedding + tree model
  - [ ] handcrafted + classical baseline
  - [ ] hybrid feature concatenation
  - [ ] direct fine-tuning plan when feasible
- [ ] Add pruning logic for obviously invalid plans

Exit criteria:

- Plans are grounded in available skills and executable resources

### Phase 5: Introduce an Execution Contract for the Coder

- [ ] Stop treating code generation as free-form text
- [ ] Require the coder stage to emit structured artifacts:
  - [ ] runnable Python script
  - [ ] config JSON
  - [ ] results JSON
  - [ ] predictions CSV
  - [ ] stderr/stdout logs
- [ ] Move toward template-backed code generation rather than ad hoc prompts
- [ ] Separate reusable pipeline utilities from per-plan generated code

Subtasks:

- [ ] Create script templates for common experiment types
- [ ] Create shared utilities for:
  - [ ] loading data
  - [ ] loading embeddings
  - [ ] metrics
  - [ ] split logic
  - [ ] feature caching
- [ ] Define file naming conventions under `outputs/<run_id>/plans/<plan_id>/`
- [ ] Add code validation before execution

Exit criteria:

- The system can generate, run, and re-run a plan deterministically

### Phase 6: Replace the Loose Manager Loop with a State Machine

- [ ] Refactor `manager.py` into explicit pipeline states
- [ ] Suggested states:
  - [ ] `init`
  - [ ] `build_registry`
  - [ ] `parse_task`
  - [ ] `retrieve_models`
  - [ ] `design_plans`
  - [ ] `execute_plan`
  - [ ] `rank_results`
  - [ ] `tune_best`
  - [ ] `aggregate`
  - [ ] `finalize`
- [ ] Persist state after each transition
- [ ] Add retries and failure handling per state
- [ ] Add resumability from the last successful state

Subtasks:

- [ ] Define a `RunState` schema
- [ ] Write transition rules
- [ ] Add per-stage status files
- [ ] Add guardrails when a stage produces invalid outputs

Exit criteria:

- Pipeline progress is recoverable and inspectable

### Phase 7: Standardize Training and Evaluation

- [ ] Centralize evaluation logic
- [ ] Standardize fold generation and metrics
- [ ] Support:
  - [ ] classification
  - [ ] regression
  - [ ] ranking
  - [ ] token tasks later
- [ ] Save fold-level predictions and scores
- [ ] Generate a leaderboard file for the current run

Subtasks:

- [ ] Build shared metric functions
- [ ] Build split utilities
- [ ] Build evaluation report writer
- [ ] Add sanity checks for leakage and shape mismatches

Exit criteria:

- Every plan is judged on a comparable evaluation contract

### Phase 8: Automate Tuning and Selection

- [ ] Make `tuner` operate on structured result files, not assumptions
- [ ] Define search spaces per model family
- [ ] Control tuning budget explicitly
- [ ] Keep track of every trial
- [ ] Promote the best tuned plan automatically

Subtasks:

- [ ] Create tuning config schema
- [ ] Add a simple search backend first:
  - [ ] grid search
  - [ ] random search
- [ ] Add per-trial logs
- [ ] Write `outputs/<run_id>/best_config.json`
- [ ] Write `outputs/<run_id>/tuning_log.json`

Exit criteria:

- The system can improve the best candidate without manual intervention

### Phase 9: Automate Ensembling and Final Artifact Generation

- [ ] Upgrade `aggregator` from a simple prompt contract into code
- [ ] Support:
  - [ ] rank averaging
  - [ ] weighted averaging
  - [ ] best-single-model fallback
- [ ] Emit final deliverables:
  - [ ] `submission.csv`
  - [ ] `final_report.json`
  - [ ] `run_summary.md`
- [ ] Record which plans contributed to the final ensemble

Subtasks:

- [ ] Define ensemble inputs
- [ ] Add score normalization rules
- [ ] Add artifact validation before writing final outputs

Exit criteria:

- Every run ends with a consistent final artifact bundle

### Phase 10: Add Reproducibility and Observability

- [ ] Log prompts, model choices, parameters, seeds, and package versions
- [ ] Record which skill files were used in each run
- [ ] Snapshot the task spec, registry version, and chosen plans
- [ ] Add structured logs instead of print-only tracing
- [ ] Add a compact human-readable report per run

Subtasks:

- [ ] Define a run manifest
- [ ] Add timestamps and durations per stage
- [ ] Add failure summaries
- [ ] Add environment capture

Exit criteria:

- You can explain why a run produced its result

### Phase 11: Add Quality Gates and Tests

- [ ] Add unit tests for registry, parsing, retrieval, metrics, and orchestration
- [ ] Add a smoke test that runs a tiny end-to-end pipeline
- [ ] Add regression fixtures for one known task
- [ ] Add validation for generated JSON artifacts

Subtasks:

- [ ] Test recursive skill discovery
- [ ] Test malformed skill handling
- [ ] Test task parsing fallback behavior
- [ ] Test retrieval ranking
- [ ] Test state resume

Exit criteria:

- Core pipeline logic is protected from trivial regressions

### Phase 12: Expose a Usable CLI

- [ ] Add a single command to run the whole pipeline
- [ ] Add subcommands for each stage
- [ ] Add flags for dry-run, resume, top-K models, and tuning budget

Suggested commands:

- [ ] `python manager.py run --task-file ...`
- [ ] `python manager.py build-registry`
- [ ] `python manager.py retrieve --task-file ...`
- [ ] `python manager.py resume --run-id ...`

Exit criteria:

- The system is operable without editing code for each run

## Recommended Build Order

### P0: Must Do First

- [ ] Phase 0: Stabilize project structure
- [ ] Phase 1: Skill registry layer
- [ ] Phase 2: Task intake and problem structuring
- [ ] Phase 3: Model skill retrieval and ranking
- [ ] Phase 6: State machine orchestration

### P1: Needed for a Real Closed Loop

- [ ] Phase 4: Planner redesign
- [ ] Phase 5: Execution contract for coder
- [ ] Phase 7: Standardized training and evaluation
- [ ] Phase 8: Tuning and selection
- [ ] Phase 9: Ensembling and final artifacts

### P2: Needed for Reliability and Reuse

- [ ] Phase 10: Reproducibility and observability
- [ ] Phase 11: Quality gates and tests
- [ ] Phase 12: CLI

## Concrete File-Level Changes Likely Needed

- [ ] Update [manager.py](/Users/kyle/Projects/bio-model-skills-creator/manager.py)
- [ ] Add `plan.md` tracking updates as implementation progresses
- [ ] Add `registry/skills_registry.json` generator
- [ ] Add a module for schemas and typed artifacts
- [ ] Add a module for task parsing
- [ ] Add a module for model retrieval
- [ ] Add a module for orchestration/state management
- [ ] Add shared experiment utilities
- [ ] Add tests under `tests/`

## Definition of Done

The AutoML chain is “done” when all of the following are true:

- [ ] A new biological task can be provided in natural language
- [ ] The system builds or refreshes a skill registry automatically
- [ ] The system selects relevant Hugging Face model skills automatically
- [ ] The planner proposes executable, diverse model plans
- [ ] The coder produces runnable experiment artifacts
- [ ] The pipeline evaluates, tunes, and ranks candidates automatically
- [ ] The final result is written with reproducible metadata
- [ ] A failed run can be resumed without manual cleanup
- [ ] The main path is covered by tests

## Immediate Next Actions

If the goal is to start implementation now, do these first:

1. [ ] Refactor `manager.py` to recursively load all skills and persist a normalized registry
2. [ ] Define `TaskSpec`, `PlanSpec`, and `RunState` schemas
3. [ ] Add a retrieval step that maps task specs to the top relevant model skills
4. [ ] Redesign `designer` to consume retrieved skills and produce structured plans
5. [ ] Replace the current prompt loop with a resumable stage-based orchestrator

