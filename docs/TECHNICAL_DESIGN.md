# TECHNICAL DESIGN — llm-bio-automl

> Companion to **[PRODUCT_DESIGN.md](PRODUCT_DESIGN.md)** (the PRD).
> The PRD answers *what* and *why*. This document answers *how*: modules, data structures, function signatures, and the file-level task list for each milestone.
> All terms are defined in plain language in **PRD §0 (Glossary)**.

| Field | Value |
|---|---|
| Version | v1.0 |
| Status | Draft (ready to implement M0→M1) |
| Owner | Kyle |
| Created | 2026-06-20 |

---

## T0. How this maps to the PRD

| PRD section | This doc |
|---|---|
| §9 Flowchart (10 nodes) | T3 Component design (one block per node) |
| §11 Architecture & data model | T1 Module map, T2 Data structures |
| §11.2 Approaches 1/2/3 | T4 Milestone → approach → tasks |
| §12 Data contracts | T2.2 Artifact schemas |
| §15 Roadmap M0–M4 | T4 Milestone task lists |
| §8 Principle "never leak" | T5 Leakage-safety rules |

We are building **Approach 1 (the rich menu)** first (PRD §11.2). Approach 2 (LLM-codegen) is deferred to M4 and only sketched here.

---

## T1. Module map (directory layout)

Deterministic execution layer lives in `src/`; the LLM orchestration layer in `src/agent/`.

| Module | Purpose | Status |
|---|---|---|
| `src/data_io.py` | load train/test; canonicalize SMILES; write `dataset_report.json` | **new** (replaces ad-hoc parts of `data_utils.py`) |
| `src/cv_split.py` | scaffold cross-validation folds + leakage check | ✅ **done** |
| `src/featurizers.py` | Featurizer **registry** + each featurizer (molecule → numbers) | **new** (M1) |
| `src/models.py` | Model **registry** + each model wrapper (numbers → pEC50) | **new** (M1) (naming caution: `src/agent/models.py` already exists and is the ChemBERTa loader — reusable, but rename it to `embedding_loader.py` to avoid the clash) |
| `src/metrics.py` | RAE / MAE / R² | **new** (M1) — centralizes scoring |
| `src/cv_runner.py` | run one PlanSpec across folds → OOF + test preds + metrics | **new** (M1) — the real "Coder" of Approach 1 |
| `src/selector.py` | deterministic ranking + leaderboard | refactor (drop the wasted LLM call) |
| `src/aggregator.py` | OOF stacking / weighted blend + multi-seed + full refit | **new** (M2) |
| `src/exporter.py` | build & validate `submission.csv` + report | exists; extend with manifest |
| `src/schemas.py` | dataclasses (TaskSpec, PlanSpec, FoldSpec, RunState) | refactor (see T2.1) |
| `src/featurizer_cache/` | on-disk cache of computed features/embeddings | **new** dir (M1) |
| `src/agent/designer_agent.py` | LLM proposes N PlanSpecs from the menu | rewrite (M3) |
| `src/agent/tuner_agent.py` | LLM proposes next hyperparameters (keep diagnostics) | refactor (M3) |
| `src/agent/manager_agent.py` | decision loop + budget + resume | refactor (M3) |

> Design rule: **the Coder (`cv_runner.py`) never changes when we add a representation or model** — we only register a new function in `featurizers.py` / `models.py`. This is what makes the menu cheap to grow (PRD §11.2).

### T1.1 Current file inventory & status (the existing codebase)

The repo uses a **two-layer** design, so files come in worker↔wrapper pairs (e.g. `runner.py` ↔ `coder_agent.py`) — they are *not* duplicates. Legend: ✅ keep · 🔁 replace/refactor in M1–M3 · ♻️ reusable · 💀 dead/stale · 🧪 scratch test.

**Layer 1 — `src/` (deterministic workers)**

| File | Purpose | Status |
|---|---|---|
| `cv_split.py` | scaffold cross-validation folds | ✅ newest (done) |
| `schemas.py` | data types (TaskSpec, PlanSpec…) | ✅ keep (refactor M1) |
| `data_utils.py` | load + validate data | ✅ keep (extend M0) |
| `task_spec_builder.py` | builds the fixed PXR task spec | ✅ keep |
| `constants.py` | run_id generator | ✅ keep |
| `download_pxr_data.py` | one-off dataset download | ✅ keep (utility) |
| `selector.py` | ranking helpers | ✅ keep |
| `tuner.py` | deterministic tuning grid | ✅ keep (fallback) |
| `exporter.py` | build submission.csv | ✅ keep (extend M2) |
| `runner.py` | featurize+train one plan (few combos) | 🔁 replaced by `featurizers.py`+`models.py`+`cv_runner.py` (M1) |
| `planner.py` | builds fixed baseline plans | 🔁 superseded by registry+designer (M1) |
| `pxr_manager.py` | old deterministic pipeline entry point | 🔁 superseded by new flow (M1) |
| `xgboost_worker.py` | runs XGBoost in a subprocess (made the tmp_xgb files) | 🔁 fold into `models.py` (M1) |
| `test.py`, `test_runner.py`, `test_selector.py` | ad-hoc smoke scripts | 🧪 scratch |

**Layer 2 — `src/agent/` (LLM / orchestration wrappers)**

| File | Purpose | Status |
|---|---|---|
| `LLM_base.py` | base agent + OpenRouter call helper | ✅ infra |
| `agent_context.py` | RunContext / AgentResult types | ✅ infra |
| `models.py` | ChemBERTa loader (tokenizer/encoder, offline, CPU) | ♻️ reuse for embedding featurizer (rename → `embedding_loader.py`) |
| `tuner_agent.py` | the one agent with a working LLM loop + diagnostics | ✅ keep (refactor M3) |
| `setup_agent.py` | wraps data validation + task spec | ✅ keep |
| `retrieval_agent.py` | picks representations | ✅ keep (refine M3) |
| `exporter_agent.py` | wraps exporter | ✅ keep |
| `run_agent_pipeline.py` | agent-pipeline CLI entry point | ✅ keep |
| `coder_agent.py` | wraps `runner.py`, locked to 2 families | 🔁 replace with `cv_runner` (M1/M3) |
| `selector_agent.py` | wraps selector + a wasteful LLM call | 🔁 simplify (M1) |
| `manager_agent.py` | fixed linear pipeline (no real decision loop) | 🔁 refactor (M3) |
| `designer_agent.py` | **stubbed** — returns 1 fixed plan, never calls the LLM | 💀 dead → rewrite (M3) |
| `test_openrouter.py`, `test_retrieval.py` | smoke scripts | 🧪 scratch |

---

## T2. Core data structures (the contracts)

### T2.1 In-memory dataclasses (`src/schemas.py`)

```python
@dataclass
class TaskSpec:
    challenge_name: str
    task_title: str
    task_description: str
    target_column: str          # "pEC50"
    primary_metric: str         # "RAE"
    submission_columns: list[str]
    data_dir: str

@dataclass
class PlanSpec:
    plan_id: str
    name: str
    featurizer: str             # a key in the featurizer registry, e.g. "morgan"
    model: str                  # a key in the model registry, e.g. "xgboost"
    params: dict                # hyperparameters for featurizer+model
    seeds: list[int]            # e.g. [42]; multi-seed averaging in M2
    skill_ref: str | None = None    # for embedding featurizers (e.g. ChemBERTa id)
    skill_path: str | None = None
    notes: str = ""

@dataclass
class FoldSpec:
    strategy: str               # "scaffold"
    n_folds: int
    seed: int
    assignments: dict[int, int] # row index -> fold id
```

> Change vs current `schemas.py`: `feature_type/model_type` → `featurizer/model`; add `seeds`; add `FoldSpec`.

### T2.2 On-disk artifact schemas (PRD §12)

`folds.json` (produced ✅ by `cv_split.py`):
```json
{"strategy":"scaffold","n_molecules":4139,"n_folds":5,"seed":42,
 "fold_sizes":[828,828,828,828,827],"scaffolds_leaking_across_folds":0,
 "assignments":{"0":3,"1":1, "...":"..."}}
```

`plans/<plan_id>/metrics.json`:
```json
{"plan_id":"...","featurizer":"morgan","model":"xgboost","params":{...},
 "cv":{"rae":{"mean":0.49,"std":0.02,"per_fold":[...]},
       "mae":{...},"r2":{...}},
 "oof_path":"...","test_path":"...","runtime_sec":0,"status":"ok","error":null}
```
- `plans/<id>/oof_predictions.csv` — columns: `row_id, y_true, y_pred`
- `plans/<id>/test_predictions.csv` — columns: `Molecule Name, SMILES, pEC50`
- `leaderboard.json`, `best_plan.json`, `ensemble_report.json` — see PRD §12.

---

## T3. Component design (Approach 1) with function signatures

### T3.1 Data layer — `src/data_io.py`
```python
def load_train(data_dir) -> pd.DataFrame        # 4139 rows
def load_test(data_dir)  -> pd.DataFrame        # 513 rows
def canonicalize_smiles(s: str) -> str | None   # RDKit canonical form; None if unparseable
def prepare_dataset(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, dict]
    # returns (clean_train, clean_test, dataset_report)
```
Notes: data is already clean (verified: 0 dupes, 0 parse failures), so this mostly canonicalizes and writes a report. It does **not** drop or impute anything silently.

### T3.2 Cross-validation — `src/cv_split.py` (✅ done)
```python
bemis_murcko_scaffold(smiles) -> str | None
assign_scaffold_folds(smiles_list, n_folds=5, seed=42) -> (fold_of_row, diagnostics)
verify_no_scaffold_leakage(smiles_list, fold_of_row) -> int   # must be 0
build_scaffold_folds_for_csv(csv_path, ...) -> diagnostics
```

### T3.3 Featurizer registry — `src/featurizers.py`  (molecule → numbers)
> "Featurizer" = the converter that turns the raw input (a SMILES *text* string) into a fixed-length list of numbers a model can read. See T-NOTE below on representations.

```python
# every featurizer implements: smiles_list -> np.ndarray of shape (n_molecules, n_features)
FEATURIZERS: dict[str, Callable[[list[str], dict], np.ndarray]] = {}

def register(name): ...                      # decorator to add to the registry

@register("morgan")            # circular fingerprint (0/1 bits)        ✅
@register("maccs")             # 166 standard substructure keys         ✅
@register("avalon")            # Avalon fingerprint (0/1 bits)          ✅
@register("rdkit_descriptors") # physicochemical properties            ✅
@register("chemberta_embedding")  # pretrained CLM vector, 77M + 100M  ✅
@register("fusion")            # concat descriptors + chemberta         ✅
# @register("molformer_embedding")  # MolFormer CLM — TODO (menu gap)  ❌
def _featurize(smiles_list, params) -> np.ndarray: ...

def featurize(name, smiles_list, params, cache_key=None) -> np.ndarray
```
Notes:
- `chemberta_embedding` downloads the model once (`DeepChem/ChemBERTa-77M-MTR`), runs a forward pass on CPU/MPS, and **caches** results in `src/featurizer_cache/` keyed by (model id, params, SMILES set hash).
- Fingerprint/descriptor featurizers are stateless → safe to compute once for all rows. Any featurizer that *learns* from data (e.g. a scaler) must be fit inside the train fold only (T5).

### T3.4 Model registry — `src/models.py`  (numbers → pEC50)
```python
MODELS: dict[str, Callable[[dict], RegressorLike]] = {}
@register("ridge"); @register("elastic_net"); @register("random_forest")  # ✅
@register("xgboost"); @register("lightgbm"); @register("catboost")         # ✅
# @register("mlp_head")   # MLP on features/embeddings — TODO (menu gap) ❌
def _make_model(params) -> object   # returns an sklearn-style .fit/.predict object
# TODO: every wrapper must accept sample_weight (from pEC50_std.error) in .fit()
```

### T3.5 Cross-validated runner — `src/cv_runner.py`  (the Approach-1 "Coder")
```python
def run_plan_cv(plan: PlanSpec, train_df, test_df, folds: FoldSpec,
                out_dir) -> dict:   # returns metrics.json content
    # for each fold k:
    #   X_tr = featurize(plan.featurizer, train_smiles[train_idx], plan.params)
    #   X_va = featurize(plan.featurizer, train_smiles[val_idx],  plan.params)  # transform only
    #   model = make_model(plan.model, plan.params).fit(X_tr, y_tr)
    #   oof[val_idx] = model.predict(X_va)
    # test_pred = average of fold models' predictions on the test set
    # write metrics.json, oof_predictions.csv, test_predictions.csv
```
This is the heart of Approach 1: one function turns *any* registered (featurizer, model, params) into honest scores + OOF, with no code generation.

### T3.6 Metrics — `src/metrics.py`
```python
def rae(y_true, y_pred) -> float   # relative absolute error (primary)
def mae(y_true, y_pred) -> float
def r2(y_true, y_pred)  -> float
def score_all(y_true, y_pred) -> dict
```

### T3.7 Selector — `src/selector.py` (refactor)
```python
def rank_plans(metrics_list, primary_metric="RAE") -> list[dict]   # deterministic sort
def pick_top_k(ranked, k) -> list[dict]
```
Removes the current LLM call that just "picks the already-top-ranked".

### T3.8 Aggregator — `src/aggregator.py` (M2)
```python
def stack_oof(oof_frames: list[pd.DataFrame], y_true) -> StackModel  # meta-model on OOF
def blend(test_preds: list[np.ndarray], weights) -> np.ndarray
def aggregate(top_k_plans, run_dir) -> dict   # writes ensemble_report.json
```

### T3.9 Exporter — `src/exporter.py` (extend)
Already builds/validates `submission.csv`. Add a reproducibility manifest (seed, package versions, plan list).

### T3.10 LLM layer (M3, sketched) — `src/agent/*`
Designer → emits N `PlanSpec` JSON from the menu; Tuner → proposes next params (keep its diagnostics); Manager → decision loop + budget + resume. All keep deterministic fallbacks (PRD §16 risks).

### T-NOTE: what "representation" means and the input for *this* competition
The raw input for this competition is **text** — a SMILES string such as `CCO` (ethanol). Models can't read text, so a **featurizer** converts it into numbers (a "representation"). Same molecule, different representations:
- **as a fingerprint** → a long list of 0/1 bits (Morgan/MACCS),
- **as descriptors** → a list of measured properties (weight, polarity, …),
- **as an embedding** → a vector from a pretrained network (ChemBERTa),
- **as a graph** → atoms-and-bonds (what GNNs use — not in Approach 1's first menu).

Approach 1 uses the text→numbers route. The pipeline is: **SMILES text → featurizer → numbers → model → predicted pEC50.**

---

## T4. Milestone → approach → technical tasks (mirrors PRD §15)

### M0 — Foundation  (approach: prerequisite)
- [x] `cv_split.py`: scaffold folds + leakage check (verified 0 leakage, balanced)
- [ ] `data_io.py`: load + canonicalize + `dataset_report.json`
- [ ] Unify LLM config: one provider, documented `.env` keys (see T7)
- **DoD:** folds verified; one clean data-prep entry point; config no longer split.

### M1 — Modeling menu  (approach: **Approach 1**)  — ✅ built, completing
- [x] `featurizers.py`: `morgan`, `maccs`, `avalon`, `rdkit_descriptors`, `chemberta_embedding` (77M+100M), `fusion`
- [x] `models.py`: `ridge`, `elastic_net`, `random_forest`, `xgboost`, `lightgbm`, `catboost`
- [x] `metrics.py` (rae/mae/r2); `cv_runner.py` (`run_plan_cv` → metrics + OOF + test preds, full-data refit, multi-seed)
- [x] First end-to-end: 24-candidate menu + ridge-stack ensemble (scaffold-CV 0.549)
- [x] **Menu complete (2026-06-21):** `molformer_embedding` featurizer · `mlp_head` model · **sample weights** from `pEC50_std.error` (threaded via `models.fit_model` + `cv_runner`). Built the missing broad-train + Set-1-judge eval (`src/analog_judge.py`, `scripts/run_judge.py`, `scripts/ensemble_compare.py`).
- **DoD met:** each module judged on **Analog Set 1**. **Kept `mlp_head`** (ensemble 0.636→0.632); **dropped `molformer`** (single RAE≈1.0) and **sample weights** (raise judge RAE) — kept in the registries as levers. See RESULTS.md §4.

### M2 — Ensemble  (approach: Approach 1)  — ✅ done
- [x] `aggregator.py`: OOF stacking (mean/nnls/ridge) + multi-seed + full-data refit; `exporter.py` manifest
- **DoD met:** ridge-stack beats best single by +5.8%.

### M2.5 — Analog pivot  (approach: Approach 1)  — ✅ done (no fold-in; judge stays out of training)
- [x] Download Analog Set 1 (253 labels); score ensemble on it (**0.633** reality vs 0.543 scaffold-CV)
- [x] **`src/analog_judge.py` + `scripts/judge.py`** — the Set-1 judge (replaces the deprecated `run_analog.py` fold-in path; reproduces 0.6329)
- [x] **`scripts/run_judge.py` + `ensemble_compare.py`** — broad-train + judge eval; menu completed & judged: **mlp_head kept** (ensemble 0.636→0.632), molformer + sample-weights dropped
- [x] **`scripts/calibrate_folds.py` + `cv_split.assign_cluster_folds`** — calibrate internal CV to the judge (cluster@0.6 ranks like the judge; `folds_calibrated.json`)
- [x] **`scripts/finalize.py`** — strong menu on calibrated folds → ridge-stack → **submission judge RAE 0.6266** (`outputs/final/`)
- **DoD met:** analog-validated submission; CV calibrated; menu frozen. (Fold-in + external data remain deferred to the data-side phase.)

### M3 — Automation  (approach: Approach 1 + LLM orchestration)  — ✅ built & working
A **judge-in-the-loop** Manager loop on top of the deterministic layer. Each round: Designer proposes N `MenuPlan`s → run on calibrated folds → judge on Set 1 → Aggregator stacks the pool → Manager continues/stops on budget. Resumable via `run_state.json`. **This is the first LLM-in-the-loop part of the pipeline** (everything before is deterministic).
- [x] **`src/agent/menu_designer.py`** — `MenuDesigner.propose()`: LLM emits N diverse `MenuPlan`s from the registries (validated by construction, skill_ref normalized, molformer excluded), informed by prior-round judge scores; **deterministic preset fallback** when the LLM is unavailable/returns junk
- [x] **`scripts/run_auto.py`** — Manager loop: seed from the frozen menu (anchor 0.6266) → propose→run→judge→stack→decide; budget (`--rounds`/`--candidates`/`--patience`), resume, **never-regress** submission; **objective = Set-1 judge**
- [x] per-candidate ✓/✗ isolation (serial; parallel is a straightforward extension)
- [x] **`src/agent/menu_tuner.py`** — LLM Tuner wired into `run_auto.py --tune-top`: proposes hyperparameter sets for the top bases (deterministic perturbation fallback), judged on the **calibrated folds + Set-1 judge** (fixing the earlier scaffold-CV tuning). Verified end-to-end (0.6266→0.6260; never-regress holds).
- **DoD met:** one command auto-produces a better submission within a budget (0.6266→0.618–0.624), resumable, never worse than the anchor; LLM Designer + Tuner both live, with fallback demonstrated. (Legacy `designer_agent.py`/`tuner_agent.py`/`manager_agent.py` superseded by the clean `MenuPlan`-based modules. Selector stays deterministic by design; retrieval dormant.)

### M4 — Escape hatch (optional)  (approach: **Approach 2 → Approach 3**)  — ❌ not needed / skip
- [ ] `custom_code` plan type: LLM writes a snippet, run in a **sandbox** with a sanity check
- **Decision (2026-06-22): skip.** The DoD trigger ("menu caps the score") never fired — the menu sweep and the M3 auto-loop both plateau at ~0.62, so the ceiling is **data**, not the method menu. Codegen's cost (sandbox + debug loop + non-reproducibility) isn't justified. Any analog-aware method we'd want (kNN, GP) can be added as ordinary menu modules without codegen. Revisit only if the data side stalls.

### M5 — Data side  (approach: **data, not method**)  ← current priority
The binding constraint is the *training distribution* (broad-only, ~0.62 ceiling), so expand the training signal. Everything is **judged on the Set-1 judge** (keep a source only if it lowers RAE); Set 1 stays OUT of training until the very end. Data files & paths are in CHALLENGE_BRIEF §5.

- [x] **(a) Auxiliary in-dataset configs — done; clean directly-usable data did NOT help.**
  - Inspected all four. Usability: `counter_assay` (2.9K) = **different target** (counter-screen, pEC50 mean 3.1≠4.3, 0 new molecules); `single_concentration` (21K) = **no pEC50** (single-conc → `log2_fc` only) but **8,135 NEW molecules**; `crudes` (456) + `semi_pure` (96) = real PXR pEC50.
  - **Leakage check caught semi_pure: 1 mol in test, 1 in Set 1 → removed.** Built `data/pxr_activity/aux_train.csv` (435 clean rows, deduped vs broad/test/Set-1). Extended `cv_runner` with `aux_train_df`/`aux_weight` (aux **always in training, never held out** → OOF + judge stay on broad).
  - **Result (`scripts/run_aux.py`):** +aux **hurts** the Set-1 judge — 7-base ensemble 0.6496 → **0.6599** (w=1.0), 0.6552 (w=0.3). Down-weighting only shrinks the harm toward 0. **Dropped** — low-fidelity HT-chem chemistry is noise for the analog task.
- [ ] **(b) ⚠️ Plan revised by competitor evidence (PRD §7.1, 2026-06-22).** The two leading solutions (RAE 0.538 / 0.566) win mainly via a **graph foundation model (ChemProp + CheMeleon)** — which we lack — and they fine-tune it on the `single_concentration` molecules. So the next lever is **method+data together**, not external rows:
  - **(b1) HIGH PRIORITY — add ChemProp/CheMeleon** as a model+featurizer (pretrained molecular MPNN; `pip install chemprop`). This is the deferred GNN and the documented ~0.62→~0.54 path; in the 0.538 solution the graph head carries 71% of the ensemble weight.
  - **(b2)** fine-tune / multi-task that graph model on `single_concentration`'s **8,135 new molecules** (binary active/inactive + continuous `log2_fc`) — the correct way to use them (appending pEC50 rows hurt, per (a)).
  - **(b3) DROP external data (DataMaster / ChEMBL / BindingDB)** — both our M5(a) result AND aetherark ("external data only hurt") confirm it's a dead end for this task.
- [ ] **(c, end-game) Set-1 fold-in** for the final Set-2 submission: refit on broad + auxiliary + **all 253 Set-1**, predict 513. ⚠️ This **consumes the judge** (we can no longer validate on Set 1) — do it last, once the recipe is frozen.
- **DoD:** ≥1 clean data source lowers the Set-1 judge RAE below 0.62; a documented fold-in submission ready for July 1; dedupe verified (0 test-analog leakage).

### M6 — Research-grounded competitor playbook  (approach: **deep research → grounded implementation**)  ← current
Pipeline: deep-research the documented top solutions → extract winning techniques (`research/worth_trying_*.md`) → implement & judge each on Set 1 → keep what lowers RAE. Mirrors the leaders' recipe (CheMeleon graph model + multitask + counter-assay + cluster CV + honest calibration).

- [x] **CheMeleon graph model** — `chemeleon_embedding` featurizer (frozen, CPU, cached) + GPU fine-tune (`scripts/finetune_cheme.py`, chemprop `--from-foundation CheMeleon`, 5-fold on calibrated folds). Ensemble judge **0.6266 → 0.6217 (frozen) → 0.6135 (fine-tuned)**. Helper: `scripts/run_cheme.py`, `outputs/cheme_ft/`.
- [x] **honest per-fold isotonic calibration** — tried; **no help on the judge** (broad-fitted calibration doesn't transfer to the analog set; 0.6135→0.6145). Kept off.
- [ ] **Multitask Chemprop** (HIGH — the documented 0.689→0.60 lever): extend `finetune_cheme.py` to multi-target — primary pEC50 + counter-assay pEC50/Emax (+ targeted external: ChEMBL NR1I2, NCATS qHTS PXR). Mask NaNs per-target. The decisive "is broad-only tapped out?" experiment.
- [ ] **counter-assay weighting** (LOW): down-weight compounds with high signal in BOTH primary & PXR-null counter (assay interference) via the existing `sample_weights.py` path. One competitor: RAE 0.81→0.58.
- [ ] **Mordred + FCFP4-count features** (LOW) for the GBDT branch; **prune ensemble members with OOF-corr > 0.9** to the blend.
- [ ] **End-game:** Set-1 fold-in (consumes the judge — last) + **mandatory method report** (no report = not ranked).
- **DoD:** best Set-1 judge RAE recorded; recipe frozen + report submitted by July 1. **Target = top-5 — genuinely achievable on public data** (per §8 most top-5 entries used no proprietary data); the gap to the lead pack (~0.50) is **execution depth** (full multitask + counter + calibration + Phase-2 fold-in), not data access.

---

## T5. Leakage-safety rules (technical) — enforce PRD principle "never leak"
1. Folds are frozen once (`folds.json`); every plan uses the same folds.
2. Any featurizer/model that *learns* parameters (scalers, target-aware encoders, the model itself) is fit on the **train fold only**, then applied to the validation fold.
3. The feature cache is keyed by content (featurizer + params + SMILES-set hash), never by fold, so caching can't leak labels.
4. OOF predictions for a row come only from a model that never saw that row in training.
5. Test predictions = average over the per-fold models (or a model refit on full train for the final submission).

---

## T6. Dependencies & environment
- **Installed:** rdkit 2026.03.1, scikit-learn 1.8, xgboost 3.2, torch 2.11, transformers 5.5, pandas 3.0, numpy 2.4.
- **To add (via `uv add`):** `lightgbm`, `catboost`.
- **ChemBERTa:** model id `DeepChem/ChemBERTa-77M-MTR`; runs on CPU/MPS (no GPU); embeddings cached to `src/featurizer_cache/`.

## T7. Config unification (M0)
- Today: selector/retrieval use OpenRouter; designer/tuner use the Anthropic SDK directly (undocumented env). 
- Target: a single `llm_config` with one provider choice and documented `.env` keys; every agent reads it. (LLM is only needed in M3, so this can be finished alongside M1.)

## T8. Testing strategy
- `cv_split`: assert 0 scaffold leakage; folds balanced (✅ verifiable now).
- `featurizers`: output shape `(n, n_features)`; deterministic; cache hit returns identical array.
- `metrics`: known-input checks for rae/mae/r2.
- `cv_runner`: every train row gets exactly one OOF prediction; no NaNs.
- `exporter`: submission has correct columns and 513 rows, no nulls.

## T9. Open technical decisions
- `n_folds` = 5 (default; revisit if folds get too small for embeddings).
- Multi-seed count for M2 (start with 3).
- Whether to weight samples by measurement uncertainty (PRD §16) — try in M1/M2 as an experiment.
- Stacking meta-model choice (start: ridge on OOF).
