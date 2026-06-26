# ARCHITECTURE — modules, roles, call flow

The pipeline mirrors **AIBuildAI** (arXiv 2604.14455, #1 MLE-Bench): a manager orchestrating
LLM sub-agents (designer / coder / tuner / selector). What's **LLM-driven** = makes a real LLM
API call to decide something; what's **fixed** = deterministic code / a fixed menu the LLM picks from.

## The M3 loop — `src/agent/manager_agent.py`
```
setup → retrieval → designer → coder → select → tuner → select → exporter
```
| stage | module | LLM? | role |
|---|---|---|---|
| setup | `setup_agent.py` | fixed | load data / config |
| retrieval | `retrieval_agent.py` (LLM picks from static `skills/models/manifest.json`) | LLM | choose model-skills — **legacy: 7 ChemBERTa only** |
| designer | `designer_agent.py` / `menu_designer.py` | LLM | propose `(featurizer × model × params)` plans from a **fixed menu** |
| coder | `coder_agent.py` | fixed | **executes** plans via `run_plan_cv` — does NOT write code |
| select | `selector_agent.py` | LLM | pick best plan(s) |
| tuner | `tuner_agent.py` / `menu_tuner.py` | LLM | tune hyperparameters of existing plans |
| exporter | `exporter_agent.py` | fixed | write the submission csv |

LLM plumbing: `src/agent/LLM_base.py` (`LLMJsonAgent.call_json` → OpenRouter HTTP, key in `.env`).

## Core ML (the "fixed menu" the designer composes)
| module | role |
|---|---|
| `src/featurizers.py` | 8 frozen featurizers: morgan/avalon/maccs/rdkit_descriptors/mordred_descriptors/chemberta_embedding/molformer_embedding/chemeleon_embedding |
| `src/models.py` | sklearn model registry (lightgbm/xgboost/rf/ridge/mlp/catboost…) |
| `src/cv_runner.py` | `run_plan_cv`: one plan → 5-fold OOF + test (leak-free, calibrated folds) |
| `src/aggregator.py` | ridge stack over members' OOF |
| `src/analog_judge.py` | the Set-1 judge (RAE/MAE) — the reward signal |
| `src/curation.py` | reactive-electrophile exclusion (0.538-solution lever) |
| `src/cv_split.py` | Butina cluster folds calibrated to the judge → `folds_calibrated.json` |
| `src/schemas.py` | `MenuPlan`, `FoldSpec` |

## Fine-tuning as a pipeline plan (NEW — the AIBuildAI capability we restored)
The performance lever (rank 84→20) is fine-tuning foundation models, which the original menu
couldn't do. Added as a first-class plan type:

| module | role | LLM? |
|---|---|---|
| `src/agent/finetune_designer.py` | **LLM decides WHICH backbones to fine-tune + epochs + stacking** (prefers decorrelated families) | **LLM** |
| `src/finetune_runner.py` | `FineTunePlan` + `build_command` (plan → GPU cmd) + `collect_results` (→ aggregator plan dir) | fixed |
| `scripts/finetune_cheme_mt5.py` | **template**: CheMeleon multitask+MAE fine-tune (single 0.5904) | fixed |
| `scripts/finetune_unimol.py` | **template**: Uni-Mol 3D fine-tune (single 0.6248) | fixed |
| `scripts/run_finetune_plan.py` | run ONE hand-specified plan end-to-end | — |
| `scripts/run_finetune_auto.py` | **end-to-end autonomy**: designer(LLM) → train each → stack → judge → **0.5706** | LLM |

**Call flow (autonomous fine-tuning):**
```
run_finetune_auto.py
  → FineTuneDesigner.propose()                  # LLM: which backbones + epochs  (call_json → OpenRouter)
  → for each plan: build_command()              # plan → "python finetune_<bb>.py ..."
                   subprocess.run(cmd)          # GPU trains via the fixed TEMPLATE
                   collect_results()            # OOF/test → outputs/finetune_auto/plans/<id>
  → aggregate() + analog_judge.judge_csv()      # stack + score
```

**The honest boundary (what's autonomous vs fixed):**
- **LLM-autonomous:** which backbones to fine-tune, epochs, that they should be decorrelated and stacked.
- **Fixed:** the backbone list (`TEMPLATES` = {chemeleon, unimol}) and the training code (per-backbone TEMPLATE script). Adding a backbone = adding a verified template (the model-integration cost). This is **template-based codegen** — safer than AIBuildAI's free codegen, but the LLM does not write the training loop.

## Live model retrieval (NEW — fixes the static-manifest bottleneck)
| module | role |
|---|---|
| `src/agent/hf_retrieval.py` | live HF Hub search, classify by family (graph/3d/smiles/multiview) |
| `scripts/discover_models.py` | HF-live + curated GitHub/Zenodo frontier → `skills/models/candidates_live.json` |

Replaces the static 7-ChemBERTa `manifest.json` with a live, family-classified candidate pool
(so retrieval can find CheMeleon/Uni-Mol/MolE, not just ChemBERTa). Not yet wired into
`retrieval_agent.py` — that's the pending "make retrieval live" step.
