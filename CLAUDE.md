# CLAUDE.md — project context (auto-loaded every session)

LLM-driven AutoML for the **OpenADMET PXR Activity Prediction** challenge. Goal: top-5 rank on
the blind Set-2 + a **methodology report due July 1**. This file is the handoff/state doc — read
`RESULTS.md` for the full numeric log and `docs/` for design docs.

## Current best (Set-1 judge)
- **RAE 0.5706 / MAE 0.4557** — `predictions/` (`oof_unimol.csv` + `test_unimol.csv` stacked with mt5).
  Ridge stack of **CheMeleon (graph, fine-tuned) + Uni-Mol (3D, fine-tuned)**. ≈ leaderboard **rank ~20/328**.
- Progression: 0.6329 → 0.6217 → 0.6135 → 0.6108 → 0.5916 → **0.5706**.
- The **pipeline itself** (frozen embeddings + sklearn, no fine-tuning) only reaches **RAE ~0.62 / rank ~84**.
  The gap to 0.57 is **fine-tuning**, which is currently done by standalone scripts OUTSIDE the pipeline.

## The pipeline (mirrors AIBuildAI, arXiv 2604.14455, #1 on MLE-Bench)
`src/agent/manager_agent.py` orchestrates: setup → **retrieval** → **designer** → coder → selector →
**tuner** → selector → exporter. LLM-driven stages: retrieval, designer, selector(s), tuner.
The original menu loop only combined **8 frozen featurizers × fixed sklearn models** (rank ~84) — no fine-tuning, which is where the performance is (rank ~20). **Full module map + call flow: `docs/ARCHITECTURE.md`.**

**LLM-orchestrated fine-tuning is now ADDED (prof-approved, end-to-end, reproduces 0.5706):**
- `src/agent/finetune_designer.py` — **LLM decides** which backbones to fine-tune + epochs + stacking (picks decorrelated families).
- `src/finetune_runner.py` — `FineTunePlan` + `build_command` (plan→GPU cmd) + `collect_results` (→ aggregator).
- `scripts/run_finetune_auto.py` — end-to-end: LLM designs → GPU trains each template → stack → judge → **0.5706**.
- **Boundary:** LLM picks WHAT to fine-tune (backbone/epochs/stack) from a FIXED list `{chemeleon, unimol}`; the training code is a verified TEMPLATE per backbone (`scripts/finetune_*.py`) — template-based codegen, not free codegen. Adding a backbone = adding a template (the model-integration cost).
- **Still pending:** wire `finetune_designer` into `manager_agent` (so the manager loop proposes FT plans natively, not just the standalone `run_finetune_auto.py`); wire `hf_retrieval` into `retrieval_agent`.

## Key findings (don't re-derive)
- **Decorrelation thesis (validated):** 2 decorrelated FINE-TUNED foundation models (graph + 3D) stacked beat a 48-base correlated ensemble. corr(unimol,mt5)=0.866. Single ≥ ensemble when members are correlated.
- **Frozen embedding ≠ fine-tuning:** frozen CheMeleon ~0.68 single / ~0.62 ensemble; FINE-TUNED 0.5904. The gain is in fine-tuning.
- **Negatives (verified, don't retry):** TTA aug10 (helps single, HURTS stack — reduces decorrelation), counter-assay weighting, Mordred features, isotonic calibration, ChemBERTa/MolFormer (SMILES transformers, ~0.74-0.82, weak).
- **Model-integration friction:** strong models live on GitHub not HF (CheMeleon, Uni-Mol, MolE, GROVER, MolCLR). Each = a dependency-archaeology battle (old torch/python, fast_transformers, torch_scatter). MolE weights are withheld; IBM multi-view blocked on fast_transformers.
- **RAE/MAE judge caveat:** our judge is the 253 public Set-1; the board scores the full 513 (260 still blind). RAE (normalized) transfers ≈ board rank; **MAE does NOT** (our 253 subset has higher MAD 0.799 vs board's ~0.758 → our MAE looks ~0.02 higher at the same RAE). Use RAE for board comparison.

## Hard constraints
- **Set-1 (`data/pxr_activity/phase1_unblinded.csv`, 253 labels) is the JUDGE — NEVER put it in training.** Read only for scoring via `src/analog_judge.py`. (End-game fold-in is the one deliberate exception, done LAST for the final Set-2 submission.)
- **Do NOT use `miscellaneous/run_analog.py`** (deprecated — it does fold-in).
- `.env` has the OpenRouter API key — gitignored, never print/commit it.

## Key files
- `src/analog_judge.py` — the Set-1 judge (RAE/MAE). `src/curation.py` — reactive-electrophile exclusion.
- `scripts/finetune_cheme_mt5.py` — CheMeleon multitask fine-tune (single 0.5904). `scripts/finetune_unimol.py` — Uni-Mol 3D (single 0.6248, `--tta-only` for TTA, kfold=1).
- `src/agent/hf_retrieval.py` + `scripts/discover_models.py` — live HF + frontier model retrieval (replaces the static 7-ChemBERTa manifest). Output: `skills/models/candidates_live.json`.
- `src/cv_runner.py` (`run_plan_cv`), `src/aggregator.py` (ridge stack), `data/pxr_activity/folds_calibrated.json` (cluster folds calibrated to the judge).

## Environment / how to run
- GPU: DSMLP A5000 pod (`launch.sh -g 1 -v a5000 -m 64`), **6h session limit**, ~11G home quota (TIGHT — auto-prune checkpoints/.sdf after extracting OOF/test). Models are small (~10M params); A5000 is plenty, no A100 needed.
- Weights auto-download on first run: CheMeleon (chemprop→Zenodo `~/.chemprop`), Uni-Mol (unimol_tools→Zenodo), ChemBERTa (transformers→HF). **Do not commit weights** — DSMLP fetches them.
- Local dev: `uv run python ...`. Fine-tuning needs the pod GPU.

## Next steps (priority order)
1. ✅ **LLM-orchestrated fine-tuning** (prof-approved) — DONE end-to-end via `run_finetune_auto.py` (reproduces 0.5706). Remaining: wire `finetune_designer` into `manager_agent` so the main loop does it natively; wire `hf_retrieval` into `retrieval_agent`.
2. **End-game Set-1 fold-in** (~0.10 on singles, irreversible — consumes the judge) for the final Set-2 submission. Needs a working GPU env (`~/.local` torch/chemprop/unimol kept). Do LAST.
3. **Methodology report** (July 1, mandatory).
4. (optional) DataMaster = data-side autonomous twin of the model-retrieval component (paper completeness, low score value).

## Running on DSMLP
- SSH: `~/.ssh/config` has `dsmlp` (ControlMaster → Duo once/8h). Pod: `launch.sh -g 1 -v a5000 -m 64`, 6h limit.
- env deps in `~/.local` (torch 2.4.1+cu121, chemprop, unimol_tools, lightgbm, xgboost, transformers; torchvision 0.19.1 matched). tensorboard `notf`/protobuf errors are HARMLESS (chemprop falls back to CSVLogger).
- `python` (not `uv`) on the pod. Fine-tune writes to `/tmp/<plan>` (pod-local, off the 11G home quota); prune GPU zombies (`pkill -9 -f finetune`) if you hit CUDA OOM with a process holding ~23G.
