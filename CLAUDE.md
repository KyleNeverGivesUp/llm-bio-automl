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
**Our pipeline is a stripped-down AIBuildAI:**
- `coder_agent` only EXECUTES fixed plans (`run_plan_cv`) — it does NOT generate code (AIBuildAI's coder writes training code).
- `tuner_agent` only tunes params — it does NOT fine-tune foundation models on GPU (AIBuildAI's tuner does).
- So the pipeline can only combine **8 frozen featurizers × fixed sklearn models** — "AutoML over a fixed menu", not feature engineering or fine-tuning.

**The rank 84 → 20 gap is exactly the fine-tuning capability we left out.** Open design question (asked the prof): bring LLM-orchestrated fine-tuning into the pipeline (LLM writes the training config/script, GPU trains) — that's where the performance is.

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
1. **Fine-tuning into the pipeline** (prof's open question) — make the coder generate fine-tune configs from verified templates + the tuner launch GPU training. Running the pipeline ON DSMLP (next to the GPU) makes this cleaner.
2. **End-game Set-1 fold-in** (~0.10 on singles, irreversible — consumes the judge) for the final Set-2 submission. Do LAST.
3. **Methodology report** (July 1, mandatory).
4. (optional) DataMaster = data-side autonomous twin of the model-retrieval component (paper completeness, low score value).
