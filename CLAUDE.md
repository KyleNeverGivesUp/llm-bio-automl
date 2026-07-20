# CLAUDE.md — project context (auto-loaded every session)

LLM-driven AutoML for the **OpenADMET PXR Activity Prediction** challenge. This file is the
handoff/state doc — read `RESULTS.md` for the full numeric log and `docs/` for design docs.

## ⛔ WORKING RULE — no independent decisions (read first, every time)
The user does the deciding; I execute exactly. **I must NOT make my own methodology/scope choices.**
- **When reproducing a reference solution (e.g. `reproduce_1/` = the #1 PXR solution): implement ONLY
  what the source report literally states, one-to-one.** Do NOT skip a technique, reduce its scope
  (e.g. 3 models when the report says 5), substitute a model/data source, add a step the source does
  not have (calibration, variance-matching, Set-1 weight tuning), or tune anything on the blind test
  (Set-2). Match the source exactly.
- **At EVERY point the source is silent or ambiguous — blend weights, which backbone, which
  descriptors, hyperparameters, data substitutions — STOP and ASK before writing any code.** Do not
  pick a default myself.
- **MECHANICAL CHECK (the abstract rule above keeps failing because I mis-file methodology choices as
  "mechanical coding"): if the source states a QUANTITY (a count like "~30 compounds", "5 models",
  "4 concentrations"; a threshold like "4.5"), my code MUST reproduce that number — CHECK it after
  coding. A mismatch = I diverged = STOP; do NOT rationalize it as "close enough / same order / more
  principled". Any threshold/cutoff the source does not literally give = picking a number = a decision
  = ASK first. (Cost the user ~1h: report said filter "~30", I removed 51.)**
- **Never propose "let me skip / just do a smaller version / I recommend dropping X" as a way to save
  effort.** If I catch myself doing that, stop and ask instead. Do not conclude a method "doesn't work"
  from an incomplete reproduction — finish it faithfully first.
- The user has repeatedly (and angrily) corrected violations of this. Treat it as a hard gate.

## STATUS (2026-07-01): competition over → pivoting to AAAI publication
- **The challenge closed July 1 23:59:59 UTC. We MISSED the final leaderboard submission** (timezone
  mix-up + a full day lost to GPU/OOM debugging). The research + results are intact and valid; only the
  leaderboard entry was missed. A submission file was built (`predictions/final_submission_0.5783_foldedlabels.csv`,
  Set-1 true labels + Set-2 predictions) but not uploaded in time.
- **Prof Pengtao Xie approved targeting AAAI** (next deadline ~Aug 2026). Goal is now a **publication**,
  not a leaderboard rank. See `docs/GENERALIZATION_PLAN.md` for the AAAI work plan (generalization →
  ablations → baselines → stats → DataMaster).
- **Lab context:** this project is an application/extension of the group's AIBuildAI/MLEvolve line
  (MLE-Bench). Differentiator = molecular-domain + autonomous discovery. **DataMaster is being built by
  Srivatsan — coordinate before duplicating it.**

## Current best (Set-1 judge)
- **Hand-assembled best: RAE 0.5706 / MAE 0.4557** — `predictions/` (`oof_unimol.csv` + `test_unimol.csv` stacked with mt5). Ridge stack of **CheMeleon (graph, fine-tuned) + Uni-Mol (3D, fine-tuned)**. ≈ leaderboard **rank ~20/328**.
- **Fully-autonomous Architecture-B run (2026-06-30, real GPU end-to-end): RAE 0.5783 / MAE 0.4619** — nnls stack of chemeleon (single judge 0.5887, weight 0.716) + unimol (single judge 0.6248, weight 0.273). Same thesis, produced with **zero human model-picking** (setup→retrieve→run→stack all LLM-driven, `skill-source=llm`). Artifacts: `~/Downloads/5783/` / DSMLP `outputs/skill_manager/`.
- Progression: 0.6329 → 0.6217 → 0.6135 → 0.6108 → 0.5916 → **0.5706** (hand); **0.5783** (autonomous).
- The **frozen path** (embeddings + sklearn, no fine-tuning) only reaches **RAE ~0.62-0.77**. The gap is **fine-tuning** — verified this run: same chemeleon is 0.7132 frozen vs 0.5887 fine-tuned.

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
- **Cold-start discovery REPRODUCED (2026-07-01):** with the registry+seed emptied, retrieve autonomously re-surfaces **CheMeleon** (post-LLM-cutoff) from recent arXiv abstracts → then locates weights (`CheMeleon Foundation Model` on Zenodo + openadmet baselines on HF). Honest boundary: discovery works + is probabilistic (LLM ranker didn't always *select* it; arXiv can 429/timeout); usability-as-finetune is still seed/template-encoded. Test: `scratchpad/coldstart_test.py`.
- **unimol batch: 32 is the validated value, NOT 64.** On the full data, batch 64 OOMs a 24G A5000 (~23G, a large-molecule attention batch; proven on a clean GPU — not a leftover-VRAM issue). batch 32 ≈ 12G fits and reproduces the validated single 0.6248. The `--batch, default=64` "raised from 32" note was an untested bump. chemeleon (D-MPNN, linear memory) is fine at 128. Each fine-tune runs as its own subprocess → its VRAM is freed on exit automatically (verified: `GPU free before unimol: 24247 MiB`).
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
- GPU: DSMLP A5000 pod (`launch.sh -g 1 -v a5000 -m 64 -c 8`), **6h session limit**. **Always pass `-c 8`**
  — without it the pod is CPU-throttled to ~1 core, which starves chemprop/TabICL dataloading and made
  every run ~4× slower (a big time sink until 2026-07-20). With `-c 8` the pod gets the full node's cores., ~11G home quota (TIGHT — auto-prune checkpoints/.sdf after extracting OOF/test). Models are small (~10M params); A5000 is plenty, no A100 needed.
- Weights auto-download on first run: CheMeleon (chemprop→Zenodo `~/.chemprop`), Uni-Mol (unimol_tools→Zenodo), ChemBERTa (transformers→HF). **Do not commit weights** — DSMLP fetches them.
- Local dev: `uv run python ...`. Fine-tuning needs the pod GPU.

## Next steps — AAAI publication track (see `docs/GENERALIZATION_PLAN.md`)
1. **De-hardcode the pipeline** — pass setup's inferred task/target/metric through ALL prompts + the fine-tune template's target columns. Currently ~5 prompts hardcode "molecular pEC50 regression / RAE" and `finetune_cheme_mt5.py` hardcodes the 5 PXR assay columns (`TARGETS`). Also PXR-specific: `setup._FALLBACK`, `JUDGE_FILE`, `folds_calibrated.json`. Full audit + the fix list live in `docs/GENERALIZATION_PLAN.md`.
2. **Generalization (the #1 experiment)** — run the SAME pipeline on ≥2 more molecular property tasks (start: **Lipophilicity + ESOL**, MoleculeNet regression, scaffold split) to show it isn't PXR-tuned.
3. **Ablations** — LLM vs fixed setup; LLM-rank vs random model selection; fine-tune vs frozen; decorrelation vs not; literature-discovery vs not.
4. **Baselines** — non-LLM AutoML (auto-sklearn / FLAML), random selection, domain SOTA (published CheMeleon/Uni-Mol/D-MPNN numbers).
5. **Statistical rigor** — multiple seeds/runs + error bars (run-to-run: 0.5706 hand vs 0.5783 autonomous).
6. **DataMaster** — data-side autonomous twin (external-data sourcing + judge-gated integration). **Coordinate with Srivatsan (he owns the DataMaster codebase).**

## Note: the WORKING pipeline is Architecture B = `src/agent/skill_manager.py`
`run_skill_manager.py` runs it end-to-end: setup → retrieve → run (finetune via template / frozen via featurizer) → stack (forward-selection + nnls). This is what produced the autonomous 0.5783. (`manager_agent.py` is the older menu-loop orchestrator.) Flags: `--fast` (smoke: 2 folds/tiny epochs), `--no-fallback` (strict: every LLM-decision re-raises instead of using a hardcoded default — proves LLM-driven), `--collect-only` (reuse predictions/, no GPU).

## Running on DSMLP
- SSH: `~/.ssh/config` has `dsmlp` (ControlMaster → Duo once/8h). Pod: `launch.sh -g 1 -v a5000 -m 64 -c 8`, 6h limit.
- env deps in `~/.local` (torch 2.4.1+cu121, chemprop, unimol_tools, lightgbm, xgboost, transformers; torchvision 0.19.1 matched). tensorboard `notf`/protobuf errors are HARMLESS (chemprop falls back to CSVLogger).
- `python` (not `uv`) on the pod. Fine-tune writes to `/tmp/<plan>` (pod-local, off the 11G home quota); prune GPU zombies (`pkill -9 -f finetune`) if you hit CUDA OOM with a process holding ~23G.
