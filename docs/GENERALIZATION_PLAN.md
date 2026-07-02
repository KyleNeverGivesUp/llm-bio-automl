# GENERALIZATION_PLAN.md — AAAI experiment #1 (generalization)

> **Goal.** Show the LLM-driven AutoML pipeline (Architecture B = `src/agent/skill_manager.py`) is a
> **general molecular-property-prediction system, not tuned to PXR**. Run the *same* pipeline (no
> per-task code changes) on ≥2 more molecular regression tasks and show it autonomously reaches
> competitive results on each. This is the #1 requirement for an AAAI submission.
>
> Prof: target **AAAI (~Aug 2026)**. Differentiator vs the lab's MLE-Bench line = molecular domain +
> autonomous (post-cutoff) model discovery.

## The 2 tasks (+ 1 optional) — MoleculeNet regression, standard & well-benchmarked

| Task | Property | # mols | Metric | Why this one |
|---|---|---|---|---|
| **PXR** (done) | pEC50 | 4,139 | RAE | our result: **0.5783** autonomous / 0.5706 hand |
| **① Lipophilicity** | logD (octanol/water) | ~4,200 | RMSE | ~same size as PXR; **CheMeleon + Uni-Mol both have published numbers** → easy SOTA compare; scaffold split is standard |
| **② ESOL** | aqueous solubility logS | ~1,128 | RMSE | canonical, small, fast; every molecular paper reports it |
| (opt) FreeSolv | hydration free energy | ~642 | RMSE | tiny/fast; adds breadth (3 sizes, 3 properties) |

Why these: all **regression + SMILES→scalar** (like PXR), so the fine-tune templates + metrics transfer
with minimal change; all have **published SOTA** (CheMeleon, Uni-Mol, D-MPNN/Chemprop) for baselines;
**scaffold split** is the field standard (reproducible). Source: DeepChem `MoleculeNet` loaders (or TDC).
Test labels are public → the "judge" is just the scaffold test split scored with the task metric (no
`phase1_unblinded.csv` equivalent needed).

## Prerequisite: de-hardcode the pipeline (make it task-general)
Today ~5 prompts + the fine-tune templates hardcode "PXR / pEC50 / RAE". Parameterize everything from
**setup's inferred task** (setup already infers task/target/metric — the downstream just ignores it):

1. **Prompts** — `skill_manager` (manager prompt + the "Predict pEC50 (metric RAE)…" skill desc),
   `finetune_designer.py:63`, `menu_designer.py:127`, `menu_tuner.py:83`: replace literal
   `"molecular pEC50 regression (metric RAE)"` with `{task_summary}` / `{metric}` injected from setup.
2. **chemeleon template** (`finetune_cheme_mt5.py` `TARGETS`) — use **single-target** from setup
   (a plain `chemprop train --from-foundation CheMeleon --target-columns <col>`), not the 5 PXR assays.
   The multitask-5 variant stays as a PXR-specific option.
3. **unimol template** (`finetune_unimol.py`) — target column from setup, not hardcoded `pEC50`.
4. **Judge** (`src/analog_judge.py`) — parameterize the metric (RMSE / MAE / RAE by config).
5. **setup `_FALLBACK` / `JUDGE_FILE` / `folds_calibrated.json`** — make generic: scaffold folds
   generated from the data; drop the PXR fallback (or make it a per-task config).

## Task list (phases)
- [ ] **Phase 0 — de-hardcode** (the 5 points above). ~1–2 days. Deliverable: pipeline runs on an
      arbitrary `data/<task>/` with a brief, no PXR strings.
- [ ] **Phase 1 — data prep**: for Lipophilicity + ESOL, download via DeepChem, write
      `data/<task>/{train.csv,test.csv,folds.json}` (scaffold split) + a short `brief.md`. ~0.5 day.
- [ ] **Phase 2 — run**: `run_skill_manager.py` end-to-end on each task (PXR + Lipo + ESOL). Capture:
      which models retrieve selected, per-model + stack metric, full logs. ~1 day GPU.
- [ ] **Phase 3 — compare**: table of *our autonomous result* vs published SOTA (CheMeleon / Uni-Mol /
      D-MPNN) on each task. ~0.5 day.
- [ ] **Phase 4 — write up** the generalization section (same pipeline, 3 tasks, competitive on each).

## Solution — how each task runs through the pipeline (post de-hardcode)
```
data/<task>/  (train.csv, test.csv, folds.json, brief.md)
   → run_skill_manager.py --data-dir data/<task>
   → setup(LLM): infers task/target/metric from brief.md
   → retrieve(LLM): searches molecular models (same families) → selects a few decorrelated
   → run: fine-tune chemeleon+unimol (single-target) + frozen others
   → stack: forward-selection + nnls
   → judge: score test predictions with the task metric
```

## Success criterion
The **same pipeline, no per-task code**, runs on PXR + Lipophilicity + ESOL and lands **within ~5–10%
of published SOTA** on each — evidence it generalizes rather than being PXR-tuned. (Ablations + baselines
are the next AAAI experiments; see `CLAUDE.md` → "Next steps".)

## Coordination
**DataMaster (data-side agent) is owned by Srivatsan** — align before building it. This plan is the
*model-side* generalization; the data-side twin is a separate track.
