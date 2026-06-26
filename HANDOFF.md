# HANDOFF — session narrative & decisions

Richer context than `CLAUDE.md` (which has the distilled state). Read both + `RESULTS.md`.
This captures the *reasoning* from the session that got us to RAE 0.5706, so a fresh Claude
(e.g. on DSMLP) can pick up where we left off without re-deriving.

## How we got to 0.5706 (the decorrelation story)
We were stuck at ~0.61 with a 48-base ensemble (all CheMeleon-flavored frozen/fine-tuned variants).
The insight: that ensemble was **saturated** — every new member was correlated, so stacking added
nothing (single ≥ ensemble). The fix was **decorrelated model families**, mirroring the 0.538
competitor (discoverybytes, an AI-generated writeup) who stacked graph + 3D + foundation:
- **mt5** = CheMeleon fine-tuned, multitask (pEC50 + counter + Emax + **single_concentration**) + **MAE loss** + reactive-electrophile exclusion → single **0.5904**.
- **Uni-Mol** = 3D transformer fine-tuned → single **0.6248**, corr with mt5 = **0.866** (decorrelated).
- Ridge stack of just these two → **0.5706**, beating the 48-base ensemble. Two decorrelated strong models > many correlated ones.

## Decisions made (and why) — don't re-litigate
- **TTA (aug10) is OFF.** It improved Uni-Mol's single (0.6248→0.6177) but HURT the stack (0.5706→0.5765): averaging 10 SMILES smooths predictions toward consensus → more correlated with mt5 → less complementary. Verified, negative.
- **Dropped IBM multi-view / GROVER / MolCLR as 3rd leg.** All are dependency-archaeology (old torch/python, `fast_transformers` won't build, `torch_scatter` version hell). MolE (Recursion) weights are withheld (code-only). ~0.01-0.02 uncertain gain for hours of build pain + would've required deleting the working env. Not worth it. The retrieval component + `skills/models/candidates_live.json` record these so we don't re-discover them.
- **AIBuildAI realization (key for the report):** our pipeline is a stripped-down clone of AIBuildAI (arXiv 2604.14455, #1 MLE-Bench) — same manager/designer/coder/tuner. AIBuildAI's coder GENERATES training code and its tuner FINE-TUNES on GPU; ours only executes fixed plans + tunes params. The **rank 84 → 20 gap = the fine-tuning capability we left out.**
- **Asked the prof:** "pipeline ≈ RAE 0.62 / rank 84; manual fine-tuning ≈ 0.57 / rank 20 — should we bring LLM-orchestrated fine-tuning into the pipeline?" (LLM writes the training config/script, GPU trains — not the LLM training the net.) Awaiting answer.

## What "fine-tuning into the pipeline" means (the plan)
A = make fine-tuning a first-class plan type the designer can propose `{backbone, targets, loss, epochs}`.
B = coder instantiates a verified TEMPLATE (`finetune_cheme_mt5.py` / `finetune_unimol.py`) from the plan (template-based codegen = safe, leak-free, vs raw codegen).
C = tuner launches the GPU job + collects OOF/test (the hard part — GPU orchestration; **running the pipeline ON DSMLP next to the A5000 makes this trivial — no remote ssh/scp**).
D = stack via aggregator + validate via `analog_judge`.
A+B+D are Mac-doable now; C is why we're moving the pipeline onto DSMLP.

## DSMLP / infra state
- Repo: `github.com/KyleNeverGivesUp/llm-bio-automl` (push the session work first; small data force-added, weights/caches excluded).
- SSH: `~/.ssh/config` has a `dsmlp` host (dsmlp-login.ucsd.edu, ControlMaster so Duo is once/8h). `ssh dsmlp`.
- VS Code: Remote-SSH to `dsmlp` (login node) for editing — shared NFS = same files the pod sees; run on GPU via integrated terminal `kubectl exec -it <pod> -- bash`.
- Pod: `launch.sh -g 1 -v a5000 -m 64`, 6h limit, ~11G quota (TIGHT). Weights auto-download. IBM dropped, so `~/.local` (torch/chemprop/unimol) should be kept/rebuilt for fine-tuning + the fold-in.

## Open threads / next
1. ✅ **Fine-tuning in the pipeline — DONE** (prof said "yes, promising"). LLM-orchestrated end-to-end:
   `finetune_designer` (LLM picks decorrelated backbones) → `finetune_runner` (plan→GPU cmd→collect)
   → `run_finetune_auto.py` → stack → judge → **0.5706**. Boundary: LLM selects from a fixed backbone
   list {chemeleon, unimol}; training code is verified templates (not free codegen). See `docs/ARCHITECTURE.md`.
   - Remaining: wire `finetune_designer` into `manager_agent` (native loop, not standalone script);
     wire `hf_retrieval` into `retrieval_agent` (live model discovery instead of static manifest).
2. End-game **Set-1 fold-in** (~0.10, irreversible) for the final Set-2 submission — do LAST.
3. **Methodology report** (July 1).
4. (optional) DataMaster = data-side autonomous twin of the model-retrieval component.
