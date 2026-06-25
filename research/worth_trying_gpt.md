# Worth-Trying — GPT Deep Research report

> Source: GPT Deep Research (report written in Chinese).
> Character: most **detailed on the actual reproducible ladder**; surfaced RyeCatcher's full submission-version RAE progression and Antonio de la Vega de León's ablations. Distinctive contribution: the **step-by-step RAE ladder** + the **counter-assay weighting 0.81→0.58** data point + a concrete **3-pillar build + 4-tier submission plan**.

## The single most useful artifact: the RAE ladder (RyeCatcher, reproducible)
| Step added | RAE |
|---|---|
| XGBoost baseline | 0.7412 |
| → Butina-aware LightGBM | 0.7249 |
| → + Mordred + CheMeleon embedding | 0.6889 |
| → + **Chemprop multitask** | **0.6039** (big jump) |
| → + AutoGluon-CheMeleon + other paths | 0.5966 |
| → + ChEMBL NR1I2 head + layered blend | **0.586** |

> We sit ~0.62 (cluster-fold ensemble w/ CheMeleon embeddings) — between the 0.689 and 0.60 rungs. The CheMeleon fine-tune we're running ≈ the "Chemprop → 0.60" rung.

## Techniques worth trying (GPT's evidence-graded table)

| Priority | Technique | Why / evidence | Adoption for us |
|---|---|---|---|
| High | **Diverse ensemble + honest calibration** | RyeCatcher 0.7412→0.586; Antonio CheMeleon-only 0.574→ensemble 0.495 MAE; gain is from blend+calibration, not a bigger single model | have ensemble; **add honest isotonic** |
| High | **analog-aware / parent-cluster GroupKFold CV** | Antonio: scaffold split ≈ random here (scaffolds too fragmented); test set IS analogs | **have** (calibrated cluster folds) |
| High | **counter-assay weighting** | a rank-18 competitor: weight samples by primary-vs-counter CI overlap → **RAE 0.81→0.58** | new; easy via our sample_weight path |
| Med-high | **Chemprop/CheMeleon backbone + targeted PXR multitask external** (NCATS qHTS PXR, ChEMBL NR1I2) | RyeCatcher T1/T1v5 backbone | new; multitask heads |
| Med-high | **descriptor/tabular branch**: FCFP4-count + RDKit + Mordred + CheMeleon-2048d → LightGBM/AutoGluon | RyeCatcher v4/T2; Antonio RF/XGB/Macau/TabPFN useful as ensemble members | partly have; **add Mordred + FCFP4-count** |
| Med | **fold Phase-1 unblinded + HTChem into training** | format core; biggest single competitive lever this phase | ⚠️ conflicts with "Set 1 = judge" — end-game only |
| Med | **honest isotonic post-hoc calibration** | RAE penalizes systematic bias; all models show low/high-tail bias | new; cheap |
| Low-med | **big foundation models as ensemble members** | MolE/KPGT/SCAGE strong on generic benchmarks, but here likely redundant w/ CheMeleon | low priority |

## Negatives (evidence-backed — avoid)
- **scaffold split ≈ random split** here → must use parent/cluster CV (Antonio).
- **NN readout transfer to test = catastrophic** (test set has SAR misses).
- **Naive multitask dumping fails** — Antonio: piling organizer aux into Chemprop multitask didn't help, sometimes the control was better. What matters: *which* tasks share representation, head weights, distribution closeness.
- **Bigger/fancier single models (3D, image-ViT, TabPFN-as-separate-signal) don't add** — correlate with strong models, no new error mode.
- **Heavy HPO ~0.01 only** and overfits internal holdout (matches our tuner finding).
- **Broad external (BindingDB, broad ChEMBL pretrain) didn't help**; only PXR-mechanism-close external helps.
- **Don't chase the high-potency tail with custom losses** — data ceiling (few ≥6.5 compounds).

## General SOTA worth knowing (NOT verified here)
- **MolE** (Nature Comms, ~842M-molecule pretrain): #1 on 10/22 TDC ADMET tasks.
- **KPGT** (graph transformer + knowledge-guided pretraining, ~2M molecules): strong on 63 property datasets — quality of pretraining objective > raw SMILES-LM scale.
- **SCAGE** (~5M compounds, explicitly tested on activity cliffs; learns fingerprint + functional groups + 2D distance + 3D angles jointly) — most cliff-relevant.
- **Chemical-LM scaling study**: PXR RAE 0.7468 (170M) → 0.6606 (1.3B) but still ≫ a tabulated SOTA 0.5633 → bigger SMILES-LM alone insufficient.

## GPT's concrete build (3 pillars + 4 submission tiers)
**Pillars:** (1) Chemprop/CheMeleon multitask backbone · (2) descriptor/tabular branch (FCFP4-count + Mordred + CheMeleon-2048d → LightGBM) · (3) validation+calibration system (Butina/parent GroupKFold + honest per-fold isotonic).
**Tiers:** baseline (single-task Chemprop + LGBM + mean) → strengthened (+counter-assay weight, GroupKFold, isotonic) → contender (+ChEMBL NR1I2/NCATS multitask, CheMeleon embeddings, layered blend) → defensive (retrain on Phase-1-unblinded + HTChem; keep 2–4 stable decorrelated members).

## Honest caveats (from the report)
- Final leaderboard not out; official RAE per-rank distribution not public (interim board shows MAE).
- No reproducible **top-10** method report found — top-tier recipes inferred from RyeCatcher (~rank 40) + Antonio + ExpansionRx, i.e. informed hypotheses.
