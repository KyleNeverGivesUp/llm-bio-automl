# COMPETITOR_ANALYSIS.md — what the PXR winners did, and what it means for our autonomous pipeline

> Post-competition analysis of the OpenADMET PXR Activity challenge (closed 2026-07-01). We read the
> published model reports of the top finishers (`assets/#1..#6`) after both label sets were unblinded.
> Purpose: (a) set an honest, proprietary-free target; (b) decide which capabilities our LLM-driven
> AutoML pipeline should gain; (c) supply the AAAI related-work / method-motivation / limitations sections.
>
> **Our own result for reference:** autonomous pipeline (CheMeleon + Uni-Mol, nnls stack) scored
> **Set-2 RAE 0.6311** (Set-1 judge 0.5706 was optimistic — see [foldin note](#validated-negatives-our-own-findings-confirmed)).

## 1. The final board (Set-2 = 260 blind molecules, the official final scorer)

| Rank | Team | Set-2 RAE | Proprietary data | One-line |
|---:|---|---:|:--:|---|
| 1 | matcha-croissant | 0.5631 | **Yes** | multitask GNN + proxy-SVR weak-compound specialist + regime blending |
| **2** | **AIDD-LiLab (main)** | **0.5676** | **No** | **#1 among public-data teams**; MF-pretrained Uni-Mol×2 + decorrelated heads + qHTS gate |
| 3 | AIDD-LiLab-Aggressive | 0.5692 | No | same, +440-retrain bet (lost by 0.0016, within noise) |
| 4 | N283T | 0.5703 | No | solo, consumer GPU; predicted-log2fc-as-feature + Caruana/TabPFN ensemble |
| 5 | toxicity | 0.5713 | Partial | Chemprop + TabPFN multitask; ensemble a >4 specialist with the overall-best |
| 6 | (tguttenb1) | 0.5741 | — | multi-representation ensemble + chronic over/under classifier (±0.2) + rescale |

**Public-data ceiling ≈ 0.5676** (AIDD-LiLab). The proprietary winner is only 0.0045 ahead, so
**proprietary data is worth ~0.0045 RAE — the gap is method, not data access.** Our 0.6311 is ~0.063
behind the public ceiling, i.e. a purely methodological gap that is closeable.

## 2. The convergent winning recipe (what ALL top teams did)

| Winning element | Teams | Our status |
|---|---|---|
| **Multi-fidelity (single-concentration) transfer — the #1 data lever** | all 6 | ⚠️ **severely under-used** (only one mt5 aux head) |
| **Error diagnosis → regime-targeted / gated correction** (tails) | all 6 | ❌ absent |
| **Calibration / variance-matching / de-compression** | #2 #4 #5 #6 | ❌ absent |
| **Decorrelation by construction** (diverse members, not accuracy) | all 6 | ✅ present, but only 2 members |
| **Counter-assay selectivity QC filter** | #2 #5 | ⚠️ have counter head, no filter |

### 2.1 Multi-fidelity single-concentration transfer (the biggest lever)
The challenge ships a cheap single-concentration `log2fc` readout for ~20K molecules (vs 4,139 dose-response
pEC50). Every top team exploited it, and it dominates:
- **#4 (rank 4):** predicting `log2fc` with ChemProp and using that **prediction as a feature** accounts for
  **73.7%** of the model's LightGBM gain — "one assay-derived signal does the work of a full descriptor stack."
- **#2:** "Multi-fidelity (MF) transfer is the single most important data lever" — pretrain on ~20K LF, fine-tune on pEC50.
- **#1 / #5 / #6:** single-concentration as an auxiliary multitask target / transferred embedding.

We only use single-concentration as **one head** in the mt5 template. We do **not** (a) pretrain the encoder
on it (MF transfer) or (b) use predicted-single-concentration as an input feature.

### 2.2 Error diagnosis → regime-targeted correction (universal, not #1-idiosyncratic)
Every team independently found the **same two-tailed error** (a regression-to-the-mean pattern), which our
own diagnosis reproduces exactly (weak `pEC50<4` over-predicted +0.87; strong `>6` under-predicted −0.93):
- **#1:** SVR-imputed proxy pEC50 for weak compounds + conditional blend at a `pEC50=4.5` threshold.
- **#2:** external qHTS classifier gate (pushes down over-predicted inactives, `pEC50<4` bias +0.56) **and** a
  Boltz MoE active-specialist (lifts the under-predicted active tail) — both confidence-gated.
- **#5:** models good at `<4` vs `>4` are **anti-correlated**; ensemble a dedicated `>4` specialist.
- **#6:** a "chronic over/under-prediction classifier" trained on OOF residual direction → capped ±0.2 correction
  (the most general, automatable form of the pattern).
- **#4:** tried tail gates, honestly reported most did not beat noise; calibration was the real gain.

### 2.3 Calibration / variance-matching
The test set is genuinely narrower than training (compression is correct, not a bug; #2 adversarial AUC≈0.97).
- **#4:** a single affine calibration was "the only clear gain" (MAE 0.44→0.41), covariate-shift reweighted.
- **#2:** variance-match every component + a `×1.08` scale tuned on the test-like subset.
- **#5:** datapoint weighting by blind-test Tanimoto similarity. **#6:** final distribution re-scaling.

## 3. Validated negatives (our own findings, confirmed by consensus)

- **Folding Set-1 into training does not help** — #2/#3 (aggressive +440 retrain lost by 0.0016, within noise),
  #4 ("mostly imported error"; "doing nothing would have been the right answer"). **Independently confirms our
  own experiment** (broad-only Set-2 0.6301 → fold-in 0.6628). See `scripts/foldin_set1_experiment.py`.
- **OOF / judge does not track the blind ranking** — #2 ("OOF is only a coarse validator"), #4 (repeatedly).
  Validates distrusting our Set-1 judge and the harness's scaffold-disjoint + multi-seed + error-bar design.
- **External data as appended labels hurts; as pretraining / gate it helps** — #2 ("wrong chemical region"),
  refines our M5a negative: the *usage mode* decides, not the data.
- **Docking / 3D structure / affinity is useless as explicit structure, only as a learned representation** —
  #1 (Boltz2/GNINA null), #2 (cofold helps only gated), #4 (Boltz pose/affinity too noisy; trunk embedding OK).
  Validates avoiding docking-as-structure.
- **Multi-seed averaging alone adds ~nothing** (#2: corr>0.97) — diversity must be by construction.

## 4. Implications for our autonomous pipeline (AAAI)

The winning recipe is **convergent and well-defined**, which sharpens our thesis to a crisp, testable question:
**can an autonomous LLM-driven pipeline discover and execute this recipe on its own?** The capabilities to build,
in priority order (all *after* the generalization floor — see `GENERALIZATION_PLAN.md`):

1. **Calibration layer** (cheapest, universal gain): affine / covariate-shift calibration on OOF → aggregator post-step.
2. **Multi-fidelity single-concentration transfer** (largest lever): pretrain-then-finetune, or predicted-single-conc
   as a feature. Data side coordinates with **DataMaster (Srivatsan)**; model side = a template/design change.
3. **Error-diagnosis → gated correction stage** (the 2nd contribution): an LLM stage that reads residuals, names the
   failure regime, and emits a capped correction — the autonomous form of what all 6 teams hand-built. Often needs
   codegen (the proxy-SVR was novel code), tying into the code-generator track.

**Framing for the paper.** Our pipeline is already competitive on the "pick decorrelated foundation models + stack"
axis. The winning margin came from moves our pipeline cannot yet perform autonomously (MF transfer, error-diagnosis
correction, calibration). Those three capabilities are the frontier of autonomous molecular AutoML, and building +
evaluating them *across tasks* (not PXR-only) is the paper's ambition beyond the generalization floor.

## 5. Sources
`assets/#1_openadmet_pxr_model_report.pdf` (matcha-croissant), `assets/#2.md` + `assets/#3.md` (AIDD-LiLab
main + aggressive), `assets/#4_MODEL_REPORT.md` (N283T), `assets/#5_toxicity model summary.pdf`, `assets/#6.md`.
Set-2 labels: `data/pxr_activity/phase2_unblinded.csv`. Our diagnosis: `scratchpad/error_diagnosis.py`.
