# Worth-Trying — Claude Opus Deep Research report

> Source: Claude Opus Deep Research, "Cracking the OpenADMET PXR Induction Blind Challenge — A Top-5 Playbook."
> Character: most **evidence-graded** of the three; clearly separates "evidence-backed (this competition)", "negative results", and "general SOTA (not verified here)". Distinctive contribution: surfaced the **RyeCatcher reproducible pipeline** + **honest per-fold isotonic calibration** + the **>0.9-correlation drop rule**.

## Techniques worth trying (Claude's ranked list, by evidence of RAE impact)

| # | Technique | Why (per report) | Evidence | Adoption for us |
|---|---|---|---|---|
| 1 | **Parent-cluster LOCO CV** — group by FCFP4 nearest-neighbour parent cluster, leave-one-cluster-out; mirrors "513 analogs of ~89 parents" | random folds leak parent series; the gate that makes CV track LB | strong (RyeCatcher) | **have** Butina cluster folds; refine toward parent-cluster LOCO |
| 2 | **Honest per-fold isotonic calibration** — fit IsotonicRegression on out-of-fold preds ONLY (never in-sample) | in-sample isotonic optimistic ~0.009 RAE; the "gate that prevented submitting overfit garbage"; only protocol whose CV↔LB matched | strong | **NEW to us**; trivial compute, high ROI for the RAE metric |
| 3 | **Decorrelated ensemble: multitask D-MPNN (Chemprop) + GBT** on FCFP4-count + Mordred + CheMeleon-2048d | "a 300-param Chemprop blended with a 4,900-feature LightGBM beat single models 10× larger" — error modes decorrelated | strong | **partly have**; add multitask + Mordred + FCFP4-count |
| 4 | **CheMeleon-pretrained Chemprop** (`--from-foundation CheMeleon`) or CheMeleon 2048-d as tabular features | foundation embeddings dominate regardless of learner | strong | **doing now** (fine-tune running) + have embeddings |
| 5 | **Multitask aux heads from external PXR data** — ChEMBL NR1I2 (~907), NCATS qHTS PXR (AID 1346982/1346985), Tox21 SR-ARE, counter-screen | "added measurable lift even though individual heads had Spearman ≤ 0.4" — lift from shared representation, **NOT** label correlation | medium | new; **as multitask heads only**, never as rows/NN-transfer |
| 6 | **Counter-screen / PXR-null exploitation** — flag/down-weight compounds with high signal in both assays (= assay interference) | organizer-recommended; mechanistic FP filter | medium | new; pairs with our sample-weight path |
| 7 | **Phase-2 reactivity** — fold unblinded Analog Set 1 + HTChem into training before predicting Set 2 | final standing is Set 2 only | format-specific | ⚠️ **conflicts with our "Set 1 = judge, never train" discipline** — end-game decision |
| 8 | **Classical SVR/XGBoost on ECFP4** as a decorrelated cliff hedge | MoleculeACE: descriptor/FP ML beats DL on cliffs | medium | easy; cheap ensemble member |

## Negative results (evidence-backed — avoid)
- **NN label transfer = catastrophic**: RyeCatcher v50 OOF 0.4536 → LB 0.658 (rank 87). Never copy a near-neighbour's assay readout (test set has "SAR misses").
- **2D rep space saturated**: ChemBERTa, GIN, UniMolV2-310M, MaskMol, image-ViT, TabPFN-on-CheMeleon all correlate **>0.88–0.93** with the simple blend → no orthogonal signal. → **drop a model if OOF-corr > 0.9** with current blend.
- **Tail-weighted/quantile losses can't fix a data ceiling**: only 11 training compounds ≥ pEC50 6.5.
- **Naive random-split CV misleads**; **zero-shot ADMET-AI/ADMETlab give negative R²**; below ~0.4 Tanimoto NN similarity to public data there's a ~0.27 "noise floor."

## General SOTA worth knowing (NOT verified on this competition)
- **MoleculeACE** (JCIM 2022): on activity cliffs, **descriptor-based ML > complex DL**; ECFP bit-level sensitivity is the key inductive bias.
- **CheMeleon** (arXiv 2506.15792): 97% win on MoleculeACE vs RF 63% — but authors caution it still "struggles to distinguish activity cliffs."
- **GraphCliff** (arXiv 2511.03170): short/long-range gating to recover ECFP-level local sensitivity in a GNN — promising, unproven at competition scale.
- GBT: XGBoost ~best accuracy, LightGBM near-equal & far faster; CatBoost competitive. TabPFN v2 competitive out-of-box but added no orthogonal signal here.

## Honest caveats (from the report)
- Final leaderboard unpublished (closes Jul 1, 2026). The 0.586 RyeCatcher pipeline is **rank ~40**, not top-5; actual top-5 methods are not public yet.
- Live #1 RAE ≈ 0.496 is **self-reported**, not official. RAE per-rank distribution not public → "top-5" = indistinguishable cluster ~0.50 (live, half-test).
