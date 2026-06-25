# Worth-Trying — MERGED & prioritized (Gemini + Claude + GPT)

> Synthesis of three deep-research reports on the OpenADMET PXR Activity challenge, deduped, ranked by **(evidence strength × ROI for us)**, and mapped to our current pipeline (~0.62 Set-1-judge RAE: fingerprints/descriptors/ChemBERTa/**CheMeleon embeddings** + GBDTs + ridge-stack on **calibrated Butina cluster folds**; CheMeleon fine-tune running).
> Per-report detail: [gemini](worth_trying_gemini.md) · [claude](worth_trying_claude.md) · [gpt](worth_trying_gpt.md).

## The consensus winning recipe (all 3 agree)
**Decorrelated ensemble of (a) a multitask Chemprop D-MPNN initialized from CheMeleon + (b) GBDTs on FCFP4-count + Mordred + CheMeleon-2048d embeddings, blended with non-negative weights on OOF, validated with parent-cluster / Butina GroupKFold CV, and post-hoc honest per-fold isotonic calibration.** Reaching the lead pack is about *variance reduction + not fooling yourself*, not a novel model — the top ~17 are a statistical tie.

### Reproducible reference (RyeCatcher) RAE ladder — our map
`XGB 0.741 → Butina-LGBM 0.725 → +Mordred+CheMeleon 0.689 → +Chemprop multitask 0.604 → +AutoGluon-CheMeleon 0.597 → +ChEMBL-NR1I2 head 0.586`. **We are ~0.62; the next documented rung is multitask Chemprop.**

## Ranked techniques (merged)

| Rank | Technique | Reports | Evidence | We | Effort |
|---|---|---|---|---|---|
| 1 | **Honest per-fold isotonic calibration** (fit on OOF only) | C, G(pt) | strong — "the gate"; in-sample is optimistic ~0.01 | ❌ missing | **low** |
| 2 | **Multitask Chemprop** (primary pEC50 + counter pEC50/Emax; +targeted PXR external) — done *right* (head weights, masked NaNs, distribution-close tasks) | all 3 | strong — the documented 0.689→0.60 jump | ❌ (running single-task) | med |
| 3 | **counter-assay sample weighting** (down-weight high-signal-in-both = interference) | C, GPT | strong — one competitor 0.81→0.58 | ❌ (have sample_weight path) | **low** |
| 4 | **Parent-cluster / Butina GroupKFold CV**, and verify **CV↔LB tracks within ~0.02** | all 3 | strong — scaffold split ≈ random here | ✅ have folds; ⚠️ not yet checked tracking | low |
| 5 | **Add Mordred + FCFP4-count features** to the GBDT branch (count > binary fingerprints) | all 3 | strong | ⚠️ have rdkit_desc/morgan-binary | low |
| 6 | **CheMeleon fine-tune / embeddings as backbone** | all 3 | strong | ✅ doing | — |
| 7 | **Drop any model with OOF-corr > 0.9 vs current blend** (prune redundant members) | C | strong — validates dropping MolFormer | partial | low |
| 8 | **Classical SVR/XGB on ECFP4** as a decorrelated cliff hedge | C | medium (MoleculeACE) | easy add | low |
| 9 | **RIGR resonance-graph augmentation** (2–3× data for the GNN; keep all forms in one fold) | G | weak (1 competitor) | new | med |
| 10 | **Phase-2: fold Set-1 + HTChem into final training** | C, GPT | format core | ⚠️ conflicts with "Set 1 = judge" | low |

## Strong negatives — DO NOT spend time on (all/most reports agree)
- **NN label transfer** to the test set → catastrophic (0.45→0.66). The test set has "SAR misses."
- **Naive aux-data dumping** (extra pEC50 rows / piling many multitask heads) → no gain or worse. *(matches our M5(a).)* Use targeted, distribution-close aux as multitask heads only.
- **Broad external pretrain** (BindingDB / broad ChEMBL) → didn't help. Only PXR-mechanism-close external helps. *(matches our "skip DataMaster".)*
- **Standalone SMILES transformers / 3D big models / image-ViT / TabPFN-as-separate-signal** → redundant (>0.88 corr). *(validates dropping MolFormer; ChemBERTa marginal.)*
- **Tail-weighted / quantile losses** → can't beat the data ceiling (≈11 train compounds ≥ pEC50 6.5).
- **Heavy HPO** → ~0.01 only, overfits internal holdout. *(matches our tuner finding.)*
- **Zero-shot ADMET-AI / ADMETlab 3.0** → negative R².

## What this VALIDATES about our work
scaffold-CV useless → cluster folds ✅ · MolFormer dropped ✅ · HPO marginal ✅ · naive aux rows hurt (M5a) ✅ · broad external skip (DataMaster) ✅ · CheMeleon is the lever ✅.

## Highest-ROI next actions (after the fine-tune returns)
1. **Honest per-fold isotonic calibration** on the final ensemble OOF (cheapest real RAE win).
2. **counter-assay weighting** (reuse `sample_weights.py`; needs the counter-screen columns).
3. **Multitask Chemprop** (primary + counter; then targeted PXR external) — the documented 0.60 rung; biggest but heaviest.
4. **Add Mordred + FCFP4-count** to the GBDT features; **prune OOF-corr>0.9 members**.
5. Verify **CV↔LB tracking** (our calibrated folds vs the Set-1 judge) is within ~0.02 — the discipline gate.

## Honest caveats (all reports)
- **Actual top-5 recipes are NOT public** (method reports release after Jul 1, 2026). Everything above is inferred from RyeCatcher (~rank 40), Antonio, and the prior ExpansionRx challenge → **informed hypotheses, not confirmed winning recipes.**
- Official **RAE per-rank distribution is not public** (interim board shows MAE 0.400–0.426; our brief's 0.528–0.538 RAE is a different cut). "Top-5" ≈ indistinguishable cluster around RAE ~0.50 (live, half-test).
- **Tension with our discipline:** the reports' Phase-2 move folds Set 1 into training; we've kept Set 1 strictly as the judge. That's an end-game decision, not yet taken.
- **Deep-research quality differed:** Claude/GPT grounded in real PXR solutions (RyeCatcher, Antonio); Gemini leaned generic/ExpansionRx-inferred. (Useful CSE190 data point.)
