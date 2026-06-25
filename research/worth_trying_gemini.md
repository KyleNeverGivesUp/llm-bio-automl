# Worth-Trying — Gemini Deep Research report

> Source: Gemini Deep Research, "Strategic Benchmarking and Advanced Modeling for the OpenADMET PXR Induction Blind Challenge."
> Character of this report: broad/strategic; grounds many claims in the *prior* ExpansionRx challenge + cheminformatics literature rather than the actual PXR top solutions. Treat its rankings as informed hypotheses. Distinctive contribution: **RIGR data augmentation** + the activity-cliff mechanism explanation.

## Techniques worth trying (Gemini's ranking)

| # | Technique | Why (per report) | Evidence | Adoption for us |
|---|---|---|---|---|
| 1 | **Descriptor foundation model + hybrid ensemble** — CheMeleon/Uni-Mol2 embeddings **concatenated with explicit descriptors** (MACCS, Maplight, Jazzy), fed to XGBoost/LightGBM, then ensemble-averaged across modalities | "single most determinative factor"; top teams (pebble #1, moka #3 in ExpansionRx) all did this | strong (cross-challenge) | **mostly have it** (CheMeleon+desc+GBDT+stack); add Uni-Mol2 only if decorrelated |
| 2 | **RIGR — Resonance-Induced Graph Representations** — enumerate Kekulé/resonance forms of each molecule → different initial GNN graphs, same target → 2–3× effective data. (SMILES randomization does NOT help GNNs.) | rced_nvx cited it as "single most determinative factor" for them | weak (1 competitor) | **new to us**; medium effort. ⚠️ all resonance forms of a val molecule MUST stay in one fold or it leaks |
| 3 | **Target-aware CV** — Taylor-Butina clustering / Bemis-Murcko scaffold / temporal sliding-window splits; force whole clusters into one fold | random splits leak analog series → optimistic CV that collapses on blind set | strong | **have it** (calibrated Butina cluster folds). Could add temporal/ID sliding-window |
| 4 | **Counter-assay multi-task learning** — output heads predict primary pEC50 + primary Emax + counter pEC50 + counter Emax simultaneously | forces shared layers to separate true binding vs assay interference; ablations show removing aux degrades primary | medium | **new to us**; the multitask Chemprop lever |
| 5 | **External target-specific data, staged fine-tuning** — pretrain on broad/noisy ChEMBL PXR (~800), then fine-tune on OpenADMET | maps global PXR rules before micro-SAR | medium | conflicts with other reports (broad external "only hurt"); use **only** targeted PXR data, as aux head |

## Negatives Gemini flags (avoid)
- **Zero-shot generic ADMET predictors** (ADMETlab 3.0, ADMET-AI) fail — covariate shift too severe.
- **Pure un-ensembled Chemprop plateaus** mid-pack — descriptor concatenation is "non-negotiable."
- **Assay artifacts**: solubility ceilings, varying screen concentrations (10/30/100 µM), HTChem yield corrections — QC before trusting numbers.
- **Leaderboard chasing** on Analog Set 1 → catastrophic regression on Set 2. Trust offline cluster-CV.

## Caveats specific to this report
- The "top-5 needs RAE < 0.55" threshold is **inferred from ExpansionRx MA-RAE** (pebble 0.5113 / shin-chan 0.5536 / temal 0.5809), not PXR.
- The interim top-10 table it shows is **MAE** (0.400–0.426), not RAE.
- Overall it's the **least grounded in actual PXR solutions** of the three — better for the *mechanism* (why GNNs fail on cliffs) and the **RIGR** idea than for concrete winning recipes.
