# reproduce_1 — faithfully reproduce the #1 PXR solution (matcha-croissant)

**Goal:** reproduce the 1st-place PXR solution's RAE as closely as possible **using public data
only** (they used some proprietary data; we substitute public equivalents and flag it). This is a
self-contained reproduction, kept separate from the main pipeline (`src/`, `scripts/`).

Scoring is post-competition on the now-public labels: Set-1 (253) is dev, **Set-2 (260) is the final
scorer, scored once**. References: #1 (matcha, proprietary) **0.5631**, #2 (best public-data)
**0.5676**, rank-12 0.586.

## #1's method — a meta-ensemble of 3 approaches

| | What | Blend |
|---|---|---|
| **Approach 1** (base, most weight) | Multitask GNN: primary **pEC50** + single-conc log2fc (all doses) + **LogD** aux, **+ precomputed descriptors**; hyperparams re-opt on phase-1; final retrained incl. phase-1 | the base |
| **Approach 2** (weak end) | SVR imputes proxy pEC50 from single-conc → a 2nd multitask GNN (w/ LogD aux) on the proxy labels | mixed in where its pred **< 4.5** |
| **Approach 3** (strong end) | Model zoo (Chemprop / CheMeleon / Unimol2 / TabICL / SVR) → **LASSO** stack on OOF | mixed in where its pred **> 4.5** |

## What we can / cannot replicate

| #1 uses | us |
|---|---|
| proprietary LogD | **public MoleculeNet Lipophilicity (logD, 4,200)** as the LogD aux — flagged |
| proprietary PXR | skip (only worth ~0.0045 per #2) |
| single-conc all 4 doses | 3 usable doses (0.98 µM had ~1 point, dropped) |
| Unimol2 / TabICL in the zoo | add if feasible |

## Files & build order

| File | Approach | Runs on | Status |
|---|---|---|---|
| `build_approach1_data.py` | 1 | local (needs net) | builds `data/pxr_activity/train_approach1.csv` (16,474 rows) |
| `approach1_experiment.py` | 1 | **GPU** | ⏳ ready to run — the base multitask GNN |
| `build_proxy_labels.py` | 2 | local | builds `data/pxr_activity/proxy_train.csv` (6,805) |
| `proxy_specialist_experiment.py` | 2 | GPU | ✅ ran (crude single-target version; to redo as multitask+LogD) |
| *(todo)* approach3 zoo + LASSO | 3 | GPU | not built |
| *(todo)* meta_blend | — | local | not built |

Not yet in Approach 1: precomputed descriptors (step 1b), phase-1 hyperparam re-opt, phase-1 fold-in
(fold-in measured to hurt our pipeline — kept optional).

## How to run

```bash
# 1. (local, once) build the multitask training data — already committed, only to reproduce
uv run python reproduce_1/build_approach1_data.py

# 2. (GPU pod) Approach 1 — the base pillar
nohup python reproduce_1/approach1_experiment.py > approach1.log 2>&1 &
grep -A6 "Approach-1 Set-2" approach1.log     # result when done
```

Each script prints its Set-2 RAE (raw + variance-match calibrated) so we can track how close each
piece gets to #1's 0.5631.
