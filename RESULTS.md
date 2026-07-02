# RESULTS & HANDOFF — PXR Activity Challenge

> **Purpose.** Snapshot of measured results, what's built, and the next levers — so another
> agent can continue without re-deriving anything. Read alongside
> [CHALLENGE_BRIEF.md](CHALLENGE_BRIEF.md), [PRODUCT_DESIGN.md](PRODUCT_DESIGN.md), [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md).
>
> | Field | Value |
> |---|---|
> | Updated | 2026-07-01 |
> | Mode | Competition CLOSED (July 1 23:59:59 UTC). **Missed the final leaderboard submission.** Pivoting to AAAI publication (~Aug 2026). |
> | Metric | RAE (lower better); leaderboard bootstraps 1000× |
> | Status | **Architecture B (`skill_manager.py`) verified END-TO-END on real GPU** → fully-autonomous run RAE **0.5783** (chemeleon 0.5887 + unimol 0.6248, nnls). Hand-assembled best remains **0.5706**. Cold-start discovery of CheMeleon reproduced. Next = AAAI plan in `docs/GENERALIZATION_PLAN.md`. |

## 2026-07-01 — Autonomous end-to-end run + competition close

- **Fully-autonomous Architecture-B run (real GPU, `run_skill_manager.py`, zero human model-picking):**
  setup(llm) → retrieve(llm; searched 33 models, selected chemeleon/unimol/frozen-others) → run(fine-tune
  chemeleon+unimol via templates) → stack(forward-selection + nnls) → **Set-1 judge RAE 0.5783 / MAE 0.4619**.
  Per-model judge: chemeleon(FT) **0.5887**, unimol(FT) **0.6248**; frozen baselines 0.669–0.767 (fine-tuning
  is the lever, quantified). nnls weights: chemeleon 0.716 / unimol 0.273. (OOF/CV estimate was 0.5465 —
  optimistic vs the 0.5783 judge; report the judge number.)
- **Cold-start discovery reproduced:** emptied registry+seed → retrieve pulled CheMeleon from recent arXiv
  abstracts → located weights on Zenodo (`CheMeleon Foundation Model`) + HF. Discovery works but is
  probabilistic (the LLM ranker didn't always select it); usability-as-finetune is seed/template-encoded.
- **unimol batch finding:** 64 OOMs 24G on full data (proven on a clean GPU — unimol's own attention, not
  leftover VRAM); **32 is the validated value** (reproduces 0.6248). chemeleon (D-MPNN) fine at 128.
- **Competition:** closed July 1 23:59:59 UTC; final submission (`predictions/final_submission_0.5783_foldedlabels.csv`,
  Set-1 true labels + Set-2 model preds) was built but **not uploaded in time**. Research value is intact for AAAI.

---

## Core principle (decision 2026-06-21) — "the judge can't join the competition"

**Analog Set 1 (253 now-public labels) is our own leaderboard judge.** We now know the true answers for these 253, so we can score ourselves exactly as the official leaderboard would.

- **Pretend we're still in Phase 1:** train on **broad 4,139 only**, predict all 513.
- **Set 1 = the judge/scorer** — use it to validate honestly and to *adapt* the pipeline to the analog distribution.
- **The judge NEVER joins the competition:** Set 1 is **never folded into training**, and it's not consumed as a one-shot test either.
- **Fold-in and external/auxiliary data are DEFERRED** to a later data-side phase. One step at a time.

---

## TL;DR

- **Submission to keep:** [`outputs/unimol/submission.csv`](outputs/unimol/submission.csv) — **Set-1 judge RAE 0.5706** (513 rows, validated). A CLEAN 2-model stack: **mt5 (CheMeleon graph) + Uni-Mol (3D)**, ridge-stacked. Beats the 48-base saturated ensemble (0.5916) with just 2 models.
- **Decorrelation thesis CONFIRMED (the key result):** Uni-Mol single is only 0.6248 (weaker than mt5's 0.5904), but corr(Uni-Mol, mt5)=0.866 (vs 0.905 within the CheMeleon family) — decorrelated enough that stacking drops mt5 0.5904 → **0.5706** (−0.020). Two decorrelated strong models beat 48 correlated ones. Gap to discoverybytes' 0.538 now ~0.033.
- **(superseded) cheme_mt5 saturated ensemble** = 0.5916; mt5 single = 0.5904.
- **mt5 = the 0.538-aligned Chemprop member:** CheMeleon + 5-task multitask (pEC50 + counter_pEC50 + Emax + counter_Emax + **single_concentration**, +8135 screen molecules) + **MAE loss** + **reactive-electrophile exclusion**. Single-model judge **0.5904** — vs mt 0.6144, single-task 0.6637. ≈ discoverybytes' Chemprop single (~0.581).
- **Saturation confirmed:** mt5 *single* 0.5904 vs mt5 *in 48-base ensemble* 0.5916 (a hair worse!); corr(mt5,mt)=0.905; dropping mt changes nothing. The 47 legacy members are dead weight — only a **decorrelated** member (Uni-Mol 3D / MolE foundation) can improve from here.
- **Progression (Set-1 judge):** m1 ensemble **0.6329** → +mlp/drop-weights **0.6320** → +calibrated cluster folds **0.6266** → +frozen CheMeleon **0.6217** → +fine-tuned CheMeleon **0.6135** → +multitask CheMeleon **0.6108** → **+mt5 (single_conc+MAE+curation) 0.5916** (single 0.5904).
- **Biggest single-model jump = multitask:** single-task fine-tune judged **0.6637**; **multitask** (pEC50 + counter_pEC50 + Emax + counter_Emax heads) judged **0.6144** — a −0.05 jump, ≈ our whole 46-base ensemble. Confirms broad-only is **not** saturated and the CheMeleon graph model + multitask is the real lever (matches the competitor 0.689→0.60 rung). The ensemble only netted −0.003 because multitask is *correlated* with the existing CheMeleon members (it *is* most of the signal, not orthogonal to it).
- **Tried, did NOT help (all within-noise / dropped):** isotonic calibration (broad→analog shift), counter-assay sample-weighting (too aggressive), Mordred features (redundant with existing descriptor/CheMeleon members). The cheap post-hoc tricks are exhausted; real gains come from new *signal* (multitask; next: targeted PXR external multitask heads + end-game Set-1 fold-in).
- **Our honest standing:** broad-only ensembles judge at **~0.627–0.636** on Set 1 (scaffold-CV says ~0.543 — overstated by ~0.09–0.10, confirmed across 22 plans, mean gap **+0.13**).
- **Menu now complete, each module judged on Set 1 (keep only what lowers judge RAE):**
  - `mlp_head` ✅ **kept** — different (non-tree) family; lowers the *ensemble* judge RAE (its single-model score is mediocre, the gain is purely diversity).
  - `molformer_embedding` ❌ **dropped** — weak on this analog task (single models RAE ≈ 1.0); slightly *raises* the ensemble judge RAE.
  - sample weights (`pEC50_std.error`) ❌ **dropped** — `inv_se`/`inv_var` *raise* judge RAE on every base (scaffold-CV sometimes liked them → another mirage). Code stays as an available lever.
- **#1 lesson (reconfirmed):** broad **scaffold-CV is NOT a reliable proxy** — it even disagrees on *which* module helps. The judge is Set 1.
- **Target:** Top-5 **rank** on the blind Set-2 — *not* a fixed RAE (0.538 is a moving reference; the bar drops in Phase 2 as everyone improves).

---

## The bar (interim leaderboard, downloaded 2026-06-21 — see CHALLENGE_BRIEF Section 8)

| Cutoff | RAE |
|---|---|
| #1 (AIDD-LiLab) | 0.528 |
| Top-3 | 0.536 |
| **Top-5** | **0.538** |
| Top-10 | 0.562 |
| constant-mean baseline | ~1.04 |

RAE_std ≈ 0.02 → the top ~8 are within ~1 bootstrap std (near statistical tie; small real gains jump many ranks). Leaders reached 0.528–0.538 **on the full 513 without Set 1 in training** → their broad models predict analogs far better than ours (0.633). Most used **no proprietary data**.

---

## Measured results

### 1. Broad scaffold-CV menu (the menu — DO NOT trust for ranking)
- 42-plan menu (7 reps × 6 models). Best single = `rdkit_descriptors + xgboost` **0.5768**.
- Ridge-stacked ensemble (42 bases) = **0.5432**. Tuning the top GBDTs did **not** help (defaults already near-optimal).
- Artifacts: `outputs/m1_menu/` (`leaderboard.json`, `plans/<id>/`, `ensemble/`, `submission.csv`).

### 2. ⚠️ The Set-1 judge says we're at 0.633 (not 0.543)
Scored the broad-only ensemble's 513 predictions against the 253 now-public labels:

| | scaffold-CV | **Set-1 judge (real)** |
|---|---|---|
| RAE | 0.5432 | **0.6329** |
| MAE | ~0.50 | 0.5055 |
| R² | 0.62 | 0.491 |

The analog set is *tighter* (y std 1.03 vs train 1.12) with activity cliffs → same MAE, worse *relative* error. **Scaffold-CV overstated us by ~0.09 RAE.** Per-base analog ranking also differs from scaffold-CV (desc+lightgbm best on analog; `elastic_net` catastrophic, RAE > 1.0).

### 3. [DEFERRED — recorded, NOT in use] fold-in measurement
We measured that folding Set 1 into training would drop single-model analog RAE ~0.10 (lightgbm 0.697→0.609, xgboost 0.722→0.615, catboost 0.721→0.616). **This is shelved** — the judge can't join the competition. Revisit only in the later data-side phase. The fold-in pipeline/output have been removed.

---

### 4. Menu completion judged on Set 1 (2026-06-21) — keep only what lowers judge RAE
Built the missing broad-train + Set-1-judge eval loop (`scripts/run_judge.py`, 22-plan matrix) and the keep/drop test (`scripts/ensemble_compare.py`). Every module trained **broad-only**, judged on the 253.

| Module | As a single model (judge RAE) | In the ensemble (judge RAE) | Verdict |
|---|---|---|---|
| baseline (35 strong m1 bases) | — | **0.6359** | — |
| **`mlp_head`** (desc/cb/fusion) | 0.75–0.81 (mediocre) | **0.6320** (−0.0039) | ✅ **keep** (diversity) |
| `molformer_embedding` | **0.98–1.00** (≈ predict-the-mean) | 0.6375 (+0.0015) | ❌ **drop** (weak rep here) |
| sample weights `inv_se`/`inv_var` | +0.02–0.03 worse on every base | — | ❌ **drop** (scaffold-CV liked them; judge didn't) |

- **Best ensemble = baseline + mlp = judge RAE 0.6320.** Small (~0.004, within bootstrap noise) but in the right direction; `mlp` is a genuinely different (non-tree) family.
- **MolFormer (IBM MoLFormer-XL) is a weak representation for this task** — ChemBERTa-77M-MTR (cb_ridge 0.74) crushes it (mf_ridge 0.98). Plausibly because ChemBERTa-MTR was pretrained with a multi-task *regression* objective closer to property prediction. Code kept for completeness, off by default.
- **Across all 22 plans, Spearman(scaffold-CV, judge) = 0.80, mean gap +0.13** — scaffold-CV ranks roughly but mis-levels badly and flips the close calls (it picked desc_xgboost as best single; the judge picked desc_lightgbm; it "liked" sample weights the judge rejected).

### 5. Calibrating internal CV to the judge (cluster folds)
`scripts/calibrate_folds.py` swept Butina Tanimoto-cluster fold cutoffs over 4 diverse probe models (judge RAE 0.70–0.99), scoring each design by Spearman(CV-rank, judge-rank). **Leakage-safe: folds partition only the broad 4,139; Set-1 labels grade the fold *designs*, never enter training.**

| design | meanCV | Spearman vs judge |
|---|---|---|
| scaffold (current) | 0.6445 | 0.800 |
| cluster@0.5 | 0.6436 | 0.800 |
| **cluster@0.6** (chosen) | 0.6444 | **1.000** |
| cluster@0.7 | 0.6487 | 0.800 |
| cluster@0.8 | 0.6480 | 1.000 (but degenerate: 254 clusters, lumpy folds) |

- **Chosen: `cluster@0.6`** → `data/pxr_activity/folds_calibrated.json` (2977 clusters, balanced folds 828×5, largest cluster 24). It matches the judge's per-model ranking (no swaps) where scaffold flips a pair, and is well-balanced. *(cluster@0.8 also hit Spearman 1.0 but is near-degenerate — a few mega-clusters; the selector now prefers balanced folds among rank-ties.)*
- **Honest nuance — what calibration did and did NOT do:** the cluster folds fix the **ranking** but do **NOT** close the absolute **level** gap. Every design sits at meanCV ≈ 0.64 vs the judge's ≈ 0.79 (these probes) — re-folding broad data can't reproduce the analog set's intrinsic difficulty (tighter targets + activity cliffs). **So: use `folds_calibrated.json` for trustworthy model *selection* on broad data; the Set-1 judge stays the authority on *absolute* RAE.**
- **Caveat:** 4-probe Spearman is coarse (values quantize to {0.8, 1.0}); this is supporting evidence, not proof. More probes / a finer rank metric would harden it.

### 6. Final calibrated submission (2026-06-22) — `scripts/finalize.py`
Re-ran the strong menu (7 reps × {ridge,rf,xgboost,lightgbm,catboost} + `mlp_head` on the 4 dense reps = 39 bases; molformer & elastic_net excluded) on the **calibrated cluster folds** with full-data refit, ridge-stacked, judged on Set 1.

| ensemble | folds | Set-1 judge RAE |
|---|---|---|
| original m1 | scaffold | 0.6329 |
| baseline + mlp | scaffold | 0.6320 |
| **strong menu + mlp (39 bases)** | **calibrated cluster@0.6** | **0.6266** ✅ |

→ **`outputs/final/submission.csv`** (513 rows, validated; `submission_manifest.json` records provenance). Stacking on cluster-fold OOF yields weights that transfer slightly better to the analog distribution. Honest read: the move from 0.6329→0.6266 is real and consistently downward but each step is within bootstrap noise (~0.02–0.03 on 253) — meaningful for *rank* only if it holds on the blind Set 2. **Broad-only modeling now looks near its ceiling (~0.62 vs the ~0.54 leaderboard top); the next real gains are the deferred data side (fold-in, external/auxiliary data).**

### 7. M3 automation — judge-in-the-loop auto-design (2026-06-22) — ✅ built & working
The first **LLM-in-the-loop** part of the pipeline (everything before was deterministic). `scripts/run_auto.py` (Manager) + two LLM agents — `src/agent/menu_designer.py` (**Designer**: proposes candidates) and `src/agent/menu_tuner.py` (**Tuner**: `--tune-top` refines the best bases' hyperparameters). Each round: propose → run on calibrated folds → judged on Set 1 → stacked → Manager continues/stops on budget. **Objective = the Set-1 judge** (not scaffold-CV — which fixes our earlier mistake of tuning on the miscalibrated metric). Selector stays deterministic by design; retrieval/legacy agents dormant.

| run | rounds | result |
|---|---|---|
| deterministic-fallback only | seed→r1 | 0.6266 → **0.6178** (added desc+morgan / desc+maccs fusions) |
| live LLM Designer | seed→r1→r2 | 0.6266 → 0.6258 → **0.6237** (r2 LLM call failed → **fell back cleanly**) |

- **Confirms more/diverse candidates can lower the ensemble RAE** — the auto-loop found fusions the fixed menu lacked (e.g. descriptors+morgan, single judge 0.6608 vs plain desc 0.697). Validated the "why not more candidates" question: yes, it helps — modestly.
- **Safety properties all demonstrated:** anchored on the frozen menu (never regress below 0.6266); **graceful LLM fallback** (round 2 ran on the deterministic preset when OpenRouter returned empty); budget (`--rounds`/`--candidates`); resume (`run_state.json`).
- **Honest caveats:** (1) gains (0.6178–0.6237) are within the ~0.02–0.03 bootstrap noise on 253; (2) **more candidates is NOT monotonically better** — the deterministic 3-fusion pool (0.6178) beat the LLM 12-candidate pool (0.6237) because a ridge stack gets diluted by many correlated mediocre bases; (3) hard-optimizing the 253-judge risks **overfitting Set 1** — the real test is the blind Set 2. So M3 is valuable as a *system* (autonomous, safe, finds candidates), but the number shouldn't be over-trusted. **M3 does not change the conclusion that broad modeling is near its ceiling; the data side remains the larger lever.**

---

## What's built (all uncommitted in git)

**Deterministic layer (`src/`):**
- `metrics.py` — rae/mae/r2/score_all.
- **`analog_judge.py`** — **the Set-1 judge.** `judge_predictions`/`judge_csv`: match a 513-row prediction file to the 253 Set-1 labels by `Molecule Name`, score RAE/MAE/R². Read-only on Set 1 (never trains on it). Reproduces 0.6329 on the m1 ensemble.
- `featurizers.py` — registry: `morgan`, `maccs`, `avalon`, `rdkit_descriptors`, `chemberta_embedding` (77M+100M), **`molformer_embedding`** ✅, `fusion`. Content-cached to `data/featurizer_cache/`. *(MolFormer needs transformers-5.x compat shims + a rotary-cache rebuild — see code; embeddings cached for all 3 splits.)*
- `models.py` — registry: `ridge`, `elastic_net`, `random_forest`, `xgboost`, `lightgbm`, `catboost`, **`mlp_head`** ✅. `fit_model()` routes **sample_weight** to a pipeline's final step / skips models that don't support it.
- **`sample_weights.py`** — `compute_sample_weights` from `pEC50_std.error` (`none`/`inv_se`/`inv_var`, quantile-floored, mean-1 normalized).
- `cv_runner.py` — runs a plan on frozen folds → leak-free OOF + test preds; `refit_full`; threads `weight_scheme`.
- `cv_split.py` — scaffold folds **+ `assign_cluster_folds`** (Butina Tanimoto clusters; `cutoff` is the calibration knob).
- `aggregator.py` — honest CV-stacked ensemble (mean/nnls/ridge).
- `data_io.py`, `menu_config.py`, `tuner_search.py`, `schemas.py`.

**Scripts (`scripts/`):**
- **`judge.py`** — score any 513-row prediction file on the Set-1 judge.
- **`run_judge.py`** — **the broad-train + Set-1-judge eval** (replaces `run_analog.py`'s empty role): runs a plan matrix broad-only, reports scaffold-CV vs judge RAE side-by-side + Spearman.
- **`ensemble_compare.py`** — keep/drop test: does adding a module to the strong base pool lower the *ensemble* judge RAE?
- **`calibrate_folds.py`** — pick the cluster-fold cutoff whose per-model ranking best matches the judge (close the 0.543→0.633 gap).
- `run_menu.py` — broad menu sweep → leaderboard (Phase-1).
- `run_ensemble.py` — stack the menu's OOF (Phase-1).
- `tune.py` — tune top pairs.
- ~~`run_analog.py`~~ — **DEPRECATED (does fold-in). Replaced by `run_judge.py` for the broad-train + Set-1-judge eval. Do not use.**

**Data:**
- `data/pxr_activity/{train.csv (4139), test.csv (513), folds.json, sample_submission.csv}`
- `data/pxr_activity/phase1_unblinded.csv` — **Set-1 judge, 253 labels**.
- `data/featurizer_cache/` — cached embeddings.

---

## Gotchas (will bite you)

1. **OpenMP deadlock on Mac.** ChemBERTa/torch hangs at 0% CPU unless `OMP_NUM_THREADS=1` (+ `torch.set_num_threads(1)`). With it: 8s for 4,652 mols.
2. **HF offline.** `src/agent/models.py` forces `HF_HUB_OFFLINE=1`; pop those env vars to download new data/models.
3. **Scale only dense reps.** StandardScaler on binary fingerprints wrecks linear models. Fingerprints = `binary: True` in `menu_config`.
4. **RDKit `Ipc` descriptor** explodes → `_maybe_scale` z-clips it.
5. **Featurizer cache key** must use only featurizer params (skill_ref/pooling).
6. **`elastic_net` is useless on analogs** (RAE > 1.0).

---

## Remaining levers (order — time-boxed to July 1)

**Step-by-step; finish the model side first, judged on Set 1. Data-side is deferred.**

1. **Finish the Approach-1 menu**, each judged on the Set-1 judge (keep only what lowers its RAE):
   `molformer_embedding` featurizer · `mlp_head` model · **sample weights** from `pEC50_std.error`.
2. **Make our internal CV faithful to the judge** (close the 0.543→0.633 gap) — see below. Without this we select blind.
3. **Analog-aware modeling** for activity cliffs (e.g. local/kNN on fingerprints, Δ/pairwise models).
4. **(DEFERRED) data-side:** fold-in, auxiliary configs (21K single-conc etc.), external data (DataMaster).
5. **MANDATORY: methodology report by July 1** (code preferred). No report = not ranked.

> **Deprioritized:** M3 LLM orchestration — out of the time budget.

### How to close the CV ↔ judge gap (lever 2)
Goal: not a lower number, but an internal CV (on broad data) that **ranks models the same way the Set-1 judge does**, so we can select trustworthily without touching the judge.
- **Analog-faithful folds:** replace random/scaffold folds with **cluster- or nearest-neighbour-based folds** (hold out tight similarity clusters), so each fold mimics "predict close analogs of what you've seen" — the real task.
- **Calibrate against the judge:** try several CV designs, pick the one whose CV RAE ≈ the judge's 0.633 **and** whose per-model ranking best correlates (Spearman) with the judge's ranking. Use that CV to select; the judge confirms.
- **Honest caveat:** Set 2 is genuinely different analogs, so a perfect match is impossible — aim for rank-agreement, not an identical number.

---

## One-paragraph status for a new agent

We built a clean Approach-1 broad menu (featurizer × model registries, leak-free scaffold-CV OOF, ridge-stacked ensemble) at scaffold-CV RAE 0.543. The 513-compound test set is an **analog set** (activity cliffs), so scaffold-CV is unreliable — confirmed: the now-public **Analog Set 1 (253 labels)** judges our broad ensemble at **0.633**. **Decision: Set 1 is our private judge and must stay OUT of training (the judge can't join the competition); we stay in a Phase-1 setup (broad-trained, predict 513).** Next, step by step: finish the Approach-1 menu (`molformer_embedding`, `mlp_head`, sample weights) judged on Set 1; make the internal CV faithful to the judge; then (later) the data side. Fold-in was measured (~0.61) but **shelved and removed**. Target = Top-5 **rank** on the blind Set-2, not a fixed RAE.
