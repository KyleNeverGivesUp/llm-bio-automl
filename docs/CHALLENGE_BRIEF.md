# CHALLENGE BRIEF / BUSINESS REQUIREMENTS — OpenADMET PXR Induction Blind Challenge

> **Purpose of this file.** This is the **external source of truth** for the competition we are building for — the business background, task, data, rules, scoring, and timeline as published by the organizers. The PRD ([PRODUCT_DESIGN.md](PRODUCT_DESIGN.md)) and technical design ([TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)) are our *internal* product/engineering response and should be updated **from** this document. When the two disagree, this file (the organizer's spec) wins.

| Field | Value |
|---|---|
| Challenge | OpenADMET PXR Induction Blind Challenge |
| Organizer | OpenADMET |
| Challenge Space | [huggingface.co/spaces/openadmet/pxr-challenge](https://huggingface.co/spaces/openadmet/pxr-challenge) (Docker app) |
| Dataset | [huggingface.co/datasets/openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) |
| Tutorial | [github.com/OpenADMET/PXR-Challenge-Tutorial](https://github.com/OpenADMET/PXR-Challenge-Tutorial/tree/main) |
| Community | `#pxr-challenge` on [Discord](https://discord.gg/MY5cEFHH3D) |
| Brief compiled | 2026-06-21 (from Space repo `app.py` / `config.py` / `data/*` and the dataset card) |
| Our scope | **Activity Track only** (Structure Track is out of scope — see PRD §4) |

> **Provenance / confidence.** Facts below were read directly from the Space's source files (`app.py`, `config.py`, `submission_store.py`, `models.py`, `data/activity_leaderboard.csv`) and the dataset card. The live app page itself can't be scraped (it renders via Docker). Items I **inferred** (rather than read verbatim) are tagged **[inferred]**.

---

## 1. Business background (why this challenge exists)

- The challenge predicts **Pregnane-X Receptor (PXR) induction**. PXR is a **nuclear hormone receptor** and a **master regulator of drug-metabolizing enzymes and transporters**.
- PXR primarily controls **CYP3A4**, the enzyme that metabolizes **~50% of all marketed drugs**.
- A compound that **induces** PXR can ramp up these enzymes and **derail drug-discovery projects** by causing adverse drug–drug interactions (it changes how other drugs are cleared). So predicting PXR induction early is a real ADMET safety/efficacy concern.
- The dataset is described as **"the largest publicly available PXR activity dataset"** — **over 11,000 compounds** screened with a **high-fidelity in-house assay** — released specifically for this blind challenge.

**Why "blind":** part of the test set is held out (blinded) until the deadline, so models are judged on genuinely unseen compounds rather than data they could have memorized.

---

## 2. Challenge structure — two tracks

| Track | Goal (verbatim) | Size | We do it? |
|---|---|---|---|
| **1. Activity Prediction** | "Predict pEC50 values for a test set of 513 compounds (split into two stages)." | 513 compounds | ✅ **yes — our only track** |
| **2. Structure Prediction** | "Predict the bound structures of 184 ligands to PXR." | 184 ligands | ❌ no (PRD Non-Goal §4) |

---

## 3. Activity Track (our focus)

### 3.1 Task
- **Input:** a molecule as a **SMILES** string.
- **Output:** **pEC50** — a continuous potency value (regression). Higher pEC50 = more potent inducer.
- **Test set:** **exactly 513 compounds**, submitted as a single file every time.

### 3.2 Two-stage (phase) structure — important
The 513-compound activity test set is split into two analog sets:

| Stage | What it is | Status as of 2026-06-21 |
|---|---|---|
| **Phase 1 — Analog Set 1** | Scored on a **live leaderboard** during Phase 1 | **Concluded May 25**; results **unblinded May 26** |
| **Phase 2 — Analog Set 2** | **Fully blinded** until the final deadline; refine predictions here | **Open now**; closes **July 1** |

- `config.py` sets **`CURRENT_PHASE = 2`** — the Space is currently running Phase 2.
- **[inferred]** From the dataset configs (below): **Analog Set 1 = 253 compounds** (now released as the `phase_1_unblinded` test split) and **Analog Set 2 = 260 compounds** (513 − 253, still blinded). **You still submit all 513 rows**; only the Phase-2 portion is scored blind at the end.
- **Actionable consequence:** the **253 Phase-1 labels are now public**, so they can be **folded into training** to improve the blinded Phase-2 predictions. This is a free accuracy gain that did not exist during Phase 1.

**Verified on the live app (2026-06-21).** The Activity ("EC50") and Structure ("Pose") tracks have **separate leaderboard tabs** — confirming Phase 2 is *not* the Structure Track. The Activity leaderboard currently shows two things:
- a **static *interim* leaderboard** — a frozen snapshot from Phase-1 close, scored on **all (Analog Sets 1 and 2)** compounds; and
- a **live Phase-2 leaderboard** — running feedback scored on the **Analog Set 1** compounds.

Analog Set 2's ground-truth labels stay hidden and decide the **final July-1 ranking**. (The two app views word this slightly differently while the Space is mid-restart — confirm the exact rule in the in-app **FAQ** tab / Challenge Announcement.) **Strategic read:** the final rank is on the blind Set 2. Use Set 1 as our **private judge** to validate/select and adapt the pipeline (kept OUT of training — the judge can't compete); broad scaffold-CV is NOT a reliable proxy.

### 3.3 Submission format (Activity Track)
- **File:** `.parquet` **or** `.csv`.
- **Rows:** **exactly 513** (all test compounds required).
- **Columns (case-sensitive, exact names):**
  - `SMILES`
  - `Molecule Name`
  - `pEC50`
- **Validation rules:** all 513 compounds must be present; column names are case-sensitive; **no NaN or infinite values** in `pEC50`.
- **Throttle:** must wait **4 hours** between consecutive submissions (`config.py`). **[inferred]** possibly relaxed later in the challenge.
- **Ranking rule:** **only your latest submission counts** toward the leaderboard.

### 3.4 Scoring — Activity Track
- **Primary metric: RAE (Relative Absolute Error)** on pEC50 — **lower is better**.
  - Standard definition: `RAE = Σ|ŷᵢ − yᵢ| / Σ|yᵢ − ȳ|`. **RAE = 1.0** means "no better than always predicting the mean"; **RAE < 1** beats that baseline.
- **Secondary metrics (reported, not ranked):** **MAE**, **R²**, **Spearman R**, **Kendall's τ**.
- **Uncertainty:** every metric is **bootstrapped over 1,000 resamples**, reported as **mean ± std** (the leaderboard stores `RAE_mean`, `RAE_std`, etc.).

---

## 4. Structure Track (out of scope — recorded for completeness)
- **Submission:** a `.zip` of **exactly 184 `.pdb`** files; each must be a **full protein–ligand complex** (not ligand-only) with the ligand residue named **`LIG`**.
- **Scoring:** primary **LDDT-PLI** (superposition-free protein–ligand contact score); secondary **BiSyRMSD** (symmetry-corrected binding-site RMSD). Invalid ligand assignments are penalized (**LDDT-PLI = 0.0**, **BiSyRMSD = 20.0 Å**).
- **We are not entering this track.**

---

## 5. Data assets (`openadmet/pxr-challenge-train-test`)

The dataset is **Apache-2.0 licensed**, ~7.1K downloads, updated **2026-06-18**. It ships **multiple configs** — the challenge itself provides extra training signal beyond the core 4.1K set:

| Config | Split | Rows | Cols | Role |
|---|---|---:|---:|---|
| `default` | **train** | **4.1K** (≈4,139) | 18 | **Primary curated training set** (high-quality dose-response pEC50) |
| `default` | **test** | **513** | 18 | **The activity test set we predict** |
| `phase_1_unblinded` | test | **253** | 17 | **Phase-1 (Analog Set 1) labels, now revealed** → usable as extra training data |
| `structure` | test | 184 | 4 | Structure-track targets (not ours) |
| `counter_assay` | train | 2.9K | 18 | Counter-screen assay (auxiliary signal) |
| `single_concentration` | train | 21.0K | 19 | Large single-concentration screen (lower-fidelity, high volume) |
| `crudes_htchem` | train | 456 | 24 | High-throughput chemistry crude samples (auxiliary) |
| `semi_pure_htchem` | train | 96 | 23 | Semi-pure HT-chem samples (auxiliary) |

**Key schema columns (`default` config):** `Molecule Name` (e.g. `OADMET-0006089`), `SMILES`, `OCNT Batch`, **`pEC50`** (target; observed range ≈ 1.6–7.6), `Emax_estimate`, **`pEC50_std.error`** (per-label uncertainty 0–0.74 — usable as a sample weight), `pEC50_ci.lower/upper`, `Emax_*` estimates/CIs, `Split` (single value "Train"), `OCNT_ID`, `source`.

**Exact file paths (load directly with pandas `pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/<file>")`):**
| Config / split | File |
|---|---|
| default / train | `pxr-challenge_TRAIN.csv` |
| default / test (blinded 513) | `pxr-challenge_TEST_BLINDED.csv` |
| **phase_1_unblinded / test (253 labels)** | `pxr-challenge_TEST_PHASE_1_UNBLINDED.csv` |
| counter_assay / train | `pxr-challenge_counter-assay_TRAIN.csv` |
| single_concentration / train | `pxr-challenge_single_concentration_TRAIN.csv` |
| crudes_htchem / train | `pxr-challenge_htchem-libraries_TRAIN.csv` |
| semi_pure_htchem / train | `pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv` |
| structure / test (184) | `pxr-challenge_structure_TEST_BLINDED.csv` |

Dataset **CHANGELOG**: 2026-05-27 added `phase_1_unblinded` labels + extra crude data; 2026-04-09 dropped some compounds, fixed CI issues, improved name joins. (So re-pull the data — earlier local copies may be stale.)

**Implications:**
- The **`pEC50_std.error`** column confirms the per-measurement uncertainty signal already noted in the PRD (sample-weighting is legitimately available).
- The **auxiliary configs** (`counter_assay`, `single_concentration`, `crudes_htchem`, `semi_pure_htchem`) are **organizer-provided extra training data** — this answers the PRD open question "are external/extra data allowed?" for *in-dataset* sources: yes, and they're sitting right here.

---

## 6. Timeline

| Date (2026) | Event |
|---|---|
| March 17 | Challenge announced |
| **April 1** | Training/test sets released; **submissions open** |
| **May 25** | **Phase 1 concludes**; interim activity leaderboard released |
| May 26 | Analog Set 1 (Phase 1) results **unblinded** |
| **July 1, 23:59:59 UTC** | **FINAL DEADLINE** — Phase 2 (Activity) **and** Structure Track close |

> **As of today (2026-06-21): Phase 1 is over; we are in Phase 2 with ~10 days left.** July 1 is the hard cutoff.

---

## 7. Rules & participation requirements

- **Methodology report is mandatory.** "A methodology report (code preferred; a written report is the minimum) must be submitted by the challenge close date (July 1) for your entry to appear on the final leaderboard." → **No report = not ranked**, no matter how good the score.
- **Proprietary-data disclosure.** A "I used proprietary data" checkbox; the flag is shown publicly.
- **Anonymous submission allowed.** You may show an alias instead of your username on the leaderboard and the Discord validation bot.
- **Latest-submission-wins** ranking (§3.3).
- **4-hour submission throttle** during the first week; "likely move to once per day" after (FAQ).
- **External data is explicitly allowed** (FAQ) — not just the in-dataset auxiliary configs. Structural↔activity data may be cross-used across tracks.
- **One account/alias per team.** "Submit under a single HuggingFace account or alias. Do not submit the same predictions from multiple accounts" (FAQ). (Note the leaderboard already shows many near-duplicate aliases — multi-account submitting is against the rules.)
- **Method report** is submitted via the **"Method Report Link" field** in the Submit tab (FAQ).
- **Validate before uploading** with the validation script in the tutorial repo; submission receipts/errors post to Discord `#pxr-challenge-submissions`.
- **Summary paper** planned (possibly merged with the Expansion-challenge paper); opt in via the submission form to be considered for inclusion.

---

## 8. Real leaderboard — the actual bar to beat (interim, downloaded 2026-06-21)

The repo's `data/activity_leaderboard.csv` is placeholder seed data (joke usernames, 2025 timestamps) — **ignore it.** The **real** Activity leaderboard was downloaded from the live app on 2026-06-21: the **static interim board**, each Phase-1 submission scored on **all 513** analog-test compounds. **328 entries.** RAE (lower = better), bootstrapped 1000×.

| Rank | User | RAE | R² | Proprietary |
|---:|---|---:|---:|---|
| 1 | AIDD-LiLab | **0.528** | 0.64 | No |
| 2 | toxicity | 0.529 | 0.66 | Yes |
| 3 | nova | 0.536 | 0.64 | No |
| 4 | N283T | 0.536 | 0.65 | No |
| 5 | sia | **0.538** | 0.64 | No |
| 6 | oxidane | 0.540 | 0.64 | No |
| 7 | PXRegressor | 0.546 | 0.64 | No |
| 8 | matcha-croissant | 0.549 | 0.62 | Yes |
| 9 | jaybirdy | 0.562 | 0.61 | No |
| 10 | Maisy | 0.562 | 0.61 | No |

**Rank cutoffs:** #1 = **0.528** · Top 3 = 0.536 · **Top 5 = 0.538** · Top 10 = 0.562. (Worst entries run to RAE > 1.0; a constant-mean baseline ≈ 1.04, so RAE ≈ 1.0 = "no better than predicting the mean".)

**Critical reads:**
- **Nobody is below 0.52.** The PRD KR "RAE < 0.50" is *below the current #1 (0.528)* — it's not "Top 5," it's "win by a landslide / likely unachievable on this metric." **The target is a Top-5 *rank* on the blind Set-2 — not a fixed RAE (the bar drops as everyone improves in Phase 2).**
- **The top is a statistical tie.** RAE_std ≈ 0.02; #1 (0.528) and #8 (0.549) differ by ~0.021 ≈ **1 bootstrap std** → small real gains jump many ranks, and rank is noisy near the top.
- Most top entries used **no proprietary data** (#2 and #8 are the exceptions).
- **Our broad ensemble's real standing (judged on Set 1) = RAE 0.633** — scaffold-CV's 0.549 did not transfer (see Section 8.1).

### 8.1 ⚠️ The test set is an *analog set* — broad scaffold-CV is not a reliable proxy
Per the live "About" page, the 513-compound test set was built by **ECFP4-Tanimoto > 0.4 similarity search around 63 selective hits** — i.e. close analogs with **deliberate activity cliffs and tight SAR**, "designed to be challenging for models." This is a **different, narrower distribution** than the broad ~4,140-compound training set. Consequence:
- Our **scaffold-CV RAE on training data is not guaranteed to predict analog-test RAE.** We need a real read on the analog distribution.
- **Analog Set 1 (253 compounds, now public) IS that distribution.** Use it as our **private judge** to validate/select and *adapt* the pipeline — but **keep it OUT of training** (decision 2026-06-21: the judge can't join the competition). Fold-in and external/auxiliary data are **deferred** to a later data-side phase.

---

## 9. Resources & links
- **Challenge Space:** https://huggingface.co/spaces/openadmet/pxr-challenge
- **Dataset:** https://huggingface.co/datasets/openadmet/pxr-challenge-train-test
- **Tutorial repo:** https://github.com/OpenADMET/PXR-Challenge-Tutorial/tree/main
- **Blog post:** "Announcing the Next OpenADMET Blind Challenge: Predicting PXR Induction"
- **Discord:** `#pxr-challenge` — https://discord.gg/MY5cEFHH3D
- **In-app nav (live app):** tabs **About · Leaderboard · Submit · FAQ**; buttons **"Challenge Announcement →"** and **"Training & Test Dataset →"**. The **FAQ** tab is the place to confirm exact phase/scoring rules.

---

## 10. Implications for our PRD (what to update)

This brief resolves several PRD open questions and adds new facts. Suggested PRD edits (not yet applied):

| PRD location | Current state | Update from this brief |
|---|---|---|
| §16 Open Q1 (private split) | "Unknown; using safe default" | Test is **513 fixed compounds**, split into **Phase 1 (253, unblinded) + Phase 2 (260, blinded)**; broad scaffold-CV is NOT a reliable proxy (real 0.633 vs 0.543) — validate on Analog Set 1 (our judge). |
| §16 Open Q2 (external data) | "To confirm" | **RESOLVED: external data is explicitly allowed** (FAQ), on top of the in-dataset auxiliary configs (counter_assay, single_concentration, crudes/semi-pure HT-chem). Structural↔activity data may be cross-used. |
| §16 Open Q4 (deadline) | "To confirm" | **Resolved: July 1, 23:59:59 UTC.** Phase 1 already closed May 25. |
| §16 verified data facts | train 4,139 / test 513 | Add: **253 Phase-1 labels now public** → add to training; **`pEC50_std.error`** confirmed as sample weight. |
| §3 KRs / §2 problem | RAE < 0.50 target vs "~0.55 baseline" | **Recalibrate: real interim Top-5 bar = RAE ≈ 0.538, #1 = 0.528, nobody < 0.52 (§8).** "< 0.50" is below #1 → drop it; target = Top-5 **rank** on the blind Set-2 (0.538 is a moving reference, not a fixed number). Our broad ensemble's real standing (Set-1 judge) = 0.633 — validate on Analog Set 1, don't trust broad scaffold-CV. |
| §10 / §15 roadmap | M3 automation, M4 escape hatch | **Time-box to July 1:** prioritize M1+M2 + Phase-1-augmented training + a methodology report; M3/M4 likely out of budget. |
| New requirement | — | **Methodology report by July 1 is mandatory for ranking** — add as a hard deliverable. |

---

### Appendix: source files read
`app.py`, `config.py` (`CURRENT_PHASE=2`, sizes 513/184, columns, 4-hour throttle), `submission_store.py` (S3 submission storage), `models.py` (Pydantic submission model), `data/activity_leaderboard.csv` + `data/structure_leaderboard.csv` (placeholder), dataset card for `openadmet/pxr-challenge-train-test` (configs/splits/schema).
