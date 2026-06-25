# PXR Structure (Pose) Prediction

Pipeline for the OpenADMET PXR challenge **Structure Track**: predict the bound
protein–ligand complex for each of 184 ligands, submit as `structures.zip`,
scored by **LDDT-PLI** (↑, primary) / BiSyRMSD (↓) via OpenStructure.

**No model training** — we run pretrained **Boltz-2** (co-folding) inference, then
convert → validate → package. Design rationale + full spec: [HANDOFF.md](HANDOFF.md).

## Status (built & verified on CPU)

- ✅ Input prep generates 184 Boltz YAMLs (verified).
- ✅ Instant baseline (official pre-generated structures) **PASSES the official
  validator** → `outputs/baseline_structures.zip` is submittable today.
- ✅ Calibration set: 62 PXR crystal references + SMILES (verified).
- ⏳ Our own Boltz-2 run + OST scoring run on the **L40S / cluster** (GPU + `ost`).

## Layout

```
src/        prep_inputs · cif_to_pdb · validate · build_submission · make_baseline_submission · config
scoring/    prep_refs · score_local            (local OST calibration vs crystals)
scripts/    run_boltz.slurm                    (L40S job)
env/        boltz-l40s.yaml (GPU) · structure-cpu.yaml (CPU)
vendor/     PXR-Challenge-Tutorial (official validator/scorer/FASTA/examples) · pxr_xtal_re-refinement (crystals)
data/       pxr-challenge_structure_TEST_BLINDED.csv  (184 SMILES + ids)
inputs/     boltz/*.yaml (generated)
outputs/    boltz/ · submission_pdbs/ · structures.zip · baseline_structures.zip
```

## Environments (keep separate from the activity-track env)

```bash
# CPU box (Mac): prep / convert / validate / package
conda env create -f env/structure-cpu.yaml && conda activate pxr-struct-cpu
# L40S box: Boltz-2 inference (+ gemmi for conversion)
conda env create -f env/boltz-l40s.yaml   && conda activate boltz-l40s
```

Run all commands from this `structure_prediction/` directory.

---

## Path A — submit a baseline TODAY (CPU, 1 min)

```bash
python -m src.make_baseline_submission                 # -> outputs/baseline_structures.zip
python -m src.validate outputs/baseline_structures.zip # -> PASS
```
Upload `outputs/baseline_structures.zip` on the challenge Space (Structure track).
This puts us on the leaderboard and confirms the submit path before we invest GPU.

## Path B — our own Boltz-2 predictions (the real pipeline)

```bash
# 1. [CPU] generate 184 Boltz inputs
python -m src.prep_inputs

# 2. [L40S] run Boltz-2 over all 184 (one job). Edit partition/account in the script first.
sbatch scripts/run_boltz.slurm
#    or directly:  boltz predict inputs/boltz --out_dir outputs/boltz \
#                       --output_format mmcif --use_msa_server --diffusion_samples 1 --override

# 3. [CPU or cluster] convert Boltz cifs -> submission PDBs (monomer + LIG)
python -m src.cif_to_pdb

# 4. [CPU] package + validate with the official validator
python -m src.build_submission --validate               # -> outputs/structures.zip
```
MSA note: the PXR protein is identical for all 184. `--use_msa_server` needs
internet on the compute node. To precompute once and reuse, generate a PXR `.a3m`
and run `python -m src.prep_inputs --msa inputs/msa/pxr.a3m`.

## Path C — local calibration (choose settings by REAL OST score)

The blinded 184 give no local score; calibrate on public PXR crystals instead.

```bash
# 1. [CPU] build references (rename bound ligand -> LIG) + fetch SMILES
python -m scoring.prep_refs                              # -> scoring/refs/*.pdb + manifest

# 2. [CPU] Boltz inputs for the 62 crystal ligands
python -m src.prep_inputs --csv scoring/calibration_manifest.csv \
       --id-col pdbid --smi-col smiles --out inputs/boltz_calib

# 3. [L40S] boltz predict inputs/boltz_calib --out_dir outputs/boltz_calib ...
# 4. [cluster] convert: python -m src.cif_to_pdb --boltz-out outputs/boltz_calib \
#                       --out outputs/calib_pdbs --csv scoring/calibration_manifest.csv
#    (cif_to_pdb reads the id column `structure`; for calib pass the manifest renamed,
#     or symlink — see note below)

# 5. [cluster + OST] score predicted vs crystals with the official scorer
apptainer pull ost.sif docker://registry.scicore.unibas.ch/schwede/openstructure:latest
apptainer exec ost.sif python -m scoring.score_local --pred-dir outputs/calib_pdbs
```
> cif_to_pdb currently keys on the `structure` column; for the calibration CSV
> (`pdbid`), either rename that column to `structure` or extend the script's
> `--csv` handling. (Left as a 1-line tweak so the test-set path stays simple.)

---

## What to hand to the L40S

Everything is relocatable — `rsync` this whole folder to the cluster:
```bash
rsync -av --exclude .venv structure_prediction/ <user>@<cluster>:~/pxr-structure/
```
Then on the cluster: create `boltz-l40s` env → `python -m src.prep_inputs` →
`sbatch scripts/run_boltz.slurm` → convert → validate. See Path B.
