#!/bin/bash
#SBATCH --job-name=pxr-boltz
#SBATCH --partition=gpu            # <-- EDIT to your cluster's L40S partition
#SBATCH --gres=gpu:l40s:1          # <-- EDIT to your cluster's GPU request syntax
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/boltz_%j.out
#SBATCH --error=logs/boltz_%j.err

# Run Boltz-2 over all 184 ligand YAMLs in one job.
# Prereqs (do once on the cluster):
#   conda env create -f env/boltz-l40s.yaml   (or: pip install boltz -U)
#   python -m src.prep_inputs                 (generates inputs/boltz/*.yaml)
#
# MSA: the PXR protein is identical for all 184 targets.
#   - If compute nodes have internet: --use_msa_server (used below) works but
#     re-queries per target. Prefer precomputing one MSA (see notes in README)
#     and running `python -m src.prep_inputs --msa inputs/msa/pxr.a3m`.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs outputs/boltz

# --- environment (EDIT for your cluster) ---
# module load cuda
# source activate boltz-l40s

INPUT_DIR=${1:-inputs/boltz}
OUT_DIR=${2:-outputs/boltz}

echo "Running Boltz-2 on ${INPUT_DIR} -> ${OUT_DIR}"
nvidia-smi || true

boltz predict "${INPUT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --output_format mmcif \
    --use_msa_server \
    --diffusion_samples 1 \
    --override

echo "Boltz done. Next: python -m src.cif_to_pdb && python -m src.build_submission --validate"
