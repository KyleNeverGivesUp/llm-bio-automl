"""Shared paths and constants for the PXR structure-prediction pipeline.

Single source of truth for where things live. Everything is rooted at the
`structure_prediction/` directory so the pipeline is relocatable (e.g. rsync the
whole folder onto the L40S cluster and it still resolves).
"""

from __future__ import annotations

from pathlib import Path

# structure_prediction/ (parent of src/)
ROOT = Path(__file__).resolve().parent.parent

# Inputs
DATA_DIR = ROOT / "data"
STRUCTURE_TEST_CSV = DATA_DIR / "pxr-challenge_structure_TEST_BLINDED.csv"
PXR_FASTA = ROOT / "vendor" / "PXR-Challenge-Tutorial" / "inputs" / "PXR_protein_sequence.fasta"

# Generated Boltz inputs / outputs
BOLTZ_INPUT_DIR = ROOT / "inputs" / "boltz"          # one <id>.yaml per ligand
BOLTZ_OUTPUT_DIR = ROOT / "outputs" / "boltz"        # boltz writes predictions here
MSA_DIR = ROOT / "inputs" / "msa"                    # precomputed PXR MSA (shared by all 184)

# Submission building
SUBMISSION_PDB_DIR = ROOT / "outputs" / "submission_pdbs"   # 184 converted .pdb
SUBMISSION_ZIP = ROOT / "outputs" / "structures.zip"

# Instant baseline (vendored official pre-generated Boltz-2 structures)
VENDOR_EXAMPLE_PDBS = ROOT / "vendor" / "PXR-Challenge-Tutorial" / "outputs" / "example_structure_submission"
BASELINE_ZIP = ROOT / "outputs" / "baseline_structures.zip"

# Official tooling (vendored, reused verbatim — do not reimplement)
VENDOR_TUTORIAL = ROOT / "vendor" / "PXR-Challenge-Tutorial"

# Local calibration set (re-refined PXR crystals with known ligand poses)
REREFINED_DIR = ROOT / "vendor" / "pxr_xtal_re-refinement" / "pxr_rerefined_structures"

# Challenge constants
STRUCTURE_DATASET_SIZE = 184
LIG_RESNAME = "LIG"


def load_pxr_sequence() -> str:
    """Return the single-chain PXR construct sequence (chain A) from the FASTA."""
    lines = PXR_FASTA.read_text().splitlines()
    seq = "".join(ln.strip() for ln in lines if ln and not ln.startswith(">"))
    if not seq:
        raise RuntimeError(f"Empty PXR sequence read from {PXR_FASTA}")
    return seq
