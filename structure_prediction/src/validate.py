"""Validate a structures.zip using the OFFICIAL validator (vendored, verbatim).

Wraps vendor/.../validation/structure_validation.py so our submissions are
checked with exactly the organizer's logic: 184 ids present, single `LIG`
residue per file, <=2 chains, ligand graph isomorphic to the expected SMILES.

    python -m src.validate outputs/structures.zip

Requires: MDAnalysis, rdkit (see env/structure-cpu.yaml).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import config

# Make the vendored official validation package importable
sys.path.insert(0, str(config.VENDOR_TUTORIAL))
from validation.structure_validation import validate_structure_submission  # noqa: E402


def expected_from_csv(csv_path: str) -> tuple[set[str], dict[str, str]]:
    df = pd.read_csv(csv_path)
    ids = {str(x).strip() for x in df["structure"].tolist()}
    smi = {str(r["structure"]).strip(): str(r["smiles"]).strip() for _, r in df.iterrows()}
    return ids, smi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip_path")
    ap.add_argument("--csv", default=str(config.STRUCTURE_TEST_CSV))
    ap.add_argument(
        "--no-smiles-check",
        action="store_true",
        help="Skip the ligand-graph-vs-SMILES isomorphism check (faster, less strict).",
    )
    args = ap.parse_args()

    expected_ids, expected_smiles = expected_from_csv(args.csv)
    ok, errors = validate_structure_submission(
        args.zip_path,
        expected_ids=expected_ids,
        expected_ligand_smiles=None if args.no_smiles_check else expected_smiles,
        require_lig_resname=True,
    )

    if ok:
        print(f"PASS: {args.zip_path} is a valid structure submission.")
    else:
        print(f"FAIL: {args.zip_path} has {len(errors)} problem(s):")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
