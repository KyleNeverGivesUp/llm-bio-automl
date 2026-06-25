"""Package converted PDBs into a challenge-ready structures.zip and validate it.

The zip must contain EXACTLY 184 .pdb files at the top level (no subdirs, no
extra files), named <structure_id>.pdb. This collects them, zips, and runs the
official validator.

    python -m src.build_submission                       # from outputs/submission_pdbs
    python -m src.build_submission --pdb-dir <dir> --zip outputs/structures.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src import config


def build_zip(pdb_dir: Path, zip_path: Path, expected_ids: list[str]) -> tuple[int, list[str]]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    present, missing = [], []
    for sid in expected_ids:
        p = pdb_dir / f"{sid}.pdb"
        (present if p.exists() else missing).append(sid)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sid in present:
            zf.write(pdb_dir / f"{sid}.pdb", arcname=f"{sid}.pdb")  # flat, no subdirs
    return len(present), missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-dir", default=str(config.SUBMISSION_PDB_DIR))
    ap.add_argument("--zip", default=str(config.SUBMISSION_ZIP))
    ap.add_argument("--csv", default=str(config.STRUCTURE_TEST_CSV))
    ap.add_argument("--validate", action="store_true", help="Run official validator after zipping.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    expected_ids = [str(x).strip() for x in df["structure"].tolist()]

    n, missing = build_zip(Path(args.pdb_dir), Path(args.zip), expected_ids)
    print(f"Zipped {n}/{len(expected_ids)} pdbs -> {args.zip}")
    if missing:
        print(f"MISSING {len(missing)} structures (zip is INCOMPLETE): {missing[:10]}")

    if args.validate:
        from src.validate import expected_from_csv
        import sys

        sys.path.insert(0, str(config.VENDOR_TUTORIAL))
        from validation.structure_validation import validate_structure_submission

        ids, smiles = expected_from_csv(args.csv)
        ok, errors = validate_structure_submission(
            args.zip, expected_ids=ids, expected_ligand_smiles=smiles
        )
        print("VALIDATION:", "PASS" if ok else f"FAIL ({len(errors)} errors)")
        for e in errors[:20]:
            print("  -", e)


if __name__ == "__main__":
    main()
