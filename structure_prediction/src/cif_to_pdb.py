"""Convert Boltz-2 output mmCIF complexes into challenge-format submission PDBs.

Target format (verified against the official example submission):
  - chain A = PXR monomer, ATOM records
  - chain B = ligand, HETATM records, residue name exactly `LIG`

Boltz writes one protein chain + one ligand chain (we only feed a monomer, so no
homodimer issue). This step finds the non-polymer residue, renames it to `LIG`,
and writes a standard PDB. Connectivity is re-perceived from geometry by RDKit
in the validator, so good Boltz geometry is all that's needed.

Runs wherever the .cif files are (L40S after boltz, or Mac if copied back).
Requires `gemmi`.

    python -m src.cif_to_pdb                       # auto-discovers boltz outputs
    python -m src.cif_to_pdb --boltz-out outputs/boltz --out outputs/submission_pdbs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gemmi

from src import config

_STD_SKIP = {"HOH", "WAT", "DOD"}


def _is_standard_polymer(resname: str) -> bool:
    info = gemmi.find_tabulated_residue(resname)
    return info is not None and (info.is_amino_acid() or info.is_nucleic_acid())


def convert_one(cif_path: Path, out_pdb: Path, lig_resname: str = config.LIG_RESNAME) -> None:
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()

    n_lig_res = 0
    for model in st:
        for chain in model:
            for res in chain:
                if _is_standard_polymer(res.name) or res.name in _STD_SKIP:
                    continue
                # Treat as the small-molecule ligand
                res.name = lig_resname
                res.het_flag = "H"
                n_lig_res += 1
    if n_lig_res == 0:
        raise RuntimeError(f"No ligand residue found in {cif_path}")
    if n_lig_res > 1:
        # Multiple non-polymer residues — collapsing them all to LIG is wrong.
        # Surface it rather than silently producing an invalid file.
        raise RuntimeError(f"{cif_path}: found {n_lig_res} non-polymer residues, expected 1")

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_pdb))


def discover_cif(boltz_out: Path, struct_id: str) -> Path | None:
    """Find the top-ranked predicted cif for a structure id under boltz out dir.

    Boltz layout: <out>/predictions/<name>/<name>_model_0.cif (rank 0 = best).
    """
    candidates = sorted(boltz_out.rglob(f"{struct_id}*_model_0.cif"))
    if not candidates:
        candidates = sorted(boltz_out.rglob(f"{struct_id}*.cif"))
    return candidates[0] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boltz-out", default=str(config.BOLTZ_OUTPUT_DIR))
    ap.add_argument("--out", default=str(config.SUBMISSION_PDB_DIR))
    ap.add_argument("--csv", default=str(config.STRUCTURE_TEST_CSV))
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.csv)
    ids = [str(x).strip() for x in df["structure"].tolist()]

    boltz_out = Path(args.boltz_out)
    out_dir = Path(args.out)

    ok, missing, failed = 0, [], []
    for struct_id in ids:
        cif = discover_cif(boltz_out, struct_id)
        if cif is None:
            missing.append(struct_id)
            continue
        try:
            convert_one(cif, out_dir / f"{struct_id}.pdb")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((struct_id, str(exc)))

    print(f"Converted {ok}/{len(ids)} -> {out_dir}")
    if missing:
        print(f"MISSING boltz output for {len(missing)}: {missing[:10]}")
    if failed:
        print(f"FAILED conversion for {len(failed)}: {failed[:5]}")


if __name__ == "__main__":
    main()
