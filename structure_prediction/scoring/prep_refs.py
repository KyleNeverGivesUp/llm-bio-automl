"""Build a local calibration set from the re-refined PXR crystal structures.

For each crystal we:
  1. find the bound drug-like ligand (largest non-polymer, non-solvent residue),
  2. write a reference complex PDB with that ligand renamed `LIG` (so the OFFICIAL
     OST scorer, which selects `rname=LIG`, works unchanged),
  3. look up the ligand SMILES from RCSB by its 3-letter component code (so we can
     run Boltz on the SAME ligand and score predicted-vs-crystal).

Output:
  scoring/refs/<pdbid>.pdb          reference complex (protein + LIG)
  scoring/calibration_manifest.csv  pdbid, lig_code, n_heavy, smiles

This is the local proxy that replaces the blinded 184 for tuning (which tool /
prep / consensus). Runs on CPU (gemmi + network). OST is only needed later, by
score_local.py.

    python -m scoring.prep_refs
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import gemmi
import pandas as pd

from src import config

REFS_DIR = config.ROOT / "scoring" / "refs"
MANIFEST = config.ROOT / "scoring" / "calibration_manifest.csv"

# residues that are never the ligand of interest
_SOLVENT_IONS = {
    "HOH", "WAT", "DOD", "GOL", "EDO", "PEG", "PG4", "1PE", "SO4", "PO4", "ACT",
    "DMS", "MES", "TRS", "EPE", "CL", "NA", "K", "MG", "ZN", "CA", "MN", "FMT",
    "IOD", "BR", "NO3", "FLC", "CIT", "BME", "IMD",
}
_MIN_HEAVY = 8  # drug/fragment-like cutoff; we still pick the single LARGEST


def _is_polymer(resname: str) -> bool:
    info = gemmi.find_tabulated_residue(resname)
    return info is not None and (info.is_amino_acid() or info.is_nucleic_acid())


def find_ligand(st: gemmi.Structure):
    """Return (chain_name, residue) of the largest drug-like ligand, or None."""
    best = None
    model = st[0]
    for chain in model:
        for res in chain:
            if _is_polymer(res.name) or res.name in _SOLVENT_IONS:
                continue
            n_heavy = sum(1 for a in res if a.element != gemmi.Element("H"))
            if n_heavy < _MIN_HEAVY:
                continue
            if best is None or n_heavy > best[0]:
                best = (n_heavy, chain.name, res.name, res.seqid)
    return best


def build_reference(pdb_path: Path, out_path: Path):
    """Rename exactly ONE bound-ligand copy to `LIG` and write the reference.

    No residue surgery: the OST scorer only ``Select("rname=LIG")``, so leaving
    the protein (incl. crystallographic dimer), waters and other hetero groups in
    place is harmless — we just need a single `LIG` residue.
    """
    st = gemmi.read_structure(str(pdb_path))
    st.setup_entities()
    found = find_ligand(st)
    if found is None:
        return None
    n_heavy, lig_chain, lig_resname, lig_seqid = found

    model = st[0]
    renamed = False
    for chain in model:
        if chain.name != lig_chain:
            continue
        for res in chain:
            if not renamed and res.name == lig_resname and res.seqid == lig_seqid:
                res.name = config.LIG_RESNAME
                res.het_flag = "H"
                renamed = True
                break
    if not renamed:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_path))
    return lig_resname, n_heavy


def fetch_smiles(lig_code: str, cache: dict) -> str | None:
    if lig_code in cache:
        return cache[lig_code]
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{lig_code}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        desc = data.get("rcsb_chem_comp_descriptor", {})
        smiles = desc.get("smiles_stereo") or desc.get("smiles")
        if not smiles:
            # fall back to the PDBx descriptor list
            for d in data.get("pdbx_chem_comp_descriptor", []):
                if d.get("type", "").upper().startswith("SMILES"):
                    smiles = d.get("descriptor")
                    break
    except Exception:
        smiles = None
    cache[lig_code] = smiles
    return smiles


def main() -> None:
    src_root = config.REREFINED_DIR
    if not src_root.exists():
        raise SystemExit(f"Re-refined structures not found at {src_root} (clone the repo).")

    cache: dict[str, str | None] = {}
    rows = []
    for d in sorted(src_root.iterdir()):
        pdb = d / f"{d.name}.pdb"
        if not pdb.exists():
            continue
        try:
            res = build_reference(pdb, REFS_DIR / f"{d.name}.pdb")
        except Exception as exc:  # noqa: BLE001
            print(f"  {d.name}: FAILED ({exc})")
            continue
        if res is None:
            print(f"  {d.name}: no drug-like ligand found")
            continue
        lig_code, n_heavy = res
        smiles = fetch_smiles(lig_code, cache)
        rows.append({"pdbid": d.name, "lig_code": lig_code, "n_heavy": n_heavy, "smiles": smiles})

    out = pd.DataFrame(rows)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MANIFEST, index=False)
    n_smiles = int(out["smiles"].notna().sum()) if len(out) else 0
    print(f"\nReference complexes: {len(out)} written to {REFS_DIR}")
    print(f"SMILES resolved:     {n_smiles}/{len(out)}")
    print(f"Manifest:            {MANIFEST}")


if __name__ == "__main__":
    main()
