"""Generate one Boltz-2 input YAML per ligand from the blinded structure CSV.

Each YAML follows the official tutorial format (inputs/pxr_x01378-1.yaml):

    version: 1
    sequences:
    - protein: {id: A, sequence: <PXR>, msa: <optional .a3m>}
    - ligand:  {id: B, smiles: <SMILES>}
    properties:
    - affinity: {binder: B}

The PXR protein is identical for all 184 targets, so if you precompute a single
MSA (recommended — see scripts/precompute_msa.sh) every YAML points at it and
Boltz skips the per-target MSA server call.

Run on CPU (Mac is fine):
    python -m src.prep_inputs            # uses --use_msa_server at predict time
    python -m src.prep_inputs --msa inputs/msa/pxr.a3m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src import config


def build_record(sequence: str, smiles: str, msa_path: str | None) -> dict:
    protein: dict = {"id": "A", "sequence": sequence}
    if msa_path:
        protein["msa"] = msa_path
    return {
        "version": 1,
        "sequences": [
            {"protein": protein},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--msa",
        default=None,
        help="Path to a precomputed PXR MSA (.a3m) shared by all targets. "
        "Omit to rely on `boltz predict --use_msa_server` at run time.",
    )
    ap.add_argument("--csv", default=str(config.STRUCTURE_TEST_CSV))
    ap.add_argument("--out", default=str(config.BOLTZ_INPUT_DIR))
    ap.add_argument("--id-col", default="structure", help="id column (test set: structure; calib: pdbid)")
    ap.add_argument("--smi-col", default="smiles")
    args = ap.parse_args()

    sequence = config.load_pxr_sequence()
    df = pd.read_csv(args.csv)

    id_col = args.id_col
    smi_col = args.smi_col
    missing = {id_col, smi_col} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns {missing}; has {list(df.columns)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    bad_smiles: list[str] = []
    for _, row in df.iterrows():
        struct_id = str(row[id_col]).strip()
        smiles = str(row[smi_col]).strip()
        if not struct_id or not smiles or smiles.lower() == "nan":
            bad_smiles.append(struct_id)
            continue
        record = build_record(sequence, smiles, args.msa)
        (out_dir / f"{struct_id}.yaml").write_text(
            yaml.safe_dump(record, sort_keys=False, default_flow_style=False)
        )
        n_written += 1

    print(f"Wrote {n_written} Boltz YAMLs to {out_dir}")
    if n_written != config.STRUCTURE_DATASET_SIZE:
        print(f"WARNING: expected {config.STRUCTURE_DATASET_SIZE}, wrote {n_written}")
    if bad_smiles:
        print(f"WARNING: skipped {len(bad_smiles)} rows with empty id/smiles: {bad_smiles[:10]}")


if __name__ == "__main__":
    main()
