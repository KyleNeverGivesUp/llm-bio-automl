"""Build an instant, submittable baseline from the official pre-generated PDBs.

The tutorial ships 184 pre-generated Boltz-2 complexes already in submission
format (vendor/.../outputs/example_structure_submission/). This zips them into a
valid structures.zip so there is a working submission on day 0 — a floor to beat,
and a way to confirm the whole validate/submit path works before we run our own
Boltz on the L40S.

    python -m src.make_baseline_submission            # writes outputs/baseline_structures.zip
"""

from __future__ import annotations

import zipfile

import pandas as pd

from src import config


def main() -> None:
    src_dir = config.VENDOR_EXAMPLE_PDBS
    pdbs = sorted(src_dir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No vendored example PDBs at {src_dir}")

    config.BASELINE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(config.BASELINE_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pdbs:
            zf.write(p, arcname=p.name)
    print(f"Wrote baseline {config.BASELINE_ZIP} ({len(pdbs)} pdbs)")

    # Coverage check vs the 184 expected ids
    df = pd.read_csv(config.STRUCTURE_TEST_CSV)
    expected = {str(x).strip() for x in df["structure"].tolist()}
    have = {p.stem for p in pdbs}
    missing = sorted(expected - have)
    extra = sorted(have - expected)
    print(f"coverage: {len(have & expected)}/{len(expected)} expected ids present")
    if missing:
        print(f"  MISSING {len(missing)}: {missing[:10]}")
    if extra:
        print(f"  EXTRA (not in test set) {len(extra)}: {extra[:10]}")


if __name__ == "__main__":
    main()
