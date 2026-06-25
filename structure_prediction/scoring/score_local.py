"""Score predicted complexes against the local PXR crystal calibration set.

Reuses the OFFICIAL OST scorer verbatim (vendor/.../evaluation), so local numbers
are directly comparable to the challenge: primary LDDT-PLI (up), secondary
BiSyRMSD (down), plus LDDT-LP and coverage, bootstrapped over 1000 resamples.

This is the calibration signal that replaces the blinded 184 for choosing tool /
prep / consensus settings.

Requires OpenStructure (`ost`). Install on the cluster, e.g.:
    apptainer pull ost.sif docker://registry.scicore.unibas.ch/schwede/openstructure:latest
    apptainer exec ost.sif python -m scoring.score_local --pred-dir <dir>
or conda: `conda install -c conda-forge openstructure`.

    python -m scoring.score_local --pred-dir outputs/calib_pdbs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import config

# Import the official evaluation package (namespace package; no __init__ needed)
sys.path.insert(0, str(config.VENDOR_TUTORIAL))
from evaluation.evaluate_predictions import (  # noqa: E402
    average_bootstrap_results_by_endpoint,
    bootstrap_structure_metrics,
    score_structure_predictions,
)
from evaluation.config import BOOTSTRAP_SAMPLES  # noqa: E402

MANIFEST = config.ROOT / "scoring" / "calibration_manifest.csv"
REFS_DIR = config.ROOT / "scoring" / "refs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, help="dir of predicted <pdbid>.pdb complexes")
    ap.add_argument("--refs", default=str(REFS_DIR))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out", default=str(config.ROOT / "scoring" / "calib_scores.csv"))
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    pred_dir, refs_dir = Path(args.pred_dir), Path(args.refs)

    predicted, refs = {}, {}
    for pdbid in manifest["pdbid"].astype(str):
        p, r = pred_dir / f"{pdbid}.pdb", refs_dir / f"{pdbid}.pdb"
        if p.exists() and r.exists():
            predicted[pdbid] = str(p)
            refs[pdbid] = str(r)

    if not predicted:
        raise SystemExit(f"No predicted/ref pairs found under {pred_dir} and {refs_dir}")

    per_compound = score_structure_predictions(predicted, refs)
    per_compound.to_csv(args.out, index=False)

    boot = bootstrap_structure_metrics(per_compound, BOOTSTRAP_SAMPLES)
    agg = average_bootstrap_results_by_endpoint(boot)

    print(f"\nScored {len(predicted)} predicted vs crystal references")
    print(agg.to_string())
    cov = float(per_compound["coverage"].mean())
    print(f"\nCoverage: {cov:.3f}")
    print(f"Per-compound scores: {args.out}")


if __name__ == "__main__":
    main()
