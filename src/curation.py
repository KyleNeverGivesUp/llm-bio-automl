"""Training-set curation — reactive-electrophile exclusion (0.538-solution lever).

The biochemical PXR assay flags compounds that *react* (covalently / promiscuously)
as false positives, and those reactive electrophiles do not appear in the blinded
test set. The 0.538 write-up removed ~754 such compounds (acrylamides, acrylates,
aldehydes, ...) and gained ~0.019 RAE. We reproduce that here as an explicit,
auditable SMARTS panel — and, per their finding, we deliberately do NOT remove
PAINS/REOS compounds (those anchor the inactive end and *helped*).

Used by the curation experiment and the multitask CSV rebuild. Set 1 is never
touched here (the judge harness merges it separately).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Reactive-electrophile SMARTS. Names mirror the write-up's three big buckets plus
# the standard covalent-warhead set; each is a structural alert for assay-reactive
# behaviour, not a drug-likeness filter.
REACTIVE_SMARTS: dict[str, str] = {
    "acrylamide": "[CX3]=[CX3]C(=O)[NX3]",
    "acrylate_ester": "[CX3]=[CX3]C(=O)[OX2H0]",
    "acrylic_acid": "[CX3]=[CX3]C(=O)[OX2H1]",
    "vinyl_ketone": "[CX3]=[CX3]C(=O)[#6;!$([CX3]=O)]",   # Michael acceptor (enone)
    "aldehyde": "[CX3H1](=O)[#6]",                          # excl. formaldehyde/formic
    "maleimide": "O=C1C=CC(=O)N1",
    "vinyl_sulfone": "[CX3]=[CX3][SX4](=O)(=O)",
    "alpha_halo_carbonyl": "[CX3](=O)[CX4;H1,H2][F,Cl,Br,I]",
    "epoxide": "[CX3]1[OX2][CX3]1",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "michael_nitrile": "[CX3]=[CX3]C#N",                   # acrylonitrile-type
}

_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in REACTIVE_SMARTS.items()}


def reactive_alerts(smiles: str) -> list[str]:
    """Return the names of reactive-electrophile alerts a molecule matches (empty if clean)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return []
    return [name for name, patt in _PATTERNS.items() if patt is not None and mol.HasSubstructMatch(patt)]


def is_reactive(smiles: str) -> bool:
    return len(reactive_alerts(smiles)) > 0


def reactive_mask(smiles_list) -> np.ndarray:
    """Boolean mask, True where the molecule trips at least one reactive alert."""
    return np.array([is_reactive(s) for s in smiles_list], dtype=bool)


def curate_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    smiles_col: str = "SMILES",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Drop reactive-electrophile rows from ``train_df``; never drops anything in ``test_df``.

    Returns ``(curated_df, report)``. ``report`` carries per-alert counts and how many
    test compounds (if any) would have tripped the same alerts — the write-up's claim is
    that number is ~0, which we verify rather than assume.
    """
    mask = reactive_mask(train_df[smiles_col].tolist())
    # per-alert tally
    alerts = [reactive_alerts(s) for s in train_df[smiles_col]]
    by_alert: dict[str, int] = {}
    for al in alerts:
        for name in al:
            by_alert[name] = by_alert.get(name, 0) + 1

    test_hits = None
    if test_df is not None:
        test_mask = reactive_mask(test_df[smiles_col].tolist())
        test_hits = int(test_mask.sum())

    curated = train_df[~mask].reset_index(drop=True)
    report = {
        "n_train_in": int(len(train_df)),
        "n_removed": int(mask.sum()),
        "n_train_out": int(len(curated)),
        "by_alert": dict(sorted(by_alert.items(), key=lambda kv: -kv[1])),
        "n_test_reactive": test_hits,  # should be ~0 per the write-up
    }
    if verbose:
        print(f"[curate] removed {report['n_removed']}/{report['n_train_in']} reactive "
              f"(test reactive: {test_hits}); by alert: {report['by_alert']}")
    return curated, report
