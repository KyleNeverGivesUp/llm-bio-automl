"""Data intake (M0, node 1) — load, canonicalize, and report on the dataset.

The competition data is already clean (verified 2026-06-20: all 4139 train + 513
test SMILES parse, 0 duplicate molecules, 0 train/test overlap), so this stage is
deliberately conservative: it **canonicalizes** SMILES to one standard form (so the
same molecule always produces the same features/folds) and writes a
``dataset_report.json`` recording exactly what it saw and changed. It never drops
or imputes anything silently — if something is wrong, the report shows it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SMILES_COL = "SMILES"
NAME_COL = "Molecule Name"
TARGET_COL = "pEC50"


def canonicalize_smiles(s: str) -> str | None:
    """RDKit canonical SMILES, or ``None`` if the string can't be parsed."""
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _canonicalize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a copy with canonical SMILES plus a small report of what changed."""
    df = df.copy()
    original = df[SMILES_COL].astype(str)
    canon = original.map(canonicalize_smiles)
    n_unparsed = int(canon.isna().sum())
    n_changed = int((canon.fillna(original) != original).sum())
    # keep originals where parsing failed (don't silently drop); report instead
    df[SMILES_COL] = canon.fillna(original)
    report = {
        "n_rows": int(len(df)),
        "n_unparsed_smiles": n_unparsed,
        "n_smiles_rewritten_by_canonicalization": n_changed,
        "n_unique_molecules": int(df[SMILES_COL].nunique()),
    }
    return df, report


def prepare_dataset(data_dir: str | Path, out_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load train/test, canonicalize SMILES, and build a dataset report.

    Returns ``(clean_train, clean_test, dataset_report)``. If ``out_path`` is given,
    the report is written there as JSON.
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    clean_train, train_rep = _canonicalize_frame(train)
    clean_test, test_rep = _canonicalize_frame(test)

    train_set = set(clean_train[SMILES_COL])
    test_set = set(clean_test[SMILES_COL])
    report = {
        "data_dir": str(data_dir),
        "train": {
            **train_rep,
            "n_duplicate_molecules": int(len(clean_train) - clean_train[SMILES_COL].nunique()),
            "target_present": TARGET_COL in clean_train.columns,
            "target_nulls": int(clean_train[TARGET_COL].isna().sum()) if TARGET_COL in clean_train else None,
        },
        "test": test_rep,
        "train_test_overlap_molecules": int(len(train_set & test_set)),
    }
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return clean_train, clean_test, report


if __name__ == "__main__":
    _, _, rep = prepare_dataset("data/pxr_activity", out_path="outputs/m1_menu/dataset_report.json")
    print(json.dumps(rep, indent=2))
