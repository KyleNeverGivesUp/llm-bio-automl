"""Scaffold-based cross-validation folds — our honest local validation.

Why scaffold split?
-------------------
"Cross-validation" means we slice the training data into a few parts ("folds"),
train on some and test on the held-out one, then average — a local "practice
score" before we ever submit.

The catch: if two molecules share the same core skeleton ("scaffold") and one
lands in the training part while the other lands in the held-out part, the model
can essentially memorize the skeleton and look better than it truly is
("leakage"). The hidden competition test set contains *new* chemistry, so a
random split would flatter us and we'd pick the wrong model.

A *scaffold split* keeps every skeleton entirely inside a single fold. The
practice score then reflects performance on unseen skeletons — honest, and
aligned with how the final ranking judges us.

This module is deterministic: same data in -> same folds out (reproducible).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # silence RDKit's noisy parse warnings


def bemis_murcko_scaffold(smiles: str) -> str | None:
    """Return the Bemis-Murcko scaffold (the core ring system) as a SMILES string.

    Returns "" for molecules with no ring system, or None if the SMILES is
    unparseable.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def assign_scaffold_folds(
    smiles_list: list[str], n_folds: int = 5, seed: int = 42
) -> tuple[list[int], dict]:
    """Assign each molecule to one of ``n_folds`` folds, keeping molecules that
    share a scaffold together.

    Method: group rows by scaffold, then greedily place the largest groups first
    into whichever fold is currently smallest (balanced bin-packing). This keeps
    folds roughly equal in size while guaranteeing a scaffold never spans folds.
    Deterministic: ties are broken by the scaffold string.

    Returns ``(fold_of_row, diagnostics)`` where ``fold_of_row[i]`` is the fold
    index (0..n_folds-1) for row ``i``.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    n_unparsed = 0
    for i, smi in enumerate(smiles_list):
        scaffold = bemis_murcko_scaffold(smi)
        if scaffold is None:
            key = f"__unparsed__{i}"  # keep it as its own singleton group
            n_unparsed += 1
        elif scaffold == "":
            key = f"__noscaffold__{i}"  # acyclic molecule -> its own group
        else:
            key = scaffold
        groups[key].append(i)

    # largest groups first; tie-break by key so the result is reproducible
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    fold_sizes = [0] * n_folds
    fold_of_row: list[int] = [-1] * len(smiles_list)
    for key, idxs in ordered:
        target = min(range(n_folds), key=lambda k: fold_sizes[k])
        for i in idxs:
            fold_of_row[i] = target
        fold_sizes[target] += len(idxs)

    diagnostics = {
        "strategy": "scaffold",
        "n_molecules": len(smiles_list),
        "n_folds": n_folds,
        "seed": seed,  # recorded for reproducibility; the split itself is deterministic
        "n_unique_scaffold_groups": len(groups),
        "n_unparsed_smiles": n_unparsed,
        "fold_sizes": fold_sizes,
    }
    return fold_of_row, diagnostics


# --------------------------------------------------------------------------- #
# Analog-faithful folds: Tanimoto-similarity cluster folds
# --------------------------------------------------------------------------- #
# Why a second fold design? The 513-compound test set is an *analog* set — close
# Tanimoto neighbours of 63 active hits, with activity cliffs and a tighter target
# spread than the broad training data. Bemis-Murcko scaffold folds only keep
# *identical* core skeletons together, so the held-out fold is often easier than
# the real analog task — measured: scaffold-CV says 0.543 while the Set-1 judge
# says 0.633. Clustering by ECFP Tanimoto keeps *similar* (not just identical)
# molecules together; the ``cutoff`` knob tunes how strict that is, which lets us
# calibrate the CV difficulty until it ranks models like the judge.
#
# Leakage note: this partitions ONLY the rows passed in (the broad 4,139). No
# external labels are used — clustering is on structure (SMILES) alone.

def _ecfp_bitvects(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> list:
    """ECFP/Morgan bit-vector per molecule (None-safe: unparseable -> empty vector,
    which is Tanimoto-distance 1 from everything, i.e. its own singleton cluster)."""
    from rdkit.DataStructs import ExplicitBitVect
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(gen.GetFingerprint(mol) if mol is not None else ExplicitBitVect(n_bits))
    return fps


def assign_cluster_folds(
    smiles_list: list[str],
    n_folds: int = 5,
    cutoff: float = 0.6,
    radius: int = 2,
    n_bits: int = 2048,
    seed: int = 42,
) -> tuple[list[int], dict]:
    """Cluster molecules by ECFP Tanimoto (Butina), then pack whole clusters into
    balanced folds — so a cluster of mutually-similar molecules never spans folds.

    ``cutoff`` is a Tanimoto *distance* threshold (1 − similarity): two molecules
    join the same cluster when their distance is below it. Smaller cutoff -> tighter,
    more numerous clusters -> easier folds; larger cutoff -> coarser clusters ->
    harder folds. This is the calibration knob for matching the Set-1 judge.

    Returns ``(fold_of_row, diagnostics)`` like ``assign_scaffold_folds``.
    """
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    n = len(smiles_list)
    fps = _ecfp_bitvects(smiles_list, radius=radius, n_bits=n_bits)

    # Condensed lower-triangle Tanimoto distances in the order Butina expects.
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    # Largest clusters first; tie-break by smallest member index for determinism.
    ordered = sorted(clusters, key=lambda c: (-len(c), min(c)))

    fold_sizes = [0] * n_folds
    fold_of_row = [-1] * n
    for cluster in ordered:
        target = min(range(n_folds), key=lambda k: fold_sizes[k])
        for i in cluster:
            fold_of_row[i] = target
        fold_sizes[target] += len(cluster)

    diagnostics = {
        "strategy": "cluster",
        "n_molecules": n,
        "n_folds": n_folds,
        "seed": seed,
        "cluster_cutoff": cutoff,
        "ecfp_radius": radius,
        "ecfp_n_bits": n_bits,
        "n_clusters": len(clusters),
        "n_singletons": sum(1 for c in clusters if len(c) == 1),
        "largest_cluster": max((len(c) for c in clusters), default=0),
        "fold_sizes": fold_sizes,
    }
    return fold_of_row, diagnostics


def verify_no_scaffold_leakage(
    smiles_list: list[str], fold_of_row: list[int]
) -> int:
    """Sanity check: confirm no real scaffold appears in more than one fold.

    Returns the number of scaffolds that leak across folds (should be 0).
    """
    scaffold_to_folds: dict[str, set[int]] = defaultdict(set)
    for i, smi in enumerate(smiles_list):
        scaffold = bemis_murcko_scaffold(smi)
        if scaffold:  # ignore unparsed / acyclic singletons
            scaffold_to_folds[scaffold].add(fold_of_row[i])
    return sum(1 for folds in scaffold_to_folds.values() if len(folds) > 1)


def save_folds(fold_of_row: list[int], diagnostics: dict, out_path: str | Path) -> None:
    """Write the frozen fold assignment + diagnostics to ``folds.json``."""
    payload = {
        **diagnostics,
        "assignments": {str(i): int(f) for i, f in enumerate(fold_of_row)},
    }
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_scaffold_folds_for_csv(
    csv_path: str | Path,
    smiles_col: str = "SMILES",
    n_folds: int = 5,
    seed: int = 42,
    out_path: str | Path | None = None,
) -> dict:
    """Convenience entry point: read a CSV, build scaffold folds, verify, and
    (optionally) save. Returns the diagnostics dict (with a leakage check)."""
    df = pd.read_csv(csv_path)
    smiles = df[smiles_col].tolist()
    fold_of_row, diagnostics = assign_scaffold_folds(smiles, n_folds=n_folds, seed=seed)
    diagnostics["scaffolds_leaking_across_folds"] = verify_no_scaffold_leakage(
        smiles, fold_of_row
    )
    if out_path is not None:
        save_folds(fold_of_row, diagnostics, out_path)
    return diagnostics


if __name__ == "__main__":
    diag = build_scaffold_folds_for_csv(
        "data/pxr_activity/train.csv",
        n_folds=5,
        out_path="data/pxr_activity/folds.json",
    )
    print("Scaffold cross-validation folds built:")
    print(f"  molecules                : {diag['n_molecules']}")
    print(f"  folds                    : {diag['n_folds']}")
    print(f"  unique scaffold groups   : {diag['n_unique_scaffold_groups']}")
    print(f"  fold sizes               : {diag['fold_sizes']}")
    print(f"  unparsed SMILES          : {diag['n_unparsed_smiles']}")
    print(f"  scaffolds leaking folds  : {diag['scaffolds_leaking_across_folds']}  (must be 0)")
    print("  saved -> data/pxr_activity/folds.json")
