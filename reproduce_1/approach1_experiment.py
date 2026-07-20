"""#1 (matcha-croissant) Approach 1, faithfully: the multitask-GNN base pillar.

#1's submission = a meta-ensemble of 3 approaches; Approach 1 is the base and carries most of the
weight. It is a multitask GNN with primary pEC50 + single-concentration log2fc (all usable doses)
+ LogD as auxiliary tasks. Versus our mt5 baseline this adds:
  - single-concentration as SEPARATE per-dose heads (8.25/33/99 uM), trained on ALL ~10,870
    screened molecules (the multi-fidelity aux signal), not one summarized head;
  - a LogD auxiliary head, trained on the public MoleculeNet Lipophilicity set (4,200 molecules;
    a public stand-in for #1's PROPRIETARY LogD — flagged, not hidden).

Not yet included (later sub-steps / known-to-hurt-us, tracked separately): precomputed descriptors
(#1 adds them; step 1b), phase-1 hyperparameter re-opt, and phase-1 fold-in (we measured fold-in
hurts our pipeline). Reactive-electrophile exclusion is off (#1 did not mention it for Approach 1).

Data: data/pxr_activity/train_approach1.csv (16,474 rows; the 4,139 pEC50-labelled broad molecules
first, then single-conc-only and LogD-only auxiliary rows) + test.csv + folds_calibrated.json.
5-fold OOF on the broad rows (aux rows are always in training); predicts pEC50 for the 513 test.

Run on the GPU pod:
    nohup python scripts/approach1_experiment.py > approach1.log 2>&1 &
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.metrics import score_all  # noqa: E402

RDLogger.DisableLog("rdApp.*")
SMILES, NAME, PRIMARY = "SMILES", "Molecule Name", "pEC50"
TARGETS = ["pEC50", "fc_8p25", "fc_33", "fc_99", "logD"]

# 1b: precomputed chemical descriptors (#1 "supplemented with precomputed chemical descriptors").
_DESC_NAMES = [n for n, _ in Descriptors._descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)


def _save_descriptors(csv_path: Path, npz_path: Path) -> None:
    """Compute the ~208 RDKit 2D descriptors for each molecule in csv_path (row-aligned) and save as
    the .npz chemprop's --descriptors-path expects (it reads arr_0 and splits by its own indices)."""
    sm = pd.read_csv(csv_path)[SMILES].tolist()
    X = np.zeros((len(sm), len(_DESC_NAMES)), dtype=np.float32)
    for i, s in enumerate(sm):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            X[i] = _DESC_CALC.CalcDescriptors(m)
    np.savez(npz_path, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))


def _primary(csv: Path) -> np.ndarray:
    df = pd.read_csv(csv)
    if PRIMARY in df.columns:
        return pd.to_numeric(df[PRIMARY], errors="coerce").to_numpy(float)
    for c in df.columns:
        if c != SMILES and pd.api.types.is_numeric_dtype(df[c]):
            return df[c].to_numpy(float)
    raise SystemExit(f"no prediction column in {csv}")


def train_fold(train_csv: Path, out_dir: Path, epochs: int, batch: int, accel: str, workers: int,
               desc_npz: Path | None = None) -> str:
    cmd = ["chemprop", "train", "-i", str(train_csv), "-s", SMILES, "--target-columns", *TARGETS,
           "-t", "regression", "--from-foundation", "CheMeleon", "--loss-function", "mae",
           "--epochs", str(epochs), "--warmup-epochs", str(max(1, min(2, epochs - 1))),
           "--split-sizes", "0.9", "0.1", "0.0", "--batch-size", str(batch),
           "-o", str(out_dir), "--accelerator", accel, "-n", str(workers)]
    if desc_npz is not None:
        _save_descriptors(train_csv, desc_npz)
        cmd += ["--descriptors-path", str(desc_npz)]
    subprocess.run(cmd, check=True)
    ckpts = (glob.glob(str(out_dir / "**" / "best*.ckpt"), recursive=True)
             or glob.glob(str(out_dir / "**" / "*.ckpt"), recursive=True))
    if not ckpts:
        raise SystemExit(f"no checkpoint under {out_dir}")
    return ckpts[0]


def predict(ckpt: str, in_csv: Path, out_csv: Path, accel: str, workers: int,
            desc_npz: Path | None = None) -> np.ndarray:
    cmd = ["chemprop", "predict", "--test-path", str(in_csv), "-s", SMILES,
           "--model-paths", ckpt, "--preds-path", str(out_csv), "--accelerator", accel, "-n", str(workers)]
    if desc_npz is not None:
        _save_descriptors(in_csv, desc_npz)
        cmd += ["--descriptors-path", str(desc_npz)]
    subprocess.run(cmd, check=True)
    return _primary(out_csv)


def score(names, pred, labels, tag) -> float:
    m = labels.merge(pd.DataFrame({NAME: names, "yhat": pred}), on=NAME, how="left")
    if m["yhat"].isna().any():
        raise SystemExit(f"{tag}: missing predictions")
    s = score_all(m[TGT].to_numpy(float), m["yhat"].to_numpy(float))
    print(f"    {tag:40s} RAE={s['rae']:.4f}  MAE={s['mae']:.4f}  R2={s['r2']:.4f}  (n={len(m)})")
    return s["rae"]


TGT = PRIMARY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/pxr_activity")
    ap.add_argument("--pool", type=Path, default=REPO / "data/pxr_activity/train_approach1.csv")
    ap.add_argument("--out", type=Path, default=Path("/tmp/approach1"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--descriptors", action="store_true",
                    help="1b: add ~208 precomputed RDKit descriptors as extra GNN input (#1 does this)")
    args = ap.parse_args()
    dtag = " + descriptors(1b)" if args.descriptors else ""

    args.out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(args.pool)
    broad = pool[pool[PRIMARY].notna()].reset_index(drop=True)        # the 4,139 pEC50-labelled rows
    extra = pool[pool[PRIMARY].isna()].reset_index(drop=True)         # sc-only + logD-only aux (always train)
    import json
    folds = json.loads((args.data_dir / "folds_calibrated.json").read_text())["assignments"]
    fold_of = np.array([int(folds[str(i)]) for i in range(len(broad))])
    print(f"[data] broad(pEC50)={len(broad)}  aux(always-train)={len(extra)}  folds={sorted(set(fold_of))}")

    test = pd.read_csv(args.data_dir / "test.csv").reset_index(drop=True)
    test[[SMILES]].to_csv(args.out / "_test_in.csv", index=False)
    oof = np.full(len(broad), np.nan)
    test_cols = []

    for k in sorted(set(fold_of.tolist())):
        print(f"\n===== FOLD {k} =====", flush=True)
        tr_mask = fold_of != k
        fold_train = pd.concat([broad[tr_mask][[SMILES, *TARGETS]], extra[[SMILES, *TARGETS]]], ignore_index=True)
        fold_train.to_csv(args.out / f"_train_f{k}.csv", index=False)
        broad[fold_of == k][[SMILES]].to_csv(args.out / f"_held_f{k}.csv", index=False)
        d_tr = args.out / f"_train_f{k}_desc.npz" if args.descriptors else None
        d_held = args.out / f"_held_f{k}_desc.npz" if args.descriptors else None
        d_test = args.out / f"_test_f{k}_desc.npz" if args.descriptors else None
        ckpt = train_fold(args.out / f"_train_f{k}.csv", args.out / f"ckpt_f{k}",
                          args.epochs, args.batch_size, args.accelerator, args.num_workers, d_tr)
        oof[fold_of == k] = predict(ckpt, args.out / f"_held_f{k}.csv", args.out / f"_oof_f{k}.csv",
                                    args.accelerator, args.num_workers, d_held)
        test_cols.append(predict(ckpt, args.out / "_test_in.csv", args.out / f"_test_f{k}.csv",
                                 args.accelerator, args.num_workers, d_test))

    # write OOF + averaged test predictions (suffix so the descriptors run doesn't overwrite the base)
    suffix = "_desc" if args.descriptors else ""
    done = ~np.isnan(oof)
    pd.DataFrame({"row_id": np.arange(len(broad))[done], SMILES: broad[SMILES][done],
                  "y_true": broad[PRIMARY][done], "y_pred": oof[done]}).to_csv(
        args.out / f"oof_approach1{suffix}.csv", index=False)
    test_pred = np.mean(np.column_stack(test_cols), axis=1)
    pd.DataFrame({NAME: test[NAME], SMILES: test[SMILES], PRIMARY: test_pred}).to_csv(
        args.out / f"test_approach1{suffix}.csv", index=False)

    # score — #1 uses NO calibration, so we report the raw multitask prediction only.
    p2 = pd.read_csv(args.data_dir / "phase2_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])
    print(f"\n=== Approach-1{dtag} Set-2 (260 blind) — no calibration ===")
    score(test[NAME], test_pred, p2, f"Approach-1{dtag}")
    print("  base Approach-1 (no descriptors) was 0.5827 | #1 (matcha) 0.5631 | #2 (best public) 0.5676")


if __name__ == "__main__":
    main()
