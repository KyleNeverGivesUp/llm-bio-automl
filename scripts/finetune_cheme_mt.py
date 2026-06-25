"""MULTITASK fine-tune of CheMeleon on PXR (run on a GPU box, e.g. A5000).

The documented competitor jump (RAE ~0.689 -> ~0.60) came from training the graph
model MULTITASK: predict the primary pEC50 *and* the PXR-null counter-screen pEC50
+ both Emax endpoints jointly. The shared encoder learns to separate true PXR
binding from assay interference. We score / stack only the PRIMARY pEC50.

Difference vs the single-task `finetune_cheme.py`: trains with 4 target columns
(masked NaN where the counter readout is missing) and extracts the pEC50 head.

Produces (same format our pipeline consumes):
  - oof_cheme_mt.csv   (row_id, SMILES, y_true, y_pred)   leak-free 5-fold OOF, primary pEC50
  - test_cheme_mt.csv  (Molecule Name, SMILES, pEC50)     513 test preds (mean of 5 folds)

**Set 1 never touched** — trains broad rows only, split by the calibrated cluster folds.

Needs (same folder): train_multitask.csv, test.csv, folds_calibrated.json  +  `pip install 'chemprop>=2.2.0'`
Run:  python finetune_cheme_mt.py --epochs 50 --accelerator gpu
Quick test:  python finetune_cheme_mt.py --max-rows 200 --folds 2 --epochs 5 --accelerator gpu
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SMILES, NAME, PRIMARY = "SMILES", "Molecule Name", "pEC50"
TARGETS = ["pEC50", "counter_pEC50", "Emax", "counter_Emax"]   # primary first; rest are aux heads


def _primary_pred(csv: Path) -> np.ndarray:
    """Return the PRIMARY (pEC50) prediction column from a chemprop predict output."""
    df = pd.read_csv(csv)
    if PRIMARY in df.columns:
        return pd.to_numeric(df[PRIMARY], errors="coerce").to_numpy(float)
    # fallback: first numeric non-SMILES column
    for c in df.columns:
        if c != SMILES and pd.api.types.is_numeric_dtype(df[c]):
            return df[c].to_numpy(float)
    raise RuntimeError(f"no prediction column found in {csv}")


def _train_fold(train_csv: Path, out_dir: Path, epochs: int, accel: str) -> str:
    subprocess.run(
        ["chemprop", "train", "-i", str(train_csv), "-s", SMILES,
         "--target-columns", *TARGETS, "-t", "regression", "--from-foundation", "CheMeleon",
         "--epochs", str(epochs), "--split-sizes", "0.9", "0.1", "0.0",
         "-o", str(out_dir), "--accelerator", accel, "-n", "0"],
        check=True,
    )
    ckpts = (glob.glob(str(out_dir / "**" / "best*.ckpt"), recursive=True)
             or glob.glob(str(out_dir / "**" / "*.ckpt"), recursive=True)
             or glob.glob(str(out_dir / "**" / "*.pt"), recursive=True))
    if not ckpts:
        raise RuntimeError(f"no checkpoint under {out_dir}")
    return ckpts[0]


def _predict(model_path: str, in_csv: Path, out_csv: Path, accel: str) -> np.ndarray:
    subprocess.run(
        ["chemprop", "predict", "--test-path", str(in_csv), "-s", SMILES,
         "--model-paths", model_path, "--preds-path", str(out_csv), "--accelerator", accel, "-n", "0"],
        check=True,
    )
    return _primary_pred(out_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multitask CheMeleon fine-tune, 5-fold OOF.")
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("cheme_mt_out"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train_multitask.csv").reset_index(drop=True)
    test = pd.read_csv(args.data_dir / "test.csv").reset_index(drop=True)
    folds = json.loads((args.data_dir / "folds_calibrated.json").read_text())["assignments"]
    fold_of_row = np.array([int(folds[str(i)]) for i in range(len(train))])

    if args.max_rows:
        train = train.iloc[: args.max_rows].reset_index(drop=True)
        fold_of_row = fold_of_row[: args.max_rows]

    test[[SMILES]].to_csv(out / "_test_in.csv", index=False)
    y = train[PRIMARY].to_numpy(float)
    oof = np.full(len(train), np.nan)
    test_cols = []
    fold_ids = sorted(set(fold_of_row.tolist()))
    if args.folds:
        fold_ids = fold_ids[: args.folds]

    for k in fold_ids:
        print(f"\n===== FOLD {k} (multitask) =====", flush=True)
        tr_mask = fold_of_row != k
        va_mask = fold_of_row == k
        train[tr_mask][[SMILES, *TARGETS]].to_csv(out / f"_train_f{k}.csv", index=False)
        train[va_mask][[SMILES]].to_csv(out / f"_held_f{k}.csv", index=False)

        ckpt = _train_fold(out / f"_train_f{k}.csv", out / f"ckpt_f{k}", args.epochs, args.accelerator)
        oof[va_mask] = _predict(ckpt, out / f"_held_f{k}.csv", out / f"_oof_f{k}.csv", args.accelerator)
        test_cols.append(_predict(ckpt, out / "_test_in.csv", out / f"_test_f{k}.csv", args.accelerator))
        print(f"fold {k}: {int(va_mask.sum())} OOF rows predicted (primary pEC50)", flush=True)

    done = ~np.isnan(oof)
    pd.DataFrame({"row_id": np.arange(len(train))[done], SMILES: train[SMILES][done],
                  "y_true": y[done], "y_pred": oof[done]}).to_csv(out / "oof_cheme_mt.csv", index=False)
    test_pred = np.mean(np.column_stack(test_cols), axis=1)
    pd.DataFrame({NAME: test[NAME], SMILES: test[SMILES], PRIMARY: test_pred}).to_csv(
        out / "test_cheme_mt.csv", index=False)
    print(f"\nDONE. wrote:\n  {out/'oof_cheme_mt.csv'} ({int(done.sum())} rows)\n  {out/'test_cheme_mt.csv'} ({len(test)} rows)")
    print("Send those two CSVs back for judging + stacking (multitask).")


if __name__ == "__main__":
    main()
