"""Fine-tune CheMeleon on the PXR data (run on a GPU box, e.g. A5000).

This is the heavy path the leaderboard leaders used: instead of reading CheMeleon's
frozen embeddings, we attach a fresh regression head and train the WHOLE graph
network end-to-end on pEC50 — so the encoder becomes PXR-specific.

Produces two CSVs in the format our pipeline already consumes:
  - oof_cheme_ft.csv   (row_id, SMILES, y_true, y_pred)  — leak-free 5-fold OOF on the broad rows
  - test_cheme_ft.csv  (Molecule Name, SMILES, pEC50)    — 513 test preds (mean of the 5 fold models)

Send those two files back; they get judged on Set 1 and stacked into the ensemble.
**Set 1 is never used here** — we train only on the broad rows, split by the SAME
calibrated cluster folds, so the OOF aligns with our other bases for stacking.

Needs (same folder):  train.csv, test.csv, folds_calibrated.json  +  `pip install 'chemprop>=2.2.0'`

Run on the GPU box:
    python finetune_cheme.py --epochs 50 --accelerator gpu
Quick local plumbing test (CPU, tiny):
    python finetune_cheme.py --max-rows 150 --folds 2 --epochs 1 --accelerator cpu
"""

from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SMILES, NAME, TARGET = "SMILES", "Molecule Name", "pEC50"


def _pred_column(csv: Path) -> np.ndarray:
    """Read chemprop predict output and return the prediction vector (robust to its column name)."""
    df = pd.read_csv(csv)
    skip = {SMILES, NAME, "row_id", "fold"}
    for c in df.columns:
        if c not in skip and pd.api.types.is_numeric_dtype(df[c]):
            return df[c].to_numpy(float)
    # fall back: last column coerced to numeric
    return pd.to_numeric(df[df.columns[-1]], errors="coerce").to_numpy(float)


def _train_fold(train_csv: Path, out_dir: Path, epochs: int, accel: str) -> str:
    subprocess.run(
        ["chemprop", "train", "-i", str(train_csv), "-s", SMILES,
         "--target-columns", TARGET, "-t", "regression", "--from-foundation", "CheMeleon",
         "--epochs", str(epochs), "--split-sizes", "0.9", "0.1", "0.0",
         "-o", str(out_dir), "--accelerator", accel, "-n", "0"],
        check=True,
    )
    ckpts = (glob.glob(str(out_dir / "**" / "best*.ckpt"), recursive=True)
             or glob.glob(str(out_dir / "**" / "*.ckpt"), recursive=True)
             or glob.glob(str(out_dir / "**" / "*.pt"), recursive=True))
    if not ckpts:
        raise RuntimeError(f"no checkpoint found under {out_dir}")
    return ckpts[0]


def _predict(model_path: str, in_csv: Path, out_csv: Path, accel: str) -> np.ndarray:
    subprocess.run(
        ["chemprop", "predict", "--test-path", str(in_csv), "-s", SMILES,
         "--model-paths", model_path, "--preds-path", str(out_csv), "--accelerator", accel, "-n", "0"],
        check=True,
    )
    return _pred_column(out_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune CheMeleon on PXR, 5-fold OOF.")
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("cheme_ft_out"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--accelerator", default="auto", help="auto|gpu|cpu|mps")
    ap.add_argument("--folds", type=int, default=None, help="limit #folds (debug)")
    ap.add_argument("--max-rows", type=int, default=None, help="subsample broad rows (debug)")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv").reset_index(drop=True)
    test = pd.read_csv(args.data_dir / "test.csv").reset_index(drop=True)
    folds = __import__("json").loads((args.data_dir / "folds_calibrated.json").read_text())
    fold_of_row = np.array([int(folds["assignments"][str(i)]) for i in range(len(train))])

    if args.max_rows:  # debug subsample (keeps fold labels)
        train = train.iloc[: args.max_rows].reset_index(drop=True)
        fold_of_row = fold_of_row[: args.max_rows]

    test[[SMILES]].to_csv(out / "_test_in.csv", index=False)
    y = train[TARGET].to_numpy(float)
    oof = np.full(len(train), np.nan)
    test_cols = []
    fold_ids = sorted(set(fold_of_row.tolist()))
    if args.folds:
        fold_ids = fold_ids[: args.folds]

    for k in fold_ids:
        print(f"\n===== FOLD {k} =====", flush=True)
        tr_mask = fold_of_row != k
        va_mask = fold_of_row == k
        train[tr_mask][[SMILES, TARGET]].to_csv(out / f"_train_f{k}.csv", index=False)
        train[va_mask][[SMILES]].to_csv(out / f"_held_f{k}.csv", index=False)

        ckpt = _train_fold(out / f"_train_f{k}.csv", out / f"ckpt_f{k}", args.epochs, args.accelerator)
        oof[va_mask] = _predict(ckpt, out / f"_held_f{k}.csv", out / f"_oof_f{k}.csv", args.accelerator)
        test_cols.append(_predict(ckpt, out / "_test_in.csv", out / f"_test_f{k}.csv", args.accelerator))
        print(f"fold {k}: {int(va_mask.sum())} OOF rows predicted", flush=True)

    done = ~np.isnan(oof)
    pd.DataFrame({"row_id": np.arange(len(train))[done], SMILES: train[SMILES][done],
                  "y_true": y[done], "y_pred": oof[done]}).to_csv(out / "oof_cheme_ft.csv", index=False)
    test_pred = np.mean(np.column_stack(test_cols), axis=1)
    pd.DataFrame({NAME: test[NAME], SMILES: test[SMILES], TARGET: test_pred}).to_csv(
        out / "test_cheme_ft.csv", index=False)
    print(f"\nDONE. wrote:\n  {out/'oof_cheme_ft.csv'} ({int(done.sum())} rows)\n  {out/'test_cheme_ft.csv'} ({len(test)} rows)")
    print("Send those two CSVs back for judging + stacking.")


if __name__ == "__main__":
    main()
