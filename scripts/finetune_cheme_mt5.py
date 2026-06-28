"""MULTITASK-5 fine-tune of CheMeleon (run on GPU, e.g. A5000).

Extends the 4-target multitask run with a 5th head: the single-concentration screen
(`sc_log2fc`, same PXR assay), which brings 8,135 NEW molecules. Those extra rows
are ALWAYS in training (never held out), so the OOF stays on the broad rows and the
graph encoder sees ~3x more PXR chemistry. We score / stack only the PRIMARY pEC50.

Targets: pEC50 (primary) + counter_pEC50 + Emax + counter_Emax + sc_log2fc.
Set 1 never touched (leakage-checked out of sc_extra).

Needs (same folder): train_multitask5.csv, sc_extra.csv, test.csv, folds_calibrated.json
  + `pip install 'chemprop>=2.2.0'`
Run:  python finetune_cheme_mt5.py --epochs 50 --accelerator gpu
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
TARGETS = ["pEC50", "counter_pEC50", "Emax", "counter_Emax", "sc_log2fc"]


def _primary_pred(csv: Path) -> np.ndarray:
    df = pd.read_csv(csv)
    if PRIMARY in df.columns:
        return pd.to_numeric(df[PRIMARY], errors="coerce").to_numpy(float)
    for c in df.columns:
        if c != SMILES and pd.api.types.is_numeric_dtype(df[c]):
            return df[c].to_numpy(float)
    raise RuntimeError(f"no prediction column in {csv}")


def _train_fold(train_csv: Path, out_dir: Path, epochs: int, accel: str,
                num_workers: int = 8, batch_size: int = 64) -> str:
    subprocess.run(
        ["chemprop", "train", "-i", str(train_csv), "-s", SMILES,
         "--target-columns", *TARGETS, "-t", "regression", "--from-foundation", "CheMeleon",
         "--loss-function", "mae",   # the 0.538 solution used MAE loss (~0.01 over MSE on noisy labels)
         "--epochs", str(epochs), "--warmup-epochs", str(max(1, min(2, epochs - 1))),  # warmup must be < epochs
         "--split-sizes", "0.9", "0.1", "0.0",
         "--batch-size", str(batch_size),
         "-o", str(out_dir), "--accelerator", accel, "-n", str(num_workers)],
        check=True,
    )
    ckpts = (glob.glob(str(out_dir / "**" / "best*.ckpt"), recursive=True)
             or glob.glob(str(out_dir / "**" / "*.ckpt"), recursive=True)
             or glob.glob(str(out_dir / "**" / "*.pt"), recursive=True))
    if not ckpts:
        raise RuntimeError(f"no checkpoint under {out_dir}")
    return ckpts[0]


def _predict(model_path: str, in_csv: Path, out_csv: Path, accel: str, num_workers: int = 8) -> np.ndarray:
    subprocess.run(
        ["chemprop", "predict", "--test-path", str(in_csv), "-s", SMILES,
         "--model-paths", model_path, "--preds-path", str(out_csv),
         "--accelerator", accel, "-n", str(num_workers)],
        check=True,
    )
    return _primary_pred(out_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multitask-5 CheMeleon fine-tune (+single-conc head), 5-fold OOF.")
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("cheme_mt5_out"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8, help="dataloader workers (8 ~= 12x faster than 0)")
    ap.add_argument("--batch-size", type=int, default=128, help="train batch size (128 ~2x faster than 64; use 64 to reproduce 0.5904 exactly)")
    ap.add_argument("--keep-reactive", action="store_true",
                    help="disable reactive-electrophile exclusion (default: drop reactive rows from TRAINING)")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train_multitask5.csv").reset_index(drop=True)
    extra_df = pd.read_csv(args.data_dir / "sc_extra.csv")
    # reactive-electrophile exclusion (0.538-solution lever): drop reactive rows from
    # TRAINING only; held-out broad rows keep their OOF (test has 0 reactive anyway).
    if not args.keep_reactive:
        extra_df = extra_df[extra_df.get("is_reactive", 0) == 0]
    extra = extra_df[[SMILES, *TARGETS]]   # always-train single-conc rows
    train_reactive = train.get("is_reactive", pd.Series(0, index=train.index)).to_numpy(int).astype(bool)
    test = pd.read_csv(args.data_dir / "test.csv").reset_index(drop=True)
    folds = json.loads((args.data_dir / "folds_calibrated.json").read_text())["assignments"]
    fold_of_row = np.array([int(folds[str(i)]) for i in range(len(train))])

    if args.max_rows:
        train = train.iloc[: args.max_rows].reset_index(drop=True)
        fold_of_row = fold_of_row[: args.max_rows]
        train_reactive = train_reactive[: args.max_rows]
        extra = extra.iloc[:500]

    test[[SMILES]].to_csv(out / "_test_in.csv", index=False)
    y = train[PRIMARY].to_numpy(float)
    oof = np.full(len(train), np.nan)
    test_cols = []
    fold_ids = sorted(set(fold_of_row.tolist()))
    if args.folds:
        fold_ids = fold_ids[: args.folds]

    for k in fold_ids:
        print(f"\n===== FOLD {k} (multitask-5, +{len(extra)} single-conc rows always-train) =====", flush=True)
        # train on broad[fold!=k] with reactive electrophiles excluded; predict OOF on ALL of fold==k
        tr_mask = (fold_of_row != k) & (~train_reactive if not args.keep_reactive else True)
        va_mask = fold_of_row == k
        # fold train = curated broad[fold!=k] (5 targets) + curated single-conc extra rows (only sc_log2fc)
        fold_train = pd.concat([train[tr_mask][[SMILES, *TARGETS]], extra], ignore_index=True)
        fold_train.to_csv(out / f"_train_f{k}.csv", index=False)
        train[va_mask][[SMILES]].to_csv(out / f"_held_f{k}.csv", index=False)

        ckpt = _train_fold(out / f"_train_f{k}.csv", out / f"ckpt_f{k}", args.epochs, args.accelerator,
                           num_workers=args.num_workers, batch_size=args.batch_size)
        oof[va_mask] = _predict(ckpt, out / f"_held_f{k}.csv", out / f"_oof_f{k}.csv", args.accelerator, args.num_workers)
        test_cols.append(_predict(ckpt, out / "_test_in.csv", out / f"_test_f{k}.csv", args.accelerator, args.num_workers))
        print(f"fold {k}: {int(va_mask.sum())} OOF rows predicted (primary pEC50)", flush=True)

    done = ~np.isnan(oof)
    pd.DataFrame({"row_id": np.arange(len(train))[done], SMILES: train[SMILES][done],
                  "y_true": y[done], "y_pred": oof[done]}).to_csv(out / "oof_cheme_mt5.csv", index=False)
    test_pred = np.mean(np.column_stack(test_cols), axis=1)
    pd.DataFrame({NAME: test[NAME], SMILES: test[SMILES], PRIMARY: test_pred}).to_csv(
        out / "test_cheme_mt5.csv", index=False)
    print(f"\nDONE. wrote:\n  {out/'oof_cheme_mt5.csv'} ({int(done.sum())} rows)\n  {out/'test_cheme_mt5.csv'} ({len(test)} rows)")


if __name__ == "__main__":
    main()
