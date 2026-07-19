"""#1 Approach 2: the proxy-label weak-end specialist (faithful multitask version).

#1: "trained a modified version of this GNN architecture on a 'proxy-svr-labels' task ... (with LogD
as an auxiliary task) to predict pEC50. ... blended with the first approach in cases where the
proxy-trained model predicted pEC50 less than 4.5."

So this is a multitask CheMeleon trained on the SVR-imputed proxy pEC50 (primary) + LogD (aux). No
5-fold (the proxy molecules are not an eval set) and NO calibration — #1 uses neither here. Output is
test_approach2.csv, consumed by meta_blend.py (mixed into Approach 1 where its prediction < 4.5).

Data: data/pxr_activity/train_approach2.csv (11,005 rows: 6,805 proxy pEC50 + 4,200 public LogD).

Run on the GPU pod:
    nohup python reproduce_1/approach2_experiment.py > approach2.log 2>&1 &
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.metrics import score_all  # noqa: E402

SMILES, NAME, PRIMARY = "SMILES", "Molecule Name", "pEC50"
TARGETS = ["pEC50", "logD"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/pxr_activity")
    ap.add_argument("--pool", type=Path, default=REPO / "data/pxr_activity/train_approach2.csv")
    ap.add_argument("--out", type=Path, default=Path("/tmp/approach2"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--accelerator", default="gpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(args.data_dir / "test.csv").reset_index(drop=True)
    test[[SMILES]].to_csv(args.out / "_test_in.csv", index=False)

    subprocess.run(
        ["chemprop", "train", "-i", str(args.pool), "-s", SMILES, "--target-columns", *TARGETS,
         "-t", "regression", "--from-foundation", "CheMeleon", "--loss-function", "mae",
         "--epochs", str(args.epochs), "--warmup-epochs", str(max(1, min(2, args.epochs - 1))),
         "--split-sizes", "0.9", "0.1", "0.0", "--batch-size", str(args.batch_size),
         "-o", str(args.out / "ckpt"), "--accelerator", args.accelerator, "-n", str(args.num_workers)],
        check=True,
    )
    ckpts = (glob.glob(str(args.out / "ckpt" / "**" / "best*.ckpt"), recursive=True)
             or glob.glob(str(args.out / "ckpt" / "**" / "*.ckpt"), recursive=True))
    if not ckpts:
        raise SystemExit("no checkpoint")
    subprocess.run(
        ["chemprop", "predict", "--test-path", str(args.out / "_test_in.csv"), "-s", SMILES,
         "--model-paths", ckpts[0], "--preds-path", str(args.out / "_test_pred.csv"),
         "--accelerator", args.accelerator, "-n", str(args.num_workers)],
        check=True,
    )
    pred_df = pd.read_csv(args.out / "_test_pred.csv")
    spec = pd.to_numeric(pred_df[PRIMARY], errors="coerce").to_numpy(float) if PRIMARY in pred_df.columns \
        else pred_df.select_dtypes("number").iloc[:, 0].to_numpy(float)
    out_csv = REPO / "predictions" / "test_approach2.csv"
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame({NAME: test[NAME], SMILES: test[SMILES], PRIMARY: spec}).to_csv(out_csv, index=False)

    # report the specialist alone on Set-2 (expected poor overall — it is a weak-end model), no calibration
    p2 = pd.read_csv(args.data_dir / "phase2_unblinded.csv")[[NAME, PRIMARY]].dropna(subset=[PRIMARY])
    m = p2.merge(pd.DataFrame({NAME: test[NAME], "yhat": spec}), on=NAME)
    s = score_all(m[PRIMARY].to_numpy(float), m["yhat"].to_numpy(float))
    print(f"\n=== Approach-2 (proxy specialist alone) Set-2 ===")
    print(f"    RAE={s['rae']:.4f}  MAE={s['mae']:.4f}  (n={len(m)})  "
          f"predicts <4.5 on {(spec < 4.5).sum()}/{len(spec)} test molecules")
    print(f"    saved {out_csv}  -> used by meta_blend.py (mixed into Approach 1 where <4.5)")


if __name__ == "__main__":
    main()
