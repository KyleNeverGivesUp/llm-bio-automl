"""Fold-in experiment: does adding Set-1 into TRAINING lower RAE on the blind Set-2?

Context (post-competition, both label sets now public):
  - The final private board scores **Set-2 only** (260 blind compounds). Folding Set-1's true
    labels into the *submission* is worthless there. But Set-1 and Set-2 are close analogs of the
    SAME 63 hits, so folding Set-1 into *training* gives near-in-distribution signal that may
    genuinely improve the Set-2 *predictions*.
  - Our broad-only CheMeleon scored Set-2 RAE 0.6301; leaderboard rank-12 = 0.586, rank-1 = 0.5631.

Design — one variable changed vs the validated 0.5904/0.6301 baseline:
  fold Set-1's pEC50 into the always-train `sc_extra` rows (mask the 4 aux heads as NaN,
  is_reactive=0), then rerun the VALIDATED mt5 template UNCHANGED (same 5 folds, batch 64,
  50 epochs). Set-2 is used ONLY for scoring + a hard leak guard, never trained on.

Run on the GPU pod (from repo root):
    python scripts/foldin_set1_experiment.py                # build + train + score
    nohup python scripts/foldin_set1_experiment.py > foldin.log 2>&1 &   # detached (recommended)
    python scripts/foldin_set1_experiment.py --build-only   # just build the data + print the cmd
    python scripts/foldin_set1_experiment.py --skip-train   # re-score an existing --out
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.metrics import score_all  # noqa: E402

RDLogger.DisableLog("rdApp.*")
SMILES, NAME, TGT = "SMILES", "Molecule Name", "pEC50"
TARGETS = ["pEC50", "counter_pEC50", "Emax", "counter_Emax", "sc_log2fc"]


def _canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def build_foldin(data_dir: Path, work: Path) -> None:
    """sc_extra + Set-1(pEC50 only) -> work/sc_extra.csv; copy the 3 unchanged files.

    Two hard leak guards (abort on either):
      - Set-1 must NOT already be in the broad training rows (else fold-in is a no-op).
      - Set-2 must NOT appear anywhere in the training pool (broad + sc_extra + folded Set-1).
    """
    sc = pd.read_csv(data_dir / "sc_extra.csv")
    p1 = pd.read_csv(data_dir / "phase1_unblinded.csv")          # Set-1 -> fold into training
    p2 = pd.read_csv(data_dir / "phase2_unblinded.csv")          # Set-2 -> leak guard + scoring only
    broad = pd.read_csv(data_dir / "train_multitask5.csv")

    broad_c = {c for c in (_canon(s) for s in broad[SMILES]) if c}
    sc_c = {c for c in (_canon(s) for s in sc[SMILES]) if c}
    p1_c = [_canon(s) for s in p1[SMILES]]
    p2_c = {c for c in (_canon(s) for s in p2[SMILES]) if c}

    if sum(c in broad_c for c in p1_c if c):
        raise SystemExit("LEAK: some Set-1 molecules are already in the broad train — abort")
    train_pool = broad_c | sc_c | {c for c in p1_c if c}
    n_s2 = len(p2_c & train_pool)
    if n_s2:
        raise SystemExit(f"LEAK: {n_s2} Set-2 molecules found in the training pool — abort")

    s1 = pd.DataFrame({SMILES: [c for c in p1_c], "pEC50": p1[TGT].astype(float).to_numpy(),
                       "counter_pEC50": np.nan, "Emax": np.nan, "counter_Emax": np.nan,
                       "sc_log2fc": np.nan, "is_reactive": 0})
    foldin = pd.concat([sc, s1], ignore_index=True)
    work.mkdir(parents=True, exist_ok=True)
    foldin.to_csv(work / "sc_extra.csv", index=False)
    for f in ("train_multitask5.csv", "test.csv", "folds_calibrated.json"):
        shutil.copy(data_dir / f, work / f)
    print(f"[build] sc_extra {len(sc)} + Set-1 {len(s1)} = {len(foldin)} always-train rows "
          f"-> {work / 'sc_extra.csv'}")
    print("[build] leak guards OK (Set-1 not in broad; Set-2 not in any training data)")


def score_set2(pred_csv: Path, data_dir: Path, tag: str) -> float:
    """Score a 513-row test prediction on the 260 blind Set-2 rows (final-rank set)."""
    p2 = pd.read_csv(data_dir / "phase2_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])
    pred = pd.read_csv(pred_csv)[[NAME, TGT]].rename(columns={TGT: "yhat"})
    m = p2.merge(pred, on=NAME, how="left")
    miss = int(m["yhat"].isna().sum())
    if miss:
        raise SystemExit(f"{tag}: {miss}/{len(p2)} Set-2 rows have no prediction")
    s = score_all(m[TGT].to_numpy(float), m["yhat"].to_numpy(float))
    print(f"  {tag:34s} Set-2 RAE={s['rae']:.4f}  MAE={s['mae']:.4f}  R2={s['r2']:.4f}  (n={len(m)})")
    return s["rae"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/pxr_activity")
    ap.add_argument("--work", type=Path, default=Path("/tmp/foldin_set1"), help="fold-in data dir (pod-local)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/cheme_foldin"), help="mt5 output dir (pod-local)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64, help="64 reproduces the 0.5904 baseline exactly")
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--build-only", action="store_true", help="build the fold-in data, print the train cmd, stop")
    ap.add_argument("--skip-train", action="store_true", help="skip training; just re-score --out")
    args = ap.parse_args()

    build_foldin(args.data_dir, args.work)
    cmd = ["python", str(REPO / "scripts/finetune_cheme_mt5.py"),
           "--data-dir", str(args.work), "--out-dir", str(args.out),
           "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
           "--accelerator", args.accelerator]
    if args.build_only:
        print("[build-only] now run:\n  " + " ".join(cmd))
        return
    if not args.skip_train:
        print("[train] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print("\n=== Set-2 (260 blind, final-rank) ===")
    base_pred = REPO / "predictions/test_cheme_mt5.csv"
    if base_pred.exists():
        score_set2(base_pred, args.data_dir, "baseline  broad-only")
    score_set2(args.out / "test_cheme_mt5.csv", args.data_dir, "FOLD-IN   broad+Set1")
    print("  reference: leaderboard rank-12 (discoverybytes) 0.586, rank-1 0.5631")


if __name__ == "__main__":
    main()
