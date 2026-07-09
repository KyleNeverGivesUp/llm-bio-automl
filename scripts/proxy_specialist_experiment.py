"""#1's lever: a weak-end specialist trained on proxy pEC50 labels, blended in conditionally.

Diagnosis that motivates this (measured on the now-public Set-2, ensemble model):
    pEC50 bin   n     bias      MAE
    <4.0        45   +0.865    0.914   <- 17% of the test carries ~36% of ALL error
    5.0-6.0    122   -0.27     ~0.29
    >6.0        21   -0.965    0.965
  Variance-match calibration (fit on OOF only) fixes the STRONG end (0.6311 -> 0.5923) but barely
  touches the weak end (+0.865 -> +0.810): a linear rescale cannot move molecules the model failed
  to *recognise* as weak — their predictions sit near the mean.

Root cause: the 4,139 dose-response molecules are 94% active (they were promoted to the expensive
curve because the cheap screen said so). The model has almost never seen a weak molecule with a
pEC50 label. `scripts/build_proxy_labels.py` recovers 6,805 screen-rejected molecules (83% weak) by
imputing pEC50 from their single-concentration readouts with an SVR.

This script (matcha-croissant / #1):
  1. fine-tune CheMeleon on those 6,805 proxy labels  -> a WEAK-END SPECIALIST
  2. predict the 513 test molecules with it
  3. calibrate the main ensemble (variance-match, fitted on OOF only)
  4. blend: where the specialist predicts < 4.5, mix it into the main prediction; else leave alone

Honesty protocol (we already learned Set-1 is an over-fittable judge):
  - the 4.5 threshold is PRE-REGISTERED from #1's report, not tuned by us;
  - the blend weight is chosen on Set-1 (the dev set), and Set-2 is scored ONCE at the end.

Run on the GPU pod (from repo root):
    nohup python scripts/proxy_specialist_experiment.py > proxy.log 2>&1 &
    python scripts/proxy_specialist_experiment.py --skip-train    # re-score existing preds
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

SMILES, NAME, TGT = "SMILES", "Molecule Name", "pEC50"


# --------------------------------------------------------------------------- #
def train_specialist(proxy_csv: Path, out_dir: Path, epochs: int, batch: int, accel: str) -> str:
    """Fine-tune CheMeleon on the proxy labels. Single target — this model only needs to be
    good at recognising weak molecules; the main ensemble owns the rest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["chemprop", "train", "-i", str(proxy_csv), "-s", SMILES, "--target-columns", TGT,
         "-t", "regression", "--from-foundation", "CheMeleon", "--loss-function", "mae",
         "--epochs", str(epochs), "--warmup-epochs", str(max(1, min(2, epochs - 1))),
         "--split-sizes", "0.9", "0.1", "0.0", "--batch-size", str(batch),
         "-o", str(out_dir), "--accelerator", accel, "-n", "8"],
        check=True,
    )
    ckpts = (glob.glob(str(out_dir / "**" / "best*.ckpt"), recursive=True)
             or glob.glob(str(out_dir / "**" / "*.ckpt"), recursive=True))
    if not ckpts:
        raise SystemExit(f"no checkpoint under {out_dir}")
    return ckpts[0]


def predict(ckpt: str, in_csv: Path, out_csv: Path, accel: str) -> np.ndarray:
    subprocess.run(
        ["chemprop", "predict", "--test-path", str(in_csv), "-s", SMILES,
         "--model-paths", ckpt, "--preds-path", str(out_csv), "--accelerator", accel, "-n", "8"],
        check=True,
    )
    df = pd.read_csv(out_csv)
    if TGT in df.columns:
        return pd.to_numeric(df[TGT], errors="coerce").to_numpy(float)
    for c in df.columns:
        if c != SMILES and pd.api.types.is_numeric_dtype(df[c]):
            return df[c].to_numpy(float)
    raise SystemExit(f"no prediction column in {out_csv}")


def variance_match(oof_csv: Path) -> tuple[float, float]:
    """Fit `new = a*old + b` so the prediction spread matches the label spread. OOF only —
    the test answers are never touched. (The affine least-squares fit gives a~1.0 because
    shrinkage is MSE-optimal; RAE is an L1 metric, so we match the distribution instead.)"""
    d = pd.read_csv(oof_csv)
    y = d["y_true"].to_numpy(float)
    p = d["y_pred"].to_numpy(float)
    a = y.std() / p.std()
    return a, y.mean() - a * p.mean()


def _score(names: pd.Series, pred: np.ndarray, labels: pd.DataFrame, tag: str) -> float:
    m = labels.merge(pd.DataFrame({NAME: names, "yhat": pred}), on=NAME, how="left")
    if m["yhat"].isna().any():
        raise SystemExit(f"{tag}: missing predictions")
    s = score_all(m[TGT].to_numpy(float), m["yhat"].to_numpy(float))
    print(f"    {tag:44s} RAE={s['rae']:.4f}  MAE={s['mae']:.4f}  R2={s['r2']:.4f}  (n={len(m)})")
    return s["rae"]


def bins(names: pd.Series, pred: np.ndarray, labels: pd.DataFrame, tag: str) -> None:
    m = labels.merge(pd.DataFrame({NAME: names, "yhat": pred}), on=NAME, how="left")
    m["r"] = m["yhat"] - m[TGT]
    m["b"] = pd.cut(m[TGT], [0, 4, 4.5, 5, 5.5, 6, 99],
                    labels=["<4", "4-4.5", "4.5-5", "5-5.5", "5.5-6", ">6"], right=False)
    print(f"\n  [{tag}] per-bin bias / MAE on Set-2")
    for lb in ["<4", "4-4.5", "4.5-5", "5-5.5", "5.5-6", ">6"]:
        g = m[m["b"] == lb]
        if len(g):
            print(f"    {lb:7s} n={len(g):3d}  bias={g['r'].mean():+.3f}  MAE={g['r'].abs().mean():.3f}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/pxr_activity")
    ap.add_argument("--proxy", type=Path, default=REPO / "data/pxr_activity/proxy_train.csv")
    ap.add_argument("--main-test", type=Path, default=REPO / "predictions/test_ensemble.csv")
    ap.add_argument("--main-oof", type=Path, default=REPO / "predictions/oof_ensemble.csv")
    ap.add_argument("--out", type=Path, default=Path("/tmp/proxy_specialist"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--threshold", type=float, default=4.5, help="pre-registered from #1's report")
    ap.add_argument("--include-real", action="store_true",
                    help="also train on the 4,139 REAL labels. Default (off) reproduces #1: a pure "
                         "proxy specialist. On: the model sees both regimes, so it stays calibrated "
                         "across the range while still gaining ~5.7k weak examples. Worth running both.")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    test = pd.read_csv(args.data_dir / "test.csv").reset_index(drop=True)
    p1 = pd.read_csv(args.data_dir / "phase1_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])
    p2 = pd.read_csv(args.data_dir / "phase2_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])

    # 1-2. the weak-end specialist
    args.out.mkdir(parents=True, exist_ok=True)
    spec_csv = args.out / "test_specialist.csv"
    if not args.skip_train:
        train_csv = args.proxy
        if args.include_real:                       # proxy (mostly weak) + the real dose-response labels
            real = pd.read_csv(args.data_dir / "train.csv")[[SMILES, TGT]]
            pool = pd.concat([pd.read_csv(args.proxy)[[SMILES, TGT]], real], ignore_index=True)
            train_csv = args.out / "_train_pool.csv"
            pool.to_csv(train_csv, index=False)
            print(f"[data] proxy {len(pool) - len(real)} + real {len(real)} = {len(pool)} training molecules")
        test[[SMILES]].to_csv(args.out / "_test_in.csv", index=False)
        ckpt = train_specialist(train_csv, args.out / "ckpt", args.epochs, args.batch_size, args.accelerator)
        spec = predict(ckpt, args.out / "_test_in.csv", spec_csv, args.accelerator)
        pd.DataFrame({NAME: test[NAME], SMILES: test[SMILES], TGT: spec}).to_csv(spec_csv, index=False)
    spec = pd.read_csv(spec_csv)[TGT].to_numpy(float)

    # 3. the main ensemble, variance-matched on its own OOF
    main_df = pd.read_csv(args.main_test)
    assert (main_df[NAME].to_numpy() == test[NAME].to_numpy()).all(), "test row order mismatch"
    raw = main_df[TGT].to_numpy(float)
    a, b = variance_match(args.main_oof)
    cal = a * raw + b
    print(f"\n[calibration] variance-match from OOF: new = {a:.3f}*old + ({b:.3f})")

    fires = spec < args.threshold
    print(f"[blend] specialist predicts <{args.threshold} on {fires.sum()}/{len(spec)} test molecules")

    # 4. choose the blend weight on Set-1 (the dev set). Set-2 is scored ONCE, after.
    print(f"\n=== choosing blend weight on Set-1 (dev) ===")
    best_w, best_r = 0.0, float("inf")
    for w in np.arange(0.0, 1.01, 0.1):
        blended = np.where(fires, w * spec + (1 - w) * cal, cal)
        r = _score(test[NAME], blended, p1, f"w={w:.1f}")
        if r < best_r:
            best_w, best_r = float(w), r
    print(f"\n  -> selected w={best_w:.1f} (Set-1 RAE {best_r:.4f})")

    blended = np.where(fires, best_w * spec + (1 - best_w) * cal, cal)

    print(f"\n=== Set-2 (260 blind, final scorer) — scored once ===")
    _score(test[NAME], raw, p2, "1. main ensemble (baseline)")
    _score(test[NAME], cal, p2, "2. + variance-match calibration")
    _score(test[NAME], spec, p2, "   (specialist alone — expected poor overall)")
    _score(test[NAME], blended, p2, f"3. + proxy weak-specialist blend (w={best_w:.1f})")
    print("\n  reference: #2 (best public-data team) 0.5676 | rank-12 0.586 | #1 (proprietary) 0.5631")

    bins(test[NAME], cal, p2, "before blend")
    bins(test[NAME], blended, p2, "after blend")


if __name__ == "__main__":
    main()
