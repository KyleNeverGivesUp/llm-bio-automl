"""#1's meta-blend: regime-switched combination of the approaches. FAITHFUL, no calibration,
NO Set-1 weight tuning (#1's report only says "blended" and gives no weight — so we use a plain
equal-weight average in each trigger region; the thresholds 4.5 are #1's).

#1: Approach 1 is the base; Approach 2 (proxy weak specialist) is blended in where IT predicts
pEC50 < 4.5; Approach 3 (model zoo, strong-end) is blended in where IT predicts pEC50 > 4.5.

  final = A1                                           (base)
  where A2 < 4.5:  final = mean(A1, A2)                (weak-end, equal weight)
  where A3 > 4.5:  final = mean(final, A3)             (strong-end, equal weight)

We add each approach ONE AT A TIME so its effect on Set-2 is visible. Set-2 scored, no tuning on it.

Inputs (raw predictions in predictions/):
  A1 = test_approach1.csv     A2 = test_approach2.csv
  A3 = test_ensemble.csv  (CheMeleon+Uni-Mol nnls — a STAND-IN for #1's real zoo until we build it)

Run (pod, reads local predictions, no GPU):  python reproduce_1/meta_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.metrics import score_all  # noqa: E402

NAME, TGT = "Molecule Name", "pEC50"
DATA = REPO / "data/pxr_activity"
THRESH = 4.5   # #1


def load(path, names):
    d = pd.read_csv(REPO / "predictions" / path)
    return pd.DataFrame({NAME: d[NAME], "p": d[TGT].to_numpy(float)}).set_index(NAME).reindex(names)["p"].to_numpy(float)


def score(names, pred, labels, tag):
    m = labels.merge(pd.DataFrame({NAME: names, "yhat": pred}), on=NAME, how="left")
    if m["yhat"].isna().any():
        raise SystemExit(f"{tag}: missing predictions")
    s = score_all(m[TGT].to_numpy(float), m["yhat"].to_numpy(float))
    print(f"    {tag:44s} RAE={s['rae']:.4f}  MAE={s['mae']:.4f}")
    return s["rae"]


def main() -> None:
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    names = test[NAME].to_numpy()
    a1 = load("test_approach1.csv", names)
    a2 = load("test_approach2.csv", names)
    a3 = load("test_ensemble.csv", names)   # stand-in zoo
    p2 = pd.read_csv(DATA / "phase2_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])

    weak, strong = a2 < THRESH, a3 > THRESH
    print(f"[gates] A2<{THRESH}: {weak.sum()}/{len(a2)}   A3>{THRESH}: {strong.sum()}/{len(a3)}")

    a1_a2 = a1.copy(); a1_a2[weak] = 0.5 * a1[weak] + 0.5 * a2[weak]          # + Approach 2
    a1_a3 = a1.copy(); a1_a3[strong] = 0.5 * a1[strong] + 0.5 * a3[strong]   # + Approach 3 (stand-in)
    full = a1_a2.copy(); full[strong] = 0.5 * full[strong] + 0.5 * a3[strong]

    print("\n=== Set-2 (260 blind) — equal-weight blend, NO calibration, NO Set-1 tuning ===")
    score(names, a1, p2, "Approach 1 (base)")
    score(names, a1_a2, p2, "  + Approach 2 blended (<4.5)")
    score(names, a1_a3, p2, "  + Approach 3 blended (>4.5) [zoo=stand-in]")
    score(names, full, p2, "  + both (full meta-blend)")
    print("\n  reference: #1 (matcha) 0.5631 | #2 (best public) 0.5676 | rank-12 0.586")
    print("  note: A3 is a STAND-IN (CheMeleon+Uni-Mol); #1's real zoo not built yet.")


if __name__ == "__main__":
    main()
