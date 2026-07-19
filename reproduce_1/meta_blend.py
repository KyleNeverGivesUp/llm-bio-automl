"""#1's meta-blend: regime-switched combination of the three approaches. NO calibration.

#1: Approach 1 is the base; Approach 2 (proxy weak specialist) is blended in where IT predicts
pEC50 < 4.5; Approach 3 (model zoo, strong-end) is blended in where IT predicts pEC50 > 4.5.

  final = A1                                            (base, everywhere)
  where A2 < 4.5:  final = w2*A2 + (1-w2)*A1            (weak-end correction)
  where A3 > 4.5:  final = w3*A3 + (1-w3)*A1            (strong-end correction)

Thresholds (4.5) are pre-registered from #1's report. The blend weights w2, w3 are chosen on Set-1
(dev); Set-2 is scored ONCE. No variance-match / calibration anywhere — #1 does not use it.

Inputs (all raw predictions in predictions/):
  A1 = test_approach1.csv (multitask base)   A2 = test_approach2.csv (proxy specialist)
  A3 = test_ensemble.csv  (CheMeleon + Uni-Mol nnls — a stand-in for #1's zoo, until we build a
       fuller zoo with Unimol2 / TabICL / SVR + LASSO)

Run (local or pod, no GPU):  uv run python reproduce_1/meta_blend.py
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
THRESH = 4.5   # pre-registered from #1


def load(path, test_names):
    d = pd.read_csv(REPO / "predictions" / path)
    d = pd.DataFrame({NAME: d[NAME], "p": d[TGT].to_numpy(float)}).set_index(NAME).reindex(test_names)
    return d["p"].to_numpy(float)


def score(names, pred, labels, tag, quiet=False):
    m = labels.merge(pd.DataFrame({NAME: names, "yhat": pred}), on=NAME, how="left")
    if m["yhat"].isna().any():
        raise SystemExit(f"{tag}: missing predictions")
    s = score_all(m[TGT].to_numpy(float), m["yhat"].to_numpy(float))
    if not quiet:
        print(f"    {tag:44s} RAE={s['rae']:.4f}  MAE={s['mae']:.4f}  R2={s['r2']:.4f}")
    return s["rae"]


def main() -> None:
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    names = test[NAME].to_numpy()
    a1 = load("test_approach1.csv", names)
    a2 = load("test_approach2.csv", names)
    a3 = load("test_ensemble.csv", names)
    p1 = pd.read_csv(DATA / "phase1_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])
    p2 = pd.read_csv(DATA / "phase2_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])

    weak = a2 < THRESH        # Approach 2 fires here
    strong = a3 > THRESH      # Approach 3 fires here
    print(f"[gates] A2<{THRESH}: {weak.sum()}/{len(a2)}   A3>{THRESH}: {strong.sum()}/{len(a3)}")

    def build(w2, w3):
        out = a1.copy()
        out[weak] = w2 * a2[weak] + (1 - w2) * a1[weak]
        out[strong] = w3 * a3[strong] + (1 - w3) * out[strong]   # applied on the (weak-updated) base
        return out

    # choose w2, w3 on Set-1 (dev)
    best, best_r = (0.0, 0.0), float("inf")
    for w2 in np.arange(0, 1.01, 0.1):
        for w3 in np.arange(0, 1.01, 0.1):
            r = score(names, build(w2, w3), p1, "", quiet=True)
            if r < best_r:
                best, best_r = (round(float(w2), 1), round(float(w3), 1)), r
    w2, w3 = best
    print(f"[dev] selected on Set-1: w2(weak)={w2}, w3(strong)={w3}  (Set-1 RAE {best_r:.4f})")

    print("\n=== Set-2 (260 blind, final scorer) — scored once, NO calibration ===")
    score(names, a1, p2, "Approach 1 (base) alone")
    score(names, a2, p2, "Approach 2 (proxy) alone")
    score(names, a3, p2, "Approach 3 (zoo) alone")
    score(names, build(w2, w3), p2, f"META-BLEND  (w2={w2}, w3={w3})")
    print("\n  reference: #1 (matcha) 0.5631 | #2 (best public) 0.5676 | rank-12 0.586")


if __name__ == "__main__":
    main()
