"""Build proxy pEC50 labels from the cheap single-concentration screen (the #1 solution's lever).

Why: the 4,139 dose-response molecules are 94% ACTIVE — they were promoted to the expensive full
curve precisely because the cheap screen said they looked active. So our model almost never sees a
WEAK molecule with a pEC50 label, and cannot recognise one (Set-2 bias at pEC50<4 is +0.81, and
that bin alone carries ~36% of the remaining error). The ~6.8k molecules the screen REJECTED are
mostly weak, and they are sitting unused in `single_concentration`.

Recipe (matcha-croissant / #1):
  1. Pivot the raw screen (one row per molecule-dose) into two columns: log2fc @ 8.25uM and @ 33uM.
  2. On the molecules that have BOTH doses AND a real pEC50 (the overlap with TRAIN.csv), fit an
     RBF SVR mapping (fc_8p25, fc_33) -> pEC50.  [validated: 5-fold CV MAE 0.276, R2 0.717]
  3. Apply that SVR to the molecules with both doses but NO pEC50 -> impute a *proxy* pEC50.
  4. Emit those as a training set. ~83% land below 4.0, i.e. the weak examples we lack.

Leak guard: the 513 blinded test molecules have NO single-concentration data (verified: 0 overlap),
so the proxy pool is disjoint from test by construction. We assert it anyway.

Run (needs network for the HF file):  uv run python scripts/build_proxy_labels.py
Output: data/pxr_activity/proxy_train.csv  (SMILES, pEC50)  -- committed, so the GPU pod needs no HF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RDLogger.DisableLog("rdApp.*")
REPO = Path(__file__).resolve().parent.parent
HF = "hf://datasets/openadmet/pxr-challenge-train-test/"
DOSE_A, DOSE_B = 8.251e-06, 3.30e-05      # the two workhorse doses (8.25 uM, 33 uM)


def _canon(s: str) -> str | None:
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/pxr_activity")
    ap.add_argument("--out", type=Path, default=REPO / "data/pxr_activity/proxy_train.csv")
    args = ap.parse_args()

    # --- 1. pivot the screen: one row per molecule, one column per dose -------------------- #
    sc = pd.read_csv(HF + "pxr-challenge_single_concentration_TRAIN.csv")
    sc["cs"] = sc["SMILES"].map(_canon)
    sc = sc[sc["cs"].notna() & sc["concentration_M"].isin([DOSE_A, DOSE_B])]
    piv = sc.pivot_table(index="cs", columns="concentration_M", values="log2_fc_estimate", aggfunc="mean")
    piv.columns = ["fc_8p25", "fc_33"]
    piv = piv.reset_index()
    both = piv[["fc_8p25", "fc_33"]].notna().all(axis=1)
    piv = piv[both]
    print(f"[pivot] {len(piv)} molecules with BOTH doses (from {sc['cs'].nunique()} screened)")

    # --- 2. teach the SVR on molecules that have both doses AND a real pEC50 --------------- #
    tr = pd.read_csv(args.data_dir / "train.csv")
    tr["cs"] = tr["SMILES"].map(_canon)
    d = piv.merge(tr[["cs", "pEC50"]], on="cs", how="left")
    lab, unlab = d[d.pEC50.notna()], d[d.pEC50.isna()]
    X = lab[["fc_8p25", "fc_33"]].to_numpy(float)
    y = lab["pEC50"].to_numpy(float)
    print(f"[fit ] SVR train (both doses + real pEC50): {len(lab)}   to impute: {len(unlab)}")

    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1))
    oof = cross_val_predict(svr, X, y, cv=KFold(5, shuffle=True, random_state=0))
    print(f"[fit ] 5-fold CV of the SVR: MAE={np.abs(oof - y).mean():.3f}  "
          f"R2={1 - ((oof - y) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
    svr.fit(X, y)

    # --- 3. impute proxy pEC50 for the screen-rejected molecules --------------------------- #
    proxy = svr.predict(unlab[["fc_8p25", "fc_33"]].to_numpy(float))
    out = pd.DataFrame({"SMILES": unlab["cs"].to_numpy(), "pEC50": proxy})

    # --- 4. leak guard: no test molecule may enter the proxy pool -------------------------- #
    test = pd.read_csv(args.data_dir / "test.csv")
    test_c = {c for c in (_canon(s) for s in test["SMILES"]) if c}
    n_leak = len(set(out["SMILES"]) & test_c)
    if n_leak:
        raise SystemExit(f"LEAK: {n_leak} of the 513 test molecules are in the proxy pool — abort")
    print(f"[leak] OK — 0 of the {len(test_c)} test molecules appear in the proxy pool")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    w = int((proxy < 4).sum())
    print(f"[done] wrote {args.out}  ({len(out)} molecules)")
    print(f"       proxy pEC50 <4 (weak): {w} ({w / len(out) * 100:.0f}%)   "
          f"vs only {(y < 4).mean() * 100:.0f}% weak among the real labels")


if __name__ == "__main__":
    main()
