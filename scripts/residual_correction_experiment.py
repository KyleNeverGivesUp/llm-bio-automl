"""#6's lever, generalized: a learned residual (bias) correction — the general post-hoc fix.

Unlike the proxy specialist (#1), this needs NO auxiliary assay data, so it is the one error-fixing
method that transfers to Lipophilicity / ESOL. Idea (matches #6's over/under classifier, but as a
regressor): train a small model that predicts, from structure alone, how much the main model will
mis-predict a molecule, then subtract that predicted residual — capped, so a wrong guess costs little.

Pipeline (all local, no GPU):
  1. Take the main ensemble's out-of-fold predictions on the 4,139 training molecules, apply the same
     variance-match calibration, and compute the residual (calibrated_pred - true) for each.
  2. Featurize with Morgan fingerprints. Cross-validate a RandomForest that maps structure -> residual.
     The CV correlation answers the key question up front: *is the residual structure-predictable at all?*
     (If it is cliff/noise-driven, as the diagnosis suggests, correlation is ~0 and the method cannot help.)
  3. Fit on all training residuals; predict the residual for each test molecule.
  4. Corrected = calibrated_pred - clip(predicted_residual, -cap, +cap).

Honesty protocol: the regressor is trained only on training molecules; the cap is chosen on Set-1
(dev); Set-2 is scored ONCE. Pre-registered cap grid from #6's ±0.2.

Run:  uv run python scripts/residual_correction_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.metrics import score_all  # noqa: E402

RDLogger.DisableLog("rdApp.*")
NAME, TGT, SMILES = "Molecule Name", "pEC50", "SMILES"
DATA = REPO / "data/pxr_activity"


def morgan(smiles_list, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """ECFP-style Morgan fingerprint: a structure -> fixed 0/1 vector. Local, no GPU."""
    out = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        for b in fp.GetOnBits():
            out[i, b] = 1.0
    return out


def variance_match(oof_csv: Path) -> tuple[float, float]:
    d = pd.read_csv(oof_csv)
    y, p = d["y_true"].to_numpy(float), d["y_pred"].to_numpy(float)
    a = y.std() / p.std()
    return a, y.mean() - a * p.mean()


def score(names, pred, labels, tag, quiet=False) -> float:
    m = labels.merge(pd.DataFrame({NAME: names, "yhat": pred}), on=NAME, how="left")
    if m["yhat"].isna().any():
        raise SystemExit(f"{tag}: missing predictions")
    s = score_all(m[TGT].to_numpy(float), m["yhat"].to_numpy(float))
    if not quiet:
        print(f"    {tag:40s} RAE={s['rae']:.4f}  MAE={s['mae']:.4f}  R2={s['r2']:.4f}  (n={len(m)})")
    return s["rae"]


def main() -> None:
    # --- 1. training residuals of the CALIBRATED main ensemble ---------------------------- #
    a, b = variance_match(DATA.parent.parent / "predictions/oof_ensemble.csv")
    oof_e = pd.read_csv(REPO / "predictions/oof_ensemble.csv")           # row_id, y_true, y_pred
    oof_c = pd.read_csv(REPO / "predictions/oof_cheme_mt5.csv")[["row_id", SMILES]]  # row_id -> SMILES
    d = oof_e.merge(oof_c, on="row_id", how="inner")
    d["cal_pred"] = a * d["y_pred"] + b
    d["resid"] = d["cal_pred"] - d["y_true"]                             # what we want to predict & subtract
    print(f"[data] {len(d)} training molecules; residual mean={d['resid'].mean():+.3f} std={d['resid'].std():.3f}")

    X = morgan(d[SMILES].tolist())
    y_res = d["resid"].to_numpy(float)

    # --- 2. can structure predict the residual at all? (the make-or-break check) ---------- #
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
    oof_res = cross_val_predict(rf, X, y_res, cv=KFold(5, shuffle=True, random_state=0), n_jobs=-1)
    corr = np.corrcoef(oof_res, y_res)[0, 1]
    print(f"[check] 5-fold CV corr(predicted residual, true residual) = {corr:.3f}")
    print("        (near 0 => residual is cliff/noise-driven and NOT structure-predictable => cannot help)")

    # --- 3. fit on all, predict test residuals -------------------------------------------- #
    rf.fit(X, y_res)
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    main_df = pd.read_csv(REPO / "predictions/test_ensemble.csv")
    assert (main_df[NAME].to_numpy() == test[NAME].to_numpy()).all()
    cal_test = a * main_df[TGT].to_numpy(float) + b
    pred_res_test = rf.predict(morgan(test[SMILES].tolist()))

    p1 = pd.read_csv(DATA / "phase1_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])
    p2 = pd.read_csv(DATA / "phase2_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])

    # --- 4. pick the cap on Set-1 (dev); Set-2 scored once -------------------------------- #
    print("\n=== choosing correction cap on Set-1 (dev) ===")
    best_cap, best_r = 0.0, score(test[NAME], cal_test, p1, "cap=0.00 (no correction)")
    for cap in (0.1, 0.2, 0.3, 0.5):
        corrected = cal_test - np.clip(pred_res_test, -cap, cap)
        r = score(test[NAME], corrected, p1, f"cap={cap:.2f}")
        if r < best_r:
            best_cap, best_r = cap, r
    print(f"  -> selected cap={best_cap:.2f} (Set-1 RAE {best_r:.4f})")

    corrected = cal_test - np.clip(pred_res_test, -best_cap, best_cap)
    print("\n=== Set-2 (260 blind, final scorer) — scored once ===")
    score(test[NAME], a * main_df[TGT].to_numpy(float) + b, p2, "1. calibrated ensemble (current best)")
    score(test[NAME], corrected, p2, f"2. + residual correction (cap={best_cap:.2f})")
    print("\n  reference: #2 (best public-data) 0.5676 | rank-12 0.586 | #1 (proprietary) 0.5631")

    # where did the correction land?
    for tag, pred in [("before", cal_test), (f"after (cap={best_cap:.2f})", corrected)]:
        m = p2.merge(pd.DataFrame({NAME: test[NAME], "yhat": pred}), on=NAME)
        m["r"] = m["yhat"] - m[TGT]
        m["bin"] = pd.cut(m[TGT], [0, 4, 5, 6, 99], labels=["<4", "4-5", "5-6", ">6"], right=False)
        print(f"\n  [{tag}] Set-2 bias by bin: " +
              "  ".join(f"{lb}:{m[m['bin'] == lb]['r'].mean():+.2f}(n{len(m[m['bin'] == lb])})"
                        for lb in ["<4", "4-5", "5-6", ">6"]))


if __name__ == "__main__":
    main()
