"""#1 Approach 3: the model zoo — 5 models, 5-fold scaffold OOF, LASSO stack, blend >4.5.

#1's report, verbatim: "training a wide variety of additional models, including Chemprop, Chameleon,
Unimol2, TabICL, and SVRs. These models were trained on 5-fold scaffold split of the train pEC50
dataset, and ensembled via LASSO regression on the out-of-fold predictions. This resulting model
performed well on high-pEC50 compounds, and was blended with the first approach in cases where it
predicted pEC50 greater than 4.5."

The 5 members (single-task pEC50, on the calibrated 5 folds):
  1. Chemprop     — plain D-MPNN, NO foundation (trained here, GPU)
  2. Chameleon    — CheMeleon; reuse predictions/{oof,test}_cheme_mt5 (a trained CheMeleon)
  3. Unimol2      — reuse predictions/{oof,test}_unimol (our Uni-Mol, stand-in for Unimol2 v2)
  4. TabICL       — TabICLRegressor on RDKit descriptors (trained here)
  5. SVR          — RBF SVR on Morgan fingerprints (trained here)
LASSO (LassoCV) stacks the 5 OOF vectors -> weights -> applied to the test -> zoo prediction.
Blend: where the zoo predicts pEC50 > 4.5, equal-weight average into Approach 1 (test_approach1_desc).
NO calibration, NO Set-1 tuning; Set-2 scored once. (4.5 threshold is #1's.)

Run on the GPU pod:  nohup python reproduce_1/approach3_zoo.py > zoo.log 2>&1 &
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.metrics import score_all  # noqa: E402

RDLogger.DisableLog("rdApp.*")
NAME, TGT, SMILES = "Molecule Name", "pEC50", "SMILES"
DATA = REPO / "data/pxr_activity"
PRED = REPO / "predictions"
OUT = Path("/tmp/zoo")
_DNAMES = [n for n, _ in Descriptors._descList]
_DCALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DNAMES)


def morgan(sl, nb=2048):
    X = np.zeros((len(sl), nb), np.float32)
    for i, s in enumerate(sl):
        m = Chem.MolFromSmiles(str(s))
        if m:
            for b in AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nb).GetOnBits():
                X[i, b] = 1.0
    return X


def descriptors(sl):
    X = np.zeros((len(sl), len(_DNAMES)), np.float32)
    for i, s in enumerate(sl):
        m = Chem.MolFromSmiles(str(s))
        if m:
            X[i] = _DCALC.CalcDescriptors(m)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def sklearn_oof(make, Xtr, Xte, y, fold):
    """5-fold OOF + averaged test for a fresh-per-fold sklearn-style model."""
    oof = np.zeros(len(y)); te = np.zeros(Xte.shape[0])
    for k in sorted(set(fold)):
        m = make(); m.fit(Xtr[fold != k], y[fold != k])
        oof[fold == k] = m.predict(Xtr[fold == k]); te += m.predict(Xte) / len(set(fold))
    return oof, te


def chemprop_plain(train_df, test_df, fold, accel="gpu", epochs=50, workers=8):
    """Plain Chemprop D-MPNN (NO --from-foundation), single-task pEC50, 5-fold OOF + test."""
    OUT.mkdir(parents=True, exist_ok=True)
    test_df[[SMILES]].to_csv(OUT / "_te.csv", index=False)
    oof = np.full(len(train_df), np.nan); te_cols = []
    for k in sorted(set(fold)):
        tr = train_df[fold != k][[SMILES, TGT]]; tr.to_csv(OUT / f"_tr{k}.csv", index=False)
        train_df[fold == k][[SMILES]].to_csv(OUT / f"_hd{k}.csv", index=False)
        subprocess.run(["chemprop", "train", "-i", str(OUT / f"_tr{k}.csv"), "-s", SMILES,
                        "--target-columns", TGT, "-t", "regression", "--loss-function", "mae",
                        "--epochs", str(epochs), "--warmup-epochs", "2", "--split-sizes", "0.9", "0.1", "0.0",
                        "-o", str(OUT / f"cp{k}"), "--accelerator", accel, "-n", str(workers)], check=True)
        ck = (glob.glob(str(OUT / f"cp{k}" / "**" / "best*.ckpt"), recursive=True)
              or glob.glob(str(OUT / f"cp{k}" / "**" / "*.ckpt"), recursive=True))[0]
        for src, dst in [(OUT / f"_hd{k}.csv", OUT / f"_ho{k}.csv"), (OUT / "_te.csv", OUT / f"_to{k}.csv")]:
            subprocess.run(["chemprop", "predict", "--test-path", str(src), "-s", SMILES,
                            "--model-paths", ck, "--preds-path", str(dst), "--accelerator", accel,
                            "-n", str(workers)], check=True)
        oof[fold == k] = pd.read_csv(OUT / f"_ho{k}.csv")[TGT].to_numpy(float)
        te_cols.append(pd.read_csv(OUT / f"_to{k}.csv")[TGT].to_numpy(float))
    return oof, np.mean(np.column_stack(te_cols), axis=1)


def main():
    train = pd.read_csv(DATA / "train.csv").reset_index(drop=True)          # 4139 broad, SMILES+pEC50
    test = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    y = train[TGT].to_numpy(float)
    fold = np.array([int(json.loads((DATA / "folds_calibrated.json").read_text())["assignments"][str(i)])
                     for i in range(len(train))])
    tnames = test[NAME].to_numpy()

    def reuse(oof_csv, test_csv):
        o = pd.read_csv(PRED / oof_csv).sort_values("row_id")["y_pred"].to_numpy(float)
        t = pd.read_csv(PRED / test_csv).set_index(NAME).reindex(tnames)[TGT].to_numpy(float)
        return o, t

    print("[members] building 5 zoo members (5-fold OOF + test) ...", flush=True)
    Xd_tr, Xd_te = descriptors(train[SMILES].tolist()), descriptors(test[SMILES].tolist())
    Xm_tr, Xm_te = morgan(train[SMILES].tolist()), morgan(test[SMILES].tolist())
    from tabicl import TabICLRegressor

    members = {}
    members["chemeleon"] = reuse("oof_cheme_mt5.csv", "test_cheme_mt5.csv")   # 2. Chameleon
    members["unimol"] = reuse("oof_unimol.csv", "test_unimol.csv")            # 3. Unimol2 (stand-in)
    members["svr"] = sklearn_oof(lambda: make_pipeline(StandardScaler(with_mean=False),
                                 SVR(kernel="rbf", C=10, gamma="scale")), Xm_tr, Xm_te, y, fold)  # 5. SVR
    members["tabicl"] = sklearn_oof(lambda: TabICLRegressor(), Xd_tr, Xd_te, y, fold)              # 4. TabICL
    members["chemprop"] = chemprop_plain(train, test, fold)                   # 1. Chemprop (GPU) — last (slow)

    for name, (o, _) in members.items():
        print(f"  {name:10s} OOF RAE={score_all(y, o)['rae']:.4f}")

    order = ["chemprop", "chemeleon", "unimol", "tabicl", "svr"]
    OOF = np.column_stack([members[n][0] for n in order])
    TE = np.column_stack([members[n][1] for n in order])
    las = LassoCV(positive=True, cv=5, max_iter=100000).fit(OOF, y)
    zoo = las.predict(TE)
    print(f"\n[LASSO] weights {dict(zip(order, np.round(las.coef_, 3)))}  intercept {las.intercept_:.3f}")
    print(f"[LASSO] zoo OOF RAE={score_all(y, las.predict(OOF))['rae']:.4f}")

    a1 = pd.read_csv(PRED / "test_approach1_desc.csv").set_index(NAME).reindex(tnames)[TGT].to_numpy(float)
    p2 = pd.read_csv(DATA / "phase2_unblinded.csv")[[NAME, TGT]].dropna(subset=[TGT])

    def sc(pred, tag):
        m = p2.merge(pd.DataFrame({NAME: tnames, "y": pred}), on=NAME)
        s = score_all(m[TGT].to_numpy(float), m["y"].to_numpy(float))
        print(f"    {tag:34s} RAE={s['rae']:.4f}  MAE={s['mae']:.4f}"); return s["rae"]

    strong = zoo > 4.5
    blend = a1.copy(); blend[strong] = 0.5 * a1[strong] + 0.5 * zoo[strong]
    pd.DataFrame({NAME: tnames, SMILES: test[SMILES], TGT: zoo}).to_csv(PRED / "test_zoo.csv", index=False)
    print(f"\n=== Set-2 (260 blind) — 5-member zoo, NO calibration ===  [zoo>4.5 on {strong.sum()}/{len(zoo)}]")
    sc(a1, "Approach 1 (descriptors)")
    sc(zoo, "zoo alone (LASSO of 5)")
    sc(blend, "+ zoo blend >4.5")
    print("  reference: #1 (matcha) 0.5631 | #2 (best public) 0.5676 | rank-12 0.586")


if __name__ == "__main__":
    main()
