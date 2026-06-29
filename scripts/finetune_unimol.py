"""Uni-Mol 3D fine-tune (run on GPU, e.g. A5000) — the DECORRELATED ensemble member.

The 0.538 solution's edge was a 3-family ensemble (graph + 3D + foundation). Our stack
is all CheMeleon-flavoured and saturates, so this adds a genuinely different model: a 3D
molecular transformer (Uni-Mol, pretrained on ~209M conformers). It is weaker standalone
(their OOF ~0.645) but its 3D geometry errors are de-correlated from the D-MPNN, so it
lifts the Ridge ensemble.

Same 5-fold calibrated-OOF protocol as the chemprop scripts → oof_unimol.csv + test_unimol.csv,
ready to drop into the aggregator. MAE objective (their JG_MAE finding). Optional `--tta N`
averages predictions over N randomized-SMILES variants per molecule (their "aug10").

Set 1 is never touched. Reactive electrophiles dropped from TRAINING only (is_reactive column).

Deps (pod):  pip install 'unimol_tools' huggingface_hub  (weights auto-download on first run)
Needs (same folder): train.csv, test.csv, folds_calibrated.json
Smoke first:  python finetune_unimol.py --smoke --accelerator gpu
Full:         python finetune_unimol.py --epochs 40 --tta 10 --accelerator gpu
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
SMILES, NAME, TARGET = "SMILES", "Molecule Name", "pEC50"

# Reactive-electrophile alerts (self-contained copy of src/curation.py so the pod script
# needs no extra files). Dropped from TRAINING only; the test set contains none.
_REACTIVE_SMARTS = [
    "[CX3]=[CX3]C(=O)[NX3]", "[CX3]=[CX3]C(=O)[OX2]", "[CX3]=[CX3]C(=O)[#6;!$([CX3]=O)]",
    "[CX3H1](=O)[#6]", "O=C1C=CC(=O)N1", "[CX3]=[CX3][SX4](=O)(=O)",
    "[CX3](=O)[CX4;H1,H2][F,Cl,Br,I]", "[CX3]1[OX2][CX3]1", "[NX2]=[CX2]=[OX1]", "[CX3]=[CX3]C#N",
]
_REACTIVE_PATTS = [p for p in (Chem.MolFromSmarts(s) for s in _REACTIVE_SMARTS) if p is not None]


def _reactive_mask(smiles_list) -> np.ndarray:
    def hit(s):
        m = Chem.MolFromSmiles(str(s))
        return bool(m) and any(m.HasSubstructMatch(p) for p in _REACTIVE_PATTS)
    return np.array([hit(s) for s in smiles_list], dtype=bool)


def _randomized_smiles(smi: str, n: int, seed: int) -> list[str]:
    """n random-atom-order SMILES for one molecule (test-time augmentation); falls back to canonical."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return [str(smi)] * n
    out = []
    for i in range(n):
        try:
            out.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
        except Exception:
            out.append(Chem.MolToSmiles(mol))
    return out


def _fit_predict(train_csv: Path, pred_frames: dict[str, pd.DataFrame], work: Path,
                 epochs: int, lr: float, batch: int, num_workers: int = 8) -> dict[str, np.ndarray]:
    """Train one Uni-Mol model on train_csv; return {key: predictions} for each frame in pred_frames."""
    from unimol_tools import MolTrain, MolPredict

    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    kw = dict(
        task="regression", data_type="molecule",
        epochs=epochs, learning_rate=lr, batch_size=batch,
        early_stopping=max(5, epochs // 4), metrics="mae",
        kfold=1,   # we already do our OWN 5-fold OOF outside; unimol's internal kfold (default 5) would be 5x redundant compute
        smiles_col=SMILES, target_cols=TARGET, save_path=str(work),
    )
    try:
        clf = MolTrain(**kw, num_workers=num_workers)   # speed: parallel dataloader (if this unimol_tools accepts it)
    except TypeError:
        clf = MolTrain(**kw)                            # older unimol_tools: no such arg, fall back
    clf.fit(data=str(train_csv))
    predictor = MolPredict(load_model=str(work))
    results = {}
    for key, frame in pred_frames.items():
        tmp = work / f"_pred_{key}.csv"
        frame[[SMILES]].to_csv(tmp, index=False)
        results[key] = np.asarray(predictor.predict(data=str(tmp))).ravel()
    return results


def _tta_predict(predictor_dir: Path, smiles: list[str], tta: int) -> np.ndarray:
    """Average predictions over `tta` randomized-SMILES variants per molecule."""
    from unimol_tools import MolPredict
    predictor = MolPredict(load_model=str(predictor_dir))
    acc = np.zeros(len(smiles))
    for j in range(tta):
        variants = [_randomized_smiles(s, 1, seed=j)[0] for s in smiles]
        tmp = predictor_dir / f"_tta_{j}.csv"
        pd.DataFrame({SMILES: variants}).to_csv(tmp, index=False)
        acc += np.asarray(predictor.predict(data=str(tmp))).ravel()
    return acc / tta


def main() -> None:
    ap = argparse.ArgumentParser(description="Uni-Mol 3D fine-tune, 5-fold calibrated OOF + test.")
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("unimol_out"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64, help="GPU batch (raised from 32; note conformer-gen CPU is the real bottleneck)")
    ap.add_argument("--num-workers", type=int, default=8, help="dataloader workers (if this unimol_tools accepts it)")
    ap.add_argument("--tta", type=int, default=0, help="randomized-SMILES test-time augmentation (their aug10 => 10)")
    ap.add_argument("--accelerator", default="gpu")  # unimol_tools picks CUDA automatically when present
    ap.add_argument("--keep-reactive", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 folds, 2 epochs, 60 rows — verify the pipeline (2 folds so a downstream ridge stack has a CV train split)")
    ap.add_argument("--tta-only", action="store_true",
                    help="skip training; load existing ckpt_f{k} and TTA-predict OOF+test (writes *_tta.csv)")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv").reset_index(drop=True)
    test = pd.read_csv(args.data_dir / "test.csv").reset_index(drop=True)
    folds = json.loads((args.data_dir / "folds_calibrated.json").read_text())["assignments"]
    fold_of_row = np.array([int(folds[str(i)]) for i in range(len(train))])

    reactive = (np.zeros(len(train), bool) if args.keep_reactive
                else train["is_reactive"].to_numpy(int).astype(bool) if "is_reactive" in train.columns
                else _reactive_mask(train[SMILES].tolist()))

    epochs, tta = args.epochs, args.tta
    if args.tta_only and tta == 0:
        tta = 10  # default to aug10 when only doing TTA prediction
    fold_ids = sorted(set(fold_of_row.tolist()))
    if args.smoke:
        epochs, tta, fold_ids = 2, 0, fold_ids[:2]   # 2 folds so OOF spans 2 folds -> stack's ridge CV has a train split
        train = train.iloc[:60].reset_index(drop=True)
        fold_of_row, reactive = fold_of_row[:60], reactive[:60]

    y = train[TARGET].to_numpy(float)
    oof = np.full(len(train), np.nan)
    test_cols = []

    for k in fold_ids:
        print(f"\n===== UNI-MOL FOLD {k}{' (TTA-only)' if args.tta_only else ''} =====", flush=True)
        va_mask = fold_of_row == k
        work = out / f"ckpt_f{k}"

        if args.tta_only:
            # reuse the already-trained fold model; TTA-predict held-out (OOF) AND test for a consistent stack
            if not work.exists():
                raise RuntimeError(f"--tta-only needs trained models; {work} not found")
            oof[va_mask] = _tta_predict(work, train[va_mask][SMILES].tolist(), tta)
            test_cols.append(_tta_predict(work, test[SMILES].tolist(), tta))
            print(f"fold {k}: {int(va_mask.sum())} OOF rows TTA-predicted (tta={tta})", flush=True)
            continue

        tr_mask = (fold_of_row != k) & (~reactive)
        fold_train = out / f"_train_f{k}.csv"
        train[tr_mask][[SMILES, TARGET]].to_csv(fold_train, index=False)
        # fit once on this fold's curated training set, predict the held-out broad rows (OOF)
        held = train[va_mask].reset_index(drop=True)
        oof[va_mask] = _fit_predict(fold_train, {"held": held}, work, epochs, args.lr, args.batch, args.num_workers)["held"]
        # test predictions from the SAME fitted model: plain, or averaged over `tta` randomized SMILES
        if tta:
            test_cols.append(_tta_predict(work, test[SMILES].tolist(), tta))
        else:
            from unimol_tools import MolPredict
            test[[SMILES]].to_csv(work / "_test.csv", index=False)
            test_cols.append(np.asarray(MolPredict(load_model=str(work)).predict(data=str(work / "_test.csv"))).ravel())
        print(f"fold {k}: {int(va_mask.sum())} OOF rows predicted", flush=True)

    suffix = "_tta" if args.tta_only else ""
    done = ~np.isnan(oof)
    pd.DataFrame({"row_id": np.arange(len(train))[done], SMILES: train[SMILES][done],
                  "y_true": y[done], "y_pred": oof[done]}).to_csv(out / f"oof_unimol{suffix}.csv", index=False)
    pd.DataFrame({NAME: test[NAME], SMILES: test[SMILES],
                  TARGET: np.mean(np.column_stack(test_cols), axis=1)}).to_csv(out / f"test_unimol{suffix}.csv", index=False)
    print(f"\nDONE. wrote:\n  {out/('oof_unimol'+suffix+'.csv')} ({int(done.sum())} rows)\n  {out/('test_unimol'+suffix+'.csv')} ({len(test)} rows)")


if __name__ == "__main__":
    main()
