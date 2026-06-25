import json
import subprocess
import sys
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from src.schemas import PlanSpec
from src.agent.models import get_model_bundle


def featurize_morgan(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    features = []
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features.append(np.zeros(n_bits, dtype=np.float32))
            continue

        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features.append(arr)

    return np.vstack(features)


def featurize_rdkit_descriptors(smiles_list: list[str]) -> np.ndarray:
    raise NotImplementedError("RDKit descriptor featurization will be added in Version 2.")


def featurize_skill_embeddings(
    smiles_list: list[str],
    skill_ref: str,
    pooling: str = "mean",
    max_length: int = 256,
    batch_size: int = 16,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    import torch

    if cache_dir is not None:
        cache_path = _embedding_cache_path(
            cache_dir=cache_dir,
            smiles_list=smiles_list,
            skill_ref=skill_ref,
            pooling=pooling,
            max_length=max_length,
        )
        if cache_path.exists():
            return np.load(cache_path)
    else:
        cache_path = None

    tokenizer, model, device = get_model_bundle(skill_ref)

    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch = smiles_list[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value for key, value in inputs.items()}

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            if pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                masked_hidden = hidden * attention_mask
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

            embeddings.append(pooled.cpu().numpy().astype(np.float32))

    features = np.vstack(embeddings)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)

    return features


def build_model(plan: PlanSpec):
    if plan.model_type == "ridge":
        return Ridge(alpha=plan.params.get("alpha", 1.0))

    if plan.model_type == "elastic_net":
        return ElasticNet(
            alpha=plan.params.get("alpha", 0.1),
            l1_ratio=plan.params.get("l1_ratio", 0.5),
            max_iter=plan.params.get("max_iter", 5000),
            random_state=plan.params.get("random_state", 42),
        )

    if plan.model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=plan.params.get("n_estimators", 300),
            max_depth=plan.params.get("max_depth", 12),
            random_state=plan.params.get("random_state", 42),
            n_jobs=-1,
        )

    if plan.model_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=plan.params.get("n_estimators", 300),
            max_depth=plan.params.get("max_depth", 6),
            learning_rate=plan.params.get("learning_rate", 0.05),
            subsample=plan.params.get("subsample", 0.9),
            colsample_bytree=plan.params.get("colsample_bytree", 0.9),
            reg_alpha=plan.params.get("reg_alpha", 0.0),
            reg_lambda=plan.params.get("reg_lambda", 1.0),
            random_state=plan.params.get("random_state", 42),
            n_jobs=plan.params.get("n_jobs", 1),
            objective="reg:squarederror",
        )

    raise ValueError(f"Unsupported model_type: {plan.model_type}")


def build_features(plan: PlanSpec, train_df: pd.DataFrame, test_df: pd.DataFrame,  cache_dir: str | Path | None = None,):
    if plan.feature_type == "morgan_fingerprint":
        radius = plan.params.get("radius", 2)
        n_bits = plan.params.get("n_bits", 2048)

        X_train = featurize_morgan(train_df["SMILES"].tolist(), radius=radius, n_bits=n_bits)
        X_test = featurize_morgan(test_df["SMILES"].tolist(), radius=radius, n_bits=n_bits)
        return X_train, X_test

    if plan.feature_type == "rdkit_descriptors":
        X_train = featurize_rdkit_descriptors(train_df["SMILES"].tolist())
        X_test = featurize_rdkit_descriptors(test_df["SMILES"].tolist())
        return X_train, X_test

    if plan.feature_type == "skill_embedding":
        if not plan.skill_ref:
            raise ValueError("skill_embedding plans must define skill_ref.")

        pooling = plan.params.get("pooling", "mean")
        max_length = int(plan.params.get("max_length", 256))
        batch_size = int(plan.params.get("batch_size", 16))

        X_train = featurize_skill_embeddings(
            train_df["SMILES"].tolist(),
            skill_ref=plan.skill_ref,
            pooling=pooling,
            max_length=max_length,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
        X_test = featurize_skill_embeddings(
            test_df["SMILES"].tolist(),
            skill_ref=plan.skill_ref,
            pooling=pooling,
            max_length=max_length,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
        return X_train, X_test

    if plan.feature_type == "skill_embedding_plus_morgan":
        if not plan.skill_ref:
            raise ValueError("skill_embedding_plus_morgan plans must define skill_ref.")

        pooling = plan.params.get("pooling", "mean")
        max_length = int(plan.params.get("max_length", 256))
        batch_size = int(plan.params.get("batch_size", 16))
        radius = int(plan.params.get("radius", 2))
        n_bits = int(plan.params.get("n_bits", 2048))

        embedding_train = featurize_skill_embeddings(
            train_df["SMILES"].tolist(),
            skill_ref=plan.skill_ref,
            pooling=pooling,
            max_length=max_length,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
        embedding_test = featurize_skill_embeddings(
            test_df["SMILES"].tolist(),
            skill_ref=plan.skill_ref,
            pooling=pooling,
            max_length=max_length,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )

        morgan_train = featurize_morgan(train_df["SMILES"].tolist(), radius=radius, n_bits=n_bits)
        morgan_test = featurize_morgan(test_df["SMILES"].tolist(), radius=radius, n_bits=n_bits)

        X_train = np.hstack([embedding_train, morgan_train])
        X_test = np.hstack([embedding_test, morgan_test])
        return X_train, X_test

    raise ValueError(f"Unsupported feature_type: {plan.feature_type}")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    baseline = np.mean(y_true)
    denominator = np.sum(np.abs(y_true - baseline))
    numerator = np.sum(np.abs(y_true - y_pred))
    rae = float(numerator / denominator) if denominator > 0 else float("inf")

    return {
        "mae": float(mae),
        "rae": rae,
        "r2": float(r2),
    }


def save_plan_metrics(metrics: dict, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_predictions(df: pd.DataFrame, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def _fit_predict_xgboost_subprocess(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    work_dir: Path,
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    fold_dir = work_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_features_path = fold_dir / "train_features.npy"
    train_targets_path = fold_dir / "train_targets.npy"
    valid_features_path = fold_dir / "valid_features.npy"
    test_features_path = fold_dir / "test_features.npy"
    valid_preds_path = fold_dir / "valid_preds.npy"
    test_preds_path = fold_dir / "test_preds.npy"
    params_path = fold_dir / "params.json"

    np.save(train_features_path, X_tr)
    np.save(train_targets_path, y_tr)
    np.save(valid_features_path, X_va)
    np.save(test_features_path, X_test)
    params_payload = dict(params)
    params_payload["n_jobs"] = int(params_payload.get("n_jobs", 1))
    params_path.write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "src.xgboost_worker",
        "--train-features",
        str(train_features_path),
        "--train-targets",
        str(train_targets_path),
        "--valid-features",
        str(valid_features_path),
        "--test-features",
        str(test_features_path),
        "--params",
        str(params_path),
        "--valid-output",
        str(valid_preds_path),
        "--test-output",
        str(test_preds_path),
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "XGBoost subprocess failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return np.load(valid_preds_path), np.load(test_preds_path)


def run_plan(plan: PlanSpec, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_root = output_path.parents[1] if len(output_path.parents) > 1 else output_path.parent
    run_cache_dir = cache_root / "embedding_cache"
    X_train, X_test = build_features(plan, train_df, test_df, cache_dir=run_cache_dir)
    y_train = train_df["pEC50"].values.astype(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_fold_preds = []

    for train_idx, valid_idx in kf.split(X_train):
        X_tr, X_va = X_train[train_idx], X_train[valid_idx]
        y_tr = y_train[train_idx]
        if plan.model_type == "xgboost":
            valid_preds, test_preds = _fit_predict_xgboost_subprocess(
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                X_test=X_test,
                params=plan.params,
                work_dir=output_path / "_xgboost_worker",
                fold_idx=len(test_fold_preds),
            )
            oof_preds[valid_idx] = valid_preds
            test_fold_preds.append(test_preds)
        else:
            model = build_model(plan)
            model.fit(X_tr, y_tr)

            oof_preds[valid_idx] = model.predict(X_va)
            test_fold_preds.append(model.predict(X_test))

    test_preds = np.mean(np.vstack(test_fold_preds), axis=0)

    metrics = compute_regression_metrics(y_train, oof_preds)
    metrics.update(
        {
            "plan_id": plan.plan_id,
            "plan_name": plan.name,
            "feature_type": plan.feature_type,
            "model_type": plan.model_type,
            "skill_ref": plan.skill_ref,
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
        }
    )

    oof_df = pd.DataFrame(
        {
            "SMILES": train_df["SMILES"],
            "Molecule Name": train_df["Molecule Name"],
            "y_true": y_train,
            "y_pred": oof_preds,
        }
    )

    test_pred_df = pd.DataFrame(
        {
            "SMILES": test_df["SMILES"],
            "Molecule Name": test_df["Molecule Name"],
            "pEC50": test_preds,
        }
    )

    save_plan_metrics(metrics, output_path / "metrics.json")
    save_predictions(oof_df, output_path / "oof_predictions.csv")
    save_predictions(test_pred_df, output_path / "test_predictions.csv")

    return metrics


def _embedding_cache_path(
    cache_dir: str | Path,
    smiles_list: list[str],
    skill_ref: str,
    pooling: str,
    max_length: int,
) -> Path:
    smiles_blob = "\n".join(smiles_list).encode("utf-8")
    smiles_hash = hashlib.sha256(smiles_blob).hexdigest()[:16]
    skill_slug = skill_ref.replace("/", "--").replace(".", "-").replace("_", "-")
    filename = f"{skill_slug}_{pooling}_{max_length}_{smiles_hash}.npy"
    return Path(cache_dir) / filename
