"""Fit an XGBoost regressor in a subprocess and write prediction arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--train-targets", required=True)
    parser.add_argument("--valid-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--valid-output", required=True)
    parser.add_argument("--test-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_train = np.load(args.train_features)
    y_train = np.load(args.train_targets)
    X_valid = np.load(args.valid_features)
    X_test = np.load(args.test_features)
    params = json.loads(Path(args.params).read_text(encoding="utf-8"))

    model = XGBRegressor(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=int(params.get("random_state", 42)),
        n_jobs=int(params.get("n_jobs", 1)),
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    valid_preds = model.predict(X_valid)
    test_preds = model.predict(X_test)

    np.save(args.valid_output, valid_preds)
    np.save(args.test_output, test_preds)


if __name__ == "__main__":
    main()
