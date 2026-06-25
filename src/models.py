"""Model registry — turn a feature vector into a predicted pEC50.

Mirror of ``featurizers.py`` for the second half of the pipeline:

    numbers (features)  -->  model  -->  predicted pEC50

Every entry is a factory ``make(params, seed) -> estimator`` returning an
sklearn-style object with ``.fit(X, y)`` / ``.predict(X)``. The cross-validated
runner calls those two methods uniformly and never special-cases a model.

Leakage note: Ridge is wrapped in a ``Pipeline([StandardScaler, Ridge])`` so the
scaler — a *learned* transform — is fit on the training fold only and merely
applied to the validation fold. Tree models (RF/XGBoost) are scale-invariant and
need no scaler. ``seed`` is threaded into models that have randomness (RF,
XGBoost) and ignored by deterministic ones (Ridge).
"""

from __future__ import annotations

from typing import Callable

# name -> fn(params: dict, seed: int) -> estimator with .fit/.predict
MODELS: dict[str, Callable[[dict, int], object]] = {}


def register(name: str):
    def _wrap(fn: Callable[[dict, int], object]):
        if name in MODELS:
            raise ValueError(f"Model '{name}' already registered")
        MODELS[name] = fn
        return fn
    return _wrap


def _maybe_scale(estimator, params: dict):
    """Wrap a linear estimator in a per-fold StandardScaler (+ z-clip) when
    ``params['scale']`` is true. Scaling is representation-dependent and matters:
      - dense descriptors/embeddings: standardize (wildly different scales) -> scale=True
      - binary fingerprints (morgan/maccs): do NOT standardize. Dividing rare bits
        by their tiny std blows them to unit variance, so the model latches onto
        noisy rare substructures and generalizes terribly -> scale=False.
    The z-clip bounds any single feature's leverage: RDKit descriptors like Ipc can
    be astronomically large, so a validation molecule outside the train range becomes
    a huge z-score and a linear model emits wild predictions (we saw a fold collapse
    to R²=-361). The scaler is a learned transform, so it lives in the Pipeline and
    is fit on the training fold only.
    """
    if not params.get("scale", True):
        return estimator

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    z = float(params.get("clip_z", 10.0))
    clip = FunctionTransformer(lambda X: np.clip(X, -z, z), feature_names_out="one-to-one")
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("clip", clip),
            ("est", estimator),
        ]
    )


@register("ridge")
def _ridge(params: dict, seed: int):
    from sklearn.linear_model import Ridge

    return _maybe_scale(Ridge(alpha=float(params.get("alpha", 1.0))), params)


@register("elastic_net")
def _elastic_net(params: dict, seed: int):
    from sklearn.linear_model import ElasticNet

    est = ElasticNet(
        alpha=float(params.get("alpha", 0.1)),
        l1_ratio=float(params.get("l1_ratio", 0.5)),
        max_iter=int(params.get("max_iter", 5000)),
        random_state=seed,
    )
    return _maybe_scale(est, params)


@register("random_forest")
def _random_forest(params: dict, seed: int):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 400)),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        max_features=params.get("max_features", "sqrt"),
        random_state=seed,
        n_jobs=int(params.get("n_jobs", -1)),
    )


@register("xgboost")
def _xgboost(params: dict, seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=int(params.get("n_estimators", 500)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        min_child_weight=float(params.get("min_child_weight", 1.0)),
        random_state=seed,
        n_jobs=int(params.get("n_jobs", 1)),
        objective="reg:squarederror",
        tree_method=params.get("tree_method", "hist"),
    )


@register("lightgbm")
def _lightgbm(params: dict, seed: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=int(params.get("n_estimators", 600)),
        num_leaves=int(params.get("num_leaves", 31)),
        max_depth=int(params.get("max_depth", -1)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=float(params.get("subsample", 0.9)),
        subsample_freq=int(params.get("subsample_freq", 1)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        min_child_samples=int(params.get("min_child_samples", 20)),
        random_state=seed,
        n_jobs=int(params.get("n_jobs", -1)),
        verbose=-1,
    )


@register("catboost")
def _catboost(params: dict, seed: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=int(params.get("n_estimators", 800)),
        depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        l2_leaf_reg=float(params.get("reg_lambda", 3.0)),
        random_seed=seed,
        thread_count=int(params.get("n_jobs", -1)),
        verbose=0,
        allow_writing_files=False,
    )


@register("mlp_head")
def _mlp_head(params: dict, seed: int):
    """A small multi-layer perceptron on top of the features — the one non-tree,
    non-linear family in the menu, so it makes *different* mistakes than the GBDTs
    and adds genuine diversity to the ensemble.

    Standardisation matters a lot for an MLP (unscaled inputs stall training), so
    it reuses the same per-fold ``StandardScaler`` pipeline as the linear models
    (``scale`` defaults True; pass ``scale=False`` for binary fingerprints).
    ``early_stopping`` carves an internal validation split out of the *training
    fold only*, so it is leak-free. MLPRegressor does not accept ``sample_weight``
    — ``fit_model`` detects that and fits unweighted (recorded honestly upstream).
    L2 is read from ``mlp_alpha`` (not ``alpha``, which the linear models own).
    """
    from sklearn.neural_network import MLPRegressor

    hidden = params.get("hidden_layer_sizes", (256, 128))
    if isinstance(hidden, list):
        hidden = tuple(hidden)
    est = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=params.get("activation", "relu"),
        alpha=float(params.get("mlp_alpha", 1e-3)),
        learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
        batch_size=params.get("batch_size", "auto"),
        max_iter=int(params.get("max_iter", 300)),
        early_stopping=bool(params.get("early_stopping", True)),
        n_iter_no_change=int(params.get("n_iter_no_change", 15)),
        validation_fraction=float(params.get("validation_fraction", 0.1)),
        random_state=seed,
    )
    return _maybe_scale(est, params)


def make_model(name: str, params: dict | None = None, seed: int = 42):
    """Build a fresh estimator for model ``name`` (uses registry defaults if unknown keys)."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODELS)}")
    return MODELS[name](params or {}, seed)


def supports_sample_weight(estimator) -> bool:
    """Whether ``estimator`` (bare or a Pipeline) accepts ``sample_weight`` in ``fit``.

    For a Pipeline we check the *final* step — that is the estimator that would
    receive the weight (linear models here are wrapped in a StandardScaler pipeline).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import has_fit_parameter

    final = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    return has_fit_parameter(final, "sample_weight")


def fit_model(estimator, X, y, sample_weight=None):
    """Fit ``estimator``, routing ``sample_weight`` to the final step of a Pipeline.

    Returns ``(estimator, weight_applied)``. ``weight_applied`` is False when
    weights were supplied but the model can't use them (e.g. the MLP head) — the
    fit still succeeds, so a weighting experiment can sweep the whole menu, and the
    caller can record honestly which models actually consumed the weights.
    """
    if sample_weight is None or not supports_sample_weight(estimator):
        estimator.fit(X, y)
        return estimator, False

    from sklearn.pipeline import Pipeline

    if isinstance(estimator, Pipeline):
        step_name = estimator.steps[-1][0]
        estimator.fit(X, y, **{f"{step_name}__sample_weight": sample_weight})
    else:
        estimator.fit(X, y, sample_weight=sample_weight)
    return estimator, True


def available_models() -> list[str]:
    return sorted(MODELS)
