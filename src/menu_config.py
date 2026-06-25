"""Shared menu configuration — the building blocks both the runner and tuner use.

Kept separate from the scripts so `run_menu`, `run_ensemble`, and `tune` all agree
on what a representation/model is, and their defaults can't drift apart.
"""

from __future__ import annotations

CHEMBERTA = "DeepChem/ChemBERTa-77M-MTR"
CHEMBERTA_100M = "DeepChem/ChemBERTa-100M-MLM"
MOLFORMER = "ibm-research/MoLFormer-XL-both-10pct"

# A "representation" = a featurizer + its params. "binary" flags 0/1 fingerprints
# (linear models must NOT standardize those — see models._maybe_scale).
REPRESENTATIONS: dict[str, dict] = {
    "morgan":            {"featurizer": "morgan", "params": {}, "binary": True},
    "maccs":             {"featurizer": "maccs", "params": {}, "binary": True},
    "avalon":            {"featurizer": "avalon", "params": {"n_bits": 1024}, "binary": True},
    "rdkit_descriptors": {"featurizer": "rdkit_descriptors", "params": {}, "binary": False},
    "chemberta":         {"featurizer": "chemberta_embedding", "params": {"skill_ref": CHEMBERTA}, "binary": False},
    "chemberta100m":     {"featurizer": "chemberta_embedding", "params": {"skill_ref": CHEMBERTA_100M}, "binary": False},
    "molformer":         {"featurizer": "molformer_embedding", "params": {"skill_ref": MOLFORMER}, "binary": False},
    "chemeleon":         {"featurizer": "chemeleon_embedding", "params": {}, "binary": False},
    "fusion_desc_cb":    {"featurizer": "fusion", "params": {"components": ["rdkit_descriptors", "chemberta_embedding"], "skill_ref": CHEMBERTA}, "binary": False},
    "fusion_desc_mf":    {"featurizer": "fusion", "params": {"components": ["rdkit_descriptors", "molformer_embedding"], "skill_ref": MOLFORMER}, "binary": False},
    "fusion_desc_cheme": {"featurizer": "fusion", "params": {"components": ["rdkit_descriptors", "chemeleon_embedding"]}, "binary": False},
}

CHEAP_REPS = ["morgan", "maccs", "avalon", "rdkit_descriptors"]
EMBED_REPS = ["chemberta", "chemberta100m", "molformer", "chemeleon", "fusion_desc_cb", "fusion_desc_mf", "fusion_desc_cheme"]

ALL_MODELS = ["ridge", "elastic_net", "random_forest", "xgboost", "lightgbm", "catboost"]

# Reasonable starting hyperparameters for the first sweep (the Tuner refines from here).
MODEL_DEFAULTS: dict[str, dict] = {
    "ridge": {"alpha": 1.0},
    "elastic_net": {"alpha": 0.1, "l1_ratio": 0.5},
    "random_forest": {"n_estimators": 400, "max_features": "sqrt"},
    "xgboost": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
    "lightgbm": {"n_estimators": 600, "learning_rate": 0.05},
    "catboost": {"n_estimators": 800, "learning_rate": 0.05},
    "mlp_head": {"hidden_layer_sizes": (256, 128), "mlp_alpha": 1e-3, "learning_rate_init": 1e-3},
}

# Deterministic linear models ignore the seed; only these benefit from multi-seed.
# The MLP head is stochastic (random init + SGD), so it also wants multi-seed.
STOCHASTIC_MODELS = {"random_forest", "xgboost", "lightgbm", "catboost", "mlp_head"}
LINEAR_MODELS = {"ridge", "elastic_net"}
# Models that need standardized inputs on dense reps (and must NOT scale binary
# fingerprints — see models._maybe_scale). The MLP joins the linear models here.
SCALED_MODELS = LINEAR_MODELS | {"mlp_head"}
SEED_POOL = [42, 1, 2, 7, 13]


def rep_base_params(rep_label: str, model: str) -> dict:
    """Featurizer params for a representation + the per-fold scaling flag a
    scale-sensitive model needs. (Model hyperparameters are layered on top by the caller.)"""
    spec = REPRESENTATIONS[rep_label]
    params = dict(spec["params"])
    if model in SCALED_MODELS:
        params["scale"] = not spec["binary"]
    return params
