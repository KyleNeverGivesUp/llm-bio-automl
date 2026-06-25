"""Featurizer registry — turn a SMILES *text* string into numbers a model can read.

A "featurizer" is the converter at the front of every pipeline:

    SMILES text  -->  featurizer  -->  fixed-length vector of numbers  -->  model

The whole point of the registry is that the cross-validated runner
(``cv_runner.py``) never has to change when we add a new representation: we just
``@register`` another function here and it becomes available to every plan.

Representations included (M1):
  - ``morgan``              circular substructure fingerprint (0/1 bits)
  - ``maccs``               166 standard substructure keys (0/1 bits)
  - ``rdkit_descriptors``   physicochemical properties (weight, logP, ...)
  - ``chemberta_embedding`` pretrained chemical-language-model vector (cached)

Leakage note: every featurizer here is **stateless** — it maps each molecule to
numbers independently, learning nothing from the dataset. That makes it safe to
compute once over all rows and then slice by fold. Any featurizer that *learns*
from data (a scaler, a target encoder) must instead be fit inside the training
fold only — in this project that lives on the model side (see ``models.py``,
where Ridge is wrapped in a per-fold ``StandardScaler``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog("rdApp.*")  # silence RDKit's noisy parse warnings

# name -> fn(smiles_list, params) -> np.ndarray of shape (n_molecules, n_features)
FEATURIZERS: dict[str, Callable[[list[str], dict], np.ndarray]] = {}

# featurizers expensive enough to be worth caching to disk (keyed by content)
CACHEABLE: set[str] = {"chemberta_embedding", "molformer_embedding", "chemeleon_embedding", "mordred_descriptors"}

# Which params actually change a cacheable featurizer's OUTPUT. The cache key uses
# only these — never model hyperparameters (alpha, n_estimators, scale, ...) that
# happen to share the plan's params dict, so the same embedding is computed once
# and reused across every model that consumes it.
CACHE_KEY_PARAMS: dict[str, list[str]] = {
    "chemberta_embedding": ["skill_ref", "pooling", "max_length"],
    "molformer_embedding": ["skill_ref", "pooling", "max_length"],
    "chemeleon_embedding": [],  # single fixed model — keyed on the SMILES set alone
    "mordred_descriptors": [],  # fixed 1613-descriptor block — keyed on the SMILES set alone
}


def register(name: str):
    """Decorator: add a featurizer function to the registry under ``name``."""
    def _wrap(fn: Callable[[list[str], dict], np.ndarray]):
        if name in FEATURIZERS:
            raise ValueError(f"Featurizer '{name}' already registered")
        FEATURIZERS[name] = fn
        return fn
    return _wrap


# --------------------------------------------------------------------------- #
# Fingerprints (stateless, cheap)
# --------------------------------------------------------------------------- #
@register("morgan")
def _morgan(smiles_list: list[str], params: dict) -> np.ndarray:
    """Morgan (ECFP-like) circular fingerprint as a dense 0/1 array."""
    radius = int(params.get("radius", 2))
    n_bits = int(params.get("n_bits", 2048))
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.vstack(rows)


@register("avalon")
def _avalon(smiles_list: list[str], params: dict) -> np.ndarray:
    """Avalon fingerprint — a different substructure hashing scheme than Morgan,
    so it makes different mistakes (useful diversity for the ensemble)."""
    from rdkit.Avalon import pyAvalonTools

    n_bits = int(params.get("n_bits", 1024))
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.vstack(rows)


@register("maccs")
def _maccs(smiles_list: list[str], params: dict) -> np.ndarray:
    """MACCS structural keys — 167-bit (index 0 is always 0) substructure presence."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(167, dtype=np.float32))
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.vstack(rows)


# --------------------------------------------------------------------------- #
# Physicochemical descriptors (stateless, cheap)
# --------------------------------------------------------------------------- #
# Cache the descriptor name/function list once; its order is deterministic, which
# keeps feature columns stable across runs (a hard requirement for caching/CV).
_DESCRIPTOR_FNS: list[tuple[str, Callable]] | None = None


def _descriptor_fns() -> list[tuple[str, Callable]]:
    global _DESCRIPTOR_FNS
    if _DESCRIPTOR_FNS is None:
        from rdkit.Chem import Descriptors

        _DESCRIPTOR_FNS = list(Descriptors.descList)
    return _DESCRIPTOR_FNS


@register("mordred_descriptors")
def _mordred_descriptors(smiles_list: list[str], params: dict) -> np.ndarray:
    """The full Mordred 2D descriptor block (~1613 features) — far richer than RDKit's.

    Used by the PXR leaders' GBDT branch. Mordred returns a 'Missing'/error object for
    descriptors it can't compute; we coerce those (and unparseable mols) to 0. Heavy
    enough to cache. Scaling, as elsewhere, is a learned per-fold transform (models.py).
    """
    import pandas as pd
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles_list]
    safe = [m if m is not None else Chem.MolFromSmiles("C") for m in mols]
    df = calc.pandas(safe, nproc=int(params.get("nproc", 1)), quiet=True)
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    for i, m in enumerate(mols):       # zero-out rows for unparseable molecules
        if m is None:
            arr[i] = 0.0
    return arr


@register("rdkit_descriptors")
def _rdkit_descriptors(smiles_list: list[str], params: dict) -> np.ndarray:
    """The full RDKit 2D descriptor block (molecular weight, logP, TPSA, ...).

    Unparseable molecules and any descriptor that errors/returns non-finite are
    set to 0. Scaling is intentionally **not** done here — it is a learned
    transform and must be fit per training fold (handled in ``models.py``).
    """
    fns = _descriptor_fns()
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(len(fns), dtype=np.float32))
            continue
        vals = np.empty(len(fns), dtype=np.float64)
        for j, (_, fn) in enumerate(fns):
            try:
                vals[j] = fn(mol)
            except Exception:
                vals[j] = 0.0
        rows.append(vals)
    arr = np.vstack(rows).astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Pretrained chemical-language-model embeddings (stateless but expensive -> cached)
# --------------------------------------------------------------------------- #
def _encode_transformer(smiles_list, tokenizer, model, pooling, max_length, batch_size):
    """Shared forward-pass + pooling loop for any HF encoder (ChemBERTa, MolFormer).

    ``mean`` pooling masks out padding before averaging the last hidden state;
    ``cls`` takes the first token. Forward pass only — no training, CPU/MPS fine.
    """
    import torch

    out: list[np.ndarray] = []
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
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            if pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                masked = hidden * attention_mask
                pooled = masked.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            out.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(out)


@register("chemberta_embedding")
def _chemberta_embedding(smiles_list: list[str], params: dict) -> np.ndarray:
    """Pooled hidden-state vector from a pretrained ChemBERTa encoder.

    Runs a forward pass only (no training), on CPU/MPS — no GPU required. The
    model id comes from ``params['skill_ref']`` (default DeepChem/ChemBERTa-77M-MTR).
    Heavy, so callers pass ``cache_dir`` via ``featurize()`` to memoize results.
    """
    import torch

    from src.agent.models import get_model_bundle

    # Belt-and-suspenders against the macOS OpenMP deadlock (see scripts/run_menu.py):
    # cap torch intra-op threads regardless of how this was launched.
    torch.set_num_threads(1)

    skill_ref = params.get("skill_ref", "DeepChem/ChemBERTa-77M-MTR")
    pooling = params.get("pooling", "mean")
    max_length = int(params.get("max_length", 256))
    batch_size = int(params.get("batch_size", 16))

    tokenizer, model, _device = get_model_bundle(skill_ref)
    return _encode_transformer(smiles_list, tokenizer, model, pooling, max_length, batch_size)


# MolFormer needs its own loader: it ships custom modeling code (trust_remote_code)
# and a custom tokenizer, so it can't reuse the ChemBERTa bundle in src/agent/models
# (which pins trust_remote_code=False). Cached in-process to avoid reloading weights.
_MOLFORMER_BUNDLE: dict[str, tuple] = {}


def _ensure_molformer_compat() -> None:
    """MolFormer's 2024-era custom code targets transformers ~4.x; we run 5.x.
    Two imports broke. Both are needed only by code paths we never call (ONNX
    export, attention-head pruning), so we patch them just enough to let the
    dynamic module import. No cache files are edited (a re-download would wipe them).

      1. ``from transformers.onnx import OnnxConfig`` — the whole ``onnx`` subpackage
         was removed. Inject a stub module exposing an empty ``OnnxConfig`` base.
      2. ``find_pruneable_heads_and_indices`` was dropped from
         ``transformers.pytorch_utils``. Re-add the standard implementation.
    """
    import sys
    import types

    try:
        from transformers.onnx import OnnxConfig  # noqa: F401
    except Exception:
        shim = types.ModuleType("transformers.onnx")
        shim.OnnxConfig = type("OnnxConfig", (), {})  # unused stub base class
        sys.modules["transformers.onnx"] = shim

    from transformers import pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        import torch

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def _get_molformer_bundle(skill_ref: str):
    _ensure_molformer_compat()
    from transformers import AutoModel, AutoTokenizer

    if skill_ref not in _MOLFORMER_BUNDLE:
        tokenizer = AutoTokenizer.from_pretrained(skill_ref, trust_remote_code=True)
        # deterministic_eval=True freezes MolFormer's stochastic rotary path so the
        # same SMILES always yields the same embedding (a hard requirement for caching/CV).
        model = AutoModel.from_pretrained(
            skill_ref, trust_remote_code=True, deterministic_eval=True
        )
        model.eval()
        _patch_get_head_mask(model)
        _fix_molformer_rotary(model)
        _MOLFORMER_BUNDLE[skill_ref] = (tokenizer, model)
    return _MOLFORMER_BUNDLE[skill_ref]


def _fix_molformer_rotary(model) -> None:
    """Recompute MolFormer's rotary cos/sin caches after loading.

    Those caches are ``persistent=False`` buffers built in ``__init__``. Under
    transformers 5.x ``from_pretrained`` (meta-device / lazy init), they get
    computed before real tensors exist and come out as NaN — and because they are
    non-persistent, the checkpoint never overwrites them. Left alone, every
    attention layer emits NaN. Rebuilding the cache on CPU restores finite values.
    """
    import torch

    n_fixed = 0
    for module in model.modules():
        if hasattr(module, "_set_cos_sin_cache") and hasattr(module, "max_seq_len_cached"):
            module._set_cos_sin_cache(
                seq_len=module.max_seq_len_cached, device="cpu", dtype=torch.float32
            )
            n_fixed += 1
    if n_fixed == 0:
        raise RuntimeError("MolFormer rotary cache not found to rebuild — model layout changed?")


def _patch_get_head_mask(model) -> None:
    """``PreTrainedModel.get_head_mask`` was removed in transformers 5.x, but
    MolFormer's forward pass still calls it. We never pass a head mask, so restore
    the standard no-op behaviour (head_mask=None -> one None per layer)."""
    import types

    if hasattr(model, "get_head_mask"):
        return

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError("MolFormer head pruning is not supported in this build")
        return [None] * num_hidden_layers

    model.get_head_mask = types.MethodType(get_head_mask, model)


@register("molformer_embedding")
def _molformer_embedding(smiles_list: list[str], params: dict) -> np.ndarray:
    """Pooled hidden-state vector from IBM's MoLFormer-XL encoder.

    A second, architecturally different chemical language model than ChemBERTa
    (rotary-attention transformer pretrained on ~1.1B molecules), so its embedding
    makes *different* mistakes — useful diversity for the ensemble. Forward pass
    only, on CPU/MPS; cached to disk via ``featurize()``.
    """
    import torch

    torch.set_num_threads(1)

    skill_ref = params.get("skill_ref", "ibm-research/MoLFormer-XL-both-10pct")
    pooling = params.get("pooling", "mean")
    max_length = int(params.get("max_length", 256))
    batch_size = int(params.get("batch_size", 16))

    tokenizer, model = _get_molformer_bundle(skill_ref)
    return _encode_transformer(smiles_list, tokenizer, model, pooling, max_length, batch_size)


# CheMeleon: a D-MPNN *graph* foundation model (pretrained on ~1M PubChem molecules
# to predict Mordred descriptors). Unlike ChemBERTa/MolFormer (1D SMILES transformers),
# it reads the molecular graph directly — the representation the PXR leaderboard leaders
# rely on (PRD §7.1). We use the pretrained encoder for embeddings only (no fine-tuning),
# so it runs on CPU/MPS and is cached. Weights live at ~/.chemprop/chemeleon_mp.pt.
_CHEMELEON_FP = None


def _get_chemeleon():
    global _CHEMELEON_FP
    if _CHEMELEON_FP is None:
        import sys
        from pathlib import Path

        vendor = str(Path(__file__).resolve().parent / "vendor")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from chemeleon_fingerprint import CheMeleonFingerprint

        _CHEMELEON_FP = CheMeleonFingerprint(device="cpu")
    return _CHEMELEON_FP


@register("chemeleon_embedding")
def _chemeleon_embedding(smiles_list: list[str], params: dict) -> np.ndarray:
    """Pooled D-MPNN graph embedding from the pretrained CheMeleon model (2048-dim).

    Forward pass only (no training), on CPU. Batched + cached to disk via ``featurize()``.
    """
    import torch

    torch.set_num_threads(1)
    fp = _get_chemeleon()
    batch_size = int(params.get("batch_size", 256))
    out: list[np.ndarray] = []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start : start + batch_size]
        out.append(np.asarray(fp(batch), dtype=np.float32))
    return np.vstack(out)


# --------------------------------------------------------------------------- #
# Public dispatch + content-addressed caching
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str | Path, name: str, params: dict, smiles_list: list[str]) -> Path:
    """Content-addressed cache file: keyed by featurizer name + the featurizer's
    *output-relevant* params + the exact SMILES set (and its order). Never keyed by
    fold or by model params, so caching cannot leak labels and is reused across models."""
    relevant_keys = CACHE_KEY_PARAMS.get(name)
    key_params = {k: params[k] for k in relevant_keys if k in params} if relevant_keys else params
    params_blob = json.dumps(key_params, sort_keys=True, default=str)
    smiles_blob = "\n".join(smiles_list)
    digest = hashlib.sha256(f"{params_blob}\n{smiles_blob}".encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{name}_{digest}.npy"


def featurize(
    name: str,
    smiles_list: list[str],
    params: dict | None = None,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Run featurizer ``name`` over ``smiles_list`` -> ``(n_molecules, n_features)``.

    When ``cache_dir`` is given and the featurizer is marked cacheable
    (e.g. embeddings), the result is memoized on disk by content hash.
    """
    params = params or {}

    # "fusion" is compositional: concatenate several featurizers into one richer
    # representation (e.g. physicochemical descriptors + a ChemBERTa embedding).
    # Each component is computed via this same dispatcher, so an embedding component
    # reuses its on-disk cache — no extra forward pass.
    if name == "fusion":
        components = params.get("components") or ["rdkit_descriptors", "chemberta_embedding"]
        mats = [featurize(c, smiles_list, params, cache_dir=cache_dir) for c in components]
        return np.hstack(mats)

    if name not in FEATURIZERS:
        raise KeyError(f"Unknown featurizer '{name}'. Registered: {sorted(FEATURIZERS)}")

    use_cache = cache_dir is not None and name in CACHEABLE
    if use_cache:
        path = _cache_path(cache_dir, name, params, smiles_list)
        if path.exists():
            return np.load(path)

    features = FEATURIZERS[name](smiles_list, params)

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
    return features


def available_featurizers() -> list[str]:
    return sorted(FEATURIZERS)
