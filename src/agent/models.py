"""Centralized model loading and in-process caching utilities."""

from __future__ import annotations

import os
from typing import Any

# Force local-cache-only loading for Hugging Face assets in this process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_TOKENIZER_CACHE: dict[str, Any] = {}
_ENCODER_CACHE: dict[str, Any] = {}
_DEVICE: str | None = None


def get_torch_device() -> str:
    return "cpu"


def get_tokenizer(skill_ref: str):
    if skill_ref not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[skill_ref] = AutoTokenizer.from_pretrained(
            skill_ref,
            use_fast=False,
            local_files_only=True,
            trust_remote_code=False,
        )
    return _TOKENIZER_CACHE[skill_ref]


def get_encoder(skill_ref: str):
    if skill_ref not in _ENCODER_CACHE:
        from transformers import AutoModel
        import torch

        model = AutoModel.from_pretrained(
            skill_ref,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        model.eval()
        _ENCODER_CACHE[skill_ref] = model
    return _ENCODER_CACHE[skill_ref]


def get_model_bundle(skill_ref: str):
    return get_tokenizer(skill_ref), get_encoder(skill_ref), get_torch_device()
