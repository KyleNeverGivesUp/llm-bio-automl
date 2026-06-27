"""Persistent, self-growing local model library — `skills/models/registry.json`.

Local-first retrieval: search here before going online. When an online (HuggingFace) search finds
new models, write them back here so the NEXT similar task is served locally — the library grows
and "learns" over time. This is the cache that turns retrieval from "always hit the network" into
"local-first, online only on a miss".

Pure stdlib; anchored to the repo root so it works regardless of CWD.
"""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "skills" / "models" / "registry.json"

# Seed entries — what we already know is good. has_template=True means we can FINE-TUNE it
# (a verified training template exists); otherwise it can only be used FROZEN (embeddings).
_SEED = [
    {"ref": "chemeleon", "family": "graph", "has_template": True, "source": "curated",
     "note": "graph D-MPNN foundation; fine-tuned single ~0.59 (best member)"},
    {"ref": "unimol", "family": "3d", "has_template": True, "source": "curated",
     "note": "3D geometry foundation; fine-tuned single ~0.62; decorrelated from graph"},
    {"ref": "DeepChem/ChemBERTa-77M-MTR", "family": "smiles", "has_template": False, "source": "curated",
     "note": "SMILES transformer; frozen embeddings only; weak on this task (~0.74)"},
]


def load() -> list[dict]:
    """Load the registry, seeding it on first use."""
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(_SEED, indent=2), encoding="utf-8")
    return list(_SEED)


def search(families: list[str] | None) -> list[dict]:
    """Return local models whose family is in `families` (all if `families` is empty)."""
    reg = load()
    if not families:
        return reg
    fams = {f.lower() for f in families}
    return [m for m in reg if str(m.get("family", "")).lower() in fams]


def add(candidates: list[dict]) -> int:
    """Write-back: append new (by ref) candidates to the registry. Returns how many were added."""
    reg = load()
    have = {m.get("ref") for m in reg}
    new = [c for c in candidates if c.get("ref") and c["ref"] not in have]
    if new:
        reg.extend(new)
        REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return len(new)
