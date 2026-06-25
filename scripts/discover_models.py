"""Discover candidate molecular models for the pipeline — live HF search + curated frontier.

Replaces the static 7-ChemBERTa manifest (skills/models/manifest.json) that the M3 retrieval
stage was locked to. Produces skills/models/candidates_live.json: a family-classified, ranked
candidate pool the selector can draw from — and critically, prefer DECORRELATED additions.

Two sources, because neither alone is enough:
  1. Live HF Hub search (src/agent/hf_retrieval.py) — finds what's published on HF.
  2. Curated FRONTIER registry below — the strong models that live on GitHub/Zenodo and so are
     invisible to an HF-only search. This is the honest lesson from manually integrating them:
     the SOTA molecular encoders (CheMeleon, Uni-Mol, GROVER, MolCLR, MolE) are NOT on HF.

Each frontier entry records source + family + integration path + whether we've validated it,
so the pipeline (or a human) knows decorrelation potential and install cost up front.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.hf_retrieval import discover_models

# Strong molecular encoders that an HF search misses (GitHub/Zenodo weights). Curated from this
# project's research + manual integration. `validated` = we actually measured it on the Set-1 judge.
FRONTIER = [
    {"ref": "CheMeleon", "family": "graph", "source": "chemprop --from-foundation CheMeleon",
     "integration": "pip:chemprop", "validated": "mt5 single 0.5904 (fine-tuned)", "decorrelation": "base"},
    {"ref": "Uni-Mol (unimolv1)", "family": "3d", "source": "pip:unimol_tools (Zenodo weights)",
     "integration": "pip:unimol_tools", "validated": "single 0.6248; +mt5 stack 0.5706", "decorrelation": "high vs graph"},
    {"ref": "dptech/Uni-Mol2", "family": "3d", "source": "HF dptech/Uni-Mol2 (pip:unimol_tools)",
     "integration": "pip:unimol_tools", "validated": None, "decorrelation": "redundant with Uni-Mol"},
    {"ref": "ibm-research/biomed.sm.mv-te-84m", "family": "multiview", "source": "HF (pip:bmfm_sm)",
     "integration": "pip:git biomed-multi-view (needs fast_transformers — build pain)", "validated": None,
     "decorrelation": "high (adds 2D-image view)"},
    {"ref": "tencent-ailab/grover", "family": "graph-transformer", "source": "GitHub + GDrive weights",
     "integration": "py3.6 env; `main.py fingerprint` CLI", "validated": None, "decorrelation": "med (graph, diff arch)"},
    {"ref": "yuyangw/MolCLR", "family": "graph-contrastive", "source": "GitHub (weights in repo)",
     "integration": "torch1.7 env; no extract code (write from finetune.py)", "validated": None, "decorrelation": "med"},
    {"ref": "recursionpharma/MolE", "family": "graph-deberta", "source": "GitHub (weights WITHHELD — code only)",
     "integration": "BLOCKED — no public pretrained checkpoint", "validated": None, "decorrelation": "high (unobtainable)"},
    {"ref": "rolayoalarcon/MolE", "family": "graph-gin", "source": "GitHub + Zenodo 10803099 (weights public)",
     "integration": "torch-geometric env", "validated": None, "decorrelation": "med"},
    {"ref": "DeepChem/ChemBERTa-*", "family": "smiles", "source": "HF (the old static manifest)",
     "integration": "pip:transformers", "validated": "frozen 0.74-0.82 (WEAK)", "decorrelation": "low + weak"},
]


def main() -> None:
    out_dir = Path("skills/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== live HF Hub search ===")
    hf = discover_models(top_k=20)
    for c in hf[:12]:
        print(f"  [{c.family:<10}] {c.model_id:<52} score={c.score:5.2f} dl={c.downloads}")

    print("\n=== curated frontier (GitHub/Zenodo — invisible to HF search) ===")
    for f in FRONTIER:
        flag = "✓" if f["validated"] else " "
        print(f"  {flag} [{f['family']:<16}] {f['ref']:<34} {f['decorrelation']}")

    payload = {
        "note": "Live HF search + curated frontier. Replaces static 7-ChemBERTa manifest.json.",
        "hf_live": [c.to_dict() for c in hf],
        "frontier": FRONTIER,
        "families_seen": sorted({c.family for c in hf} | {f["family"] for f in FRONTIER}),
    }
    out = out_dir / "candidates_live.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    old = json.loads((out_dir / "manifest.json").read_text())["skills"] if (out_dir / "manifest.json").exists() else []
    print(f"\nBEFORE: static manifest = {len(old)} models, all family=smiles (ChemBERTa)")
    print(f"AFTER : {len(hf)} live HF + {len(FRONTIER)} frontier, families={payload['families_seen']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
