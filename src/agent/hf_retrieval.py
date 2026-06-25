"""Live model retrieval — query the Hugging Face Hub for candidate molecular models.

This is the missing piece the professor flagged: the M3 retrieval stage reads a STATIC
hand-built manifest (``skills/models/manifest.json``, 7 ChemBERTa variants) and can only
PICK from it. It can never discover CheMeleon / Uni-Mol / IBM-multi-view — the strong,
decorrelated models that actually moved our score — because they were never in the list.

This module replaces that static lookup with a LIVE Hub search: aggregate several queries,
drop the LLM-chat / generation junk, classify each hit into a representation *family*
(graph / 3d / smiles / multiview / descriptor) so the downstream selector can prefer
DECORRELATED additions, and rank by relevance + popularity. Pure stdlib (urllib) so it runs
headless in the pipeline — no MCP, no extra deps.

Used by ``scripts/discover_models.py`` to refresh the candidate manifest.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

HF_API = "https://huggingface.co/api/models"

# Default sweep — several angles, because HF keyword search is narrow and one query misses most.
DEFAULT_QUERIES = [
    "molecule", "molecular property", "SMILES", "drug discovery",
    "molecular representation", "chemistry property prediction", "compound",
]

# Tags that mark a model as USABLE for property prediction vs. LLM/chat/generation noise.
_GOOD_TAGS = {
    "molecular-property-prediction", "chemistry", "drug-discovery", "molecules",
    "small-molecules", "binding-affinity-prediction", "moleculenet", "cheminformatics",
}
_BAD_TAGS = {  # generation / chat / vision — not feature extractors for regression
    "text-generation", "conversational", "gguf", "text2text-generation",
    "image-text-to-text", "question-answering", "fill-mask",
}

# Representation family by tag/name cue → lets the selector pick something DECORRELATED
# from what's already in the ensemble (graph≠3d≠smiles≠multiview).
_FAMILY_CUES = {
    "3d": ["uni-mol", "unimol", "3d", "conformer", "geometry", "geometric", "schnet", "gem"],
    "multiview": ["multi-view", "multiview", "multimodal", "multi-modal"],
    "graph": ["gnn", "graph", "mpnn", "d-mpnn", "gin", "gcn", "grover", "graphormer", "chemeleon", "mole-bert"],
    "smiles": ["smiles", "molformer", "chemberta", "selformer", "bert", "roberta", "gpt", "transformer-language"],
    "descriptor": ["descriptor", "fingerprint", "mordred", "ecfp", "morgan"],
}


@dataclass
class Candidate:
    model_id: str
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    library: str = ""
    family: str = "unknown"
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ref": self.model_id, "downloads": self.downloads, "likes": self.likes,
            "library": self.library, "family": self.family, "score": round(self.score, 3),
            "tags": [t for t in self.tags if "license" not in t][:12],
        }


def _hf_get(query: str, limit: int) -> list[dict]:
    params = {"search": query, "sort": "downloads", "direction": -1, "limit": limit, "full": "true"}
    url = HF_API + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            return json.load(r)
    except Exception as exc:  # network hiccup on one query shouldn't kill the sweep
        print(f"[hf_retrieval] query {query!r} failed: {exc}")
        return []


def _classify_family(model_id: str, tags: list[str]) -> str:
    hay = (model_id + " " + " ".join(tags)).lower()
    for fam, cues in _FAMILY_CUES.items():
        if any(c in hay for c in cues):
            return fam
    return "unknown"


def _score(downloads: int, likes: int, tags: set[str]) -> float:
    import math
    good = len(tags & _GOOD_TAGS)
    bad = len(tags & _BAD_TAGS)
    pop = math.log10(downloads + 10)  # popularity, log-scaled
    return 2.0 * good - 3.0 * bad + pop + 0.2 * math.log10(likes + 1)


def discover_models(
    queries: list[str] | None = None,
    limit_each: int = 25,
    top_k: int = 20,
    min_score: float = 0.0,
) -> list[Candidate]:
    """Live-search the Hub across several queries; return deduped, family-tagged, ranked candidates."""
    queries = queries or DEFAULT_QUERIES
    by_id: dict[str, Candidate] = {}
    for q in queries:
        for m in _hf_get(q, limit_each):
            mid = m.get("id")
            if not mid or mid in by_id:
                continue
            tags = m.get("tags", []) or []
            tset = set(tags)
            if tset & _BAD_TAGS and not (tset & _GOOD_TAGS):
                continue  # pure generation/chat junk
            by_id[mid] = Candidate(
                model_id=mid, downloads=int(m.get("downloads", 0) or 0),
                likes=int(m.get("likes", 0) or 0), tags=tags,
                library=m.get("library_name", "") or "",
                family=_classify_family(mid, tags),
                score=_score(int(m.get("downloads", 0) or 0), int(m.get("likes", 0) or 0), tset),
            )
    ranked = sorted(by_id.values(), key=lambda c: -c.score)
    return [c for c in ranked if c.score >= min_score][:top_k]


if __name__ == "__main__":
    for c in discover_models(top_k=15):
        print(f"  [{c.family:<9}] {c.model_id:<55} score={c.score:5.2f} dl={c.downloads}")
