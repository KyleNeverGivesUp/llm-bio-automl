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
GITHUB_API = "https://api.github.com/search/repositories"
ZENODO_API = "https://zenodo.org/api/records"
ARXIV_API = "https://export.arxiv.org/api/query"


def arxiv_recent(query: str, n: int = 8) -> list[dict]:
    """Keyless arXiv search, newest-first → recent papers (title + abstract). Lets the pipeline read
    CURRENT literature and learn about post-cutoff SOTA models the LLM's static knowledge can't name."""
    import re
    params = {"search_query": query, "max_results": n, "sortBy": "submittedDate", "sortOrder": "descending"}
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=25) as r:
            xml = r.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        print(f"[hf_retrieval] arxiv {query!r} failed: {exc}")
        return []
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    out = []
    for e in entries:
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        if t:
            out.append({"title": " ".join(t.group(1).split()),
                        "abstract": " ".join(s.group(1).split())[:600] if s else ""})
    return out

# The strong molecular foundation models (CheMeleon, Uni-Mol, MolE) are NOT on the HF Hub — they
# live on GitHub (code) + Zenodo (weights). HF keyword search only surfaces weak/auxiliary models.
# So we search all three sources; Zenodo is noisy, so we filter to model/software with chem keywords.
_CHEM_KW = ("mol", "chem", "smiles", "drug", "compound", "graph", "ligand", "admet", "qsar")  # broad (zenodo titles)
# Stricter set for GitHub: bare "mol"/"graph" let in software noise (moleculer, ansible/molecule,
# cashapp/molecule). Require a real chemistry signal instead. Also reused to tighten HF.
_CHEM_KW_GH = ("molecular", "smiles", "chem", "drug discovery", "ligand", "qsar", "admet",
               "cheminform", "rdkit", "chembl", "bioactiv", "compound", "molecular property")
# General chat/LLM markers — these get mis-named "drug-discovery-*" but are not molecular models.
_LLM_CUES = ("qwen", "llama", "gpt-", "mistral", "gemma", "gguf", "-7b", "-14b", "-20b", "-70b")

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


def _github_search(query: str, limit: int) -> list[Candidate]:
    """Search GitHub repos — this is where strong models (CheMeleon, Uni-Mol) live as code.

    Unauthenticated GitHub search is 10 req/min (easily exhausted). Set GITHUB_TOKEN in the env for
    a 30/min authenticated limit — strongly recommended for repeated runs.
    """
    import math
    import os
    url = GITHUB_API + "?" + urllib.parse.urlencode({"q": query, "sort": "stars", "per_page": limit})
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "llm-bio-automl"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            items = json.load(r).get("items", [])
    except Exception as exc:  # rate limit (60/hr unauth) or network — skip this source for this query
        print(f"[hf_retrieval] github {query!r} failed: {exc}")
        return []
    out = []
    for it in items:
        name = it.get("full_name", "")
        desc = it.get("description") or ""
        if not any(k in (name + " " + desc).lower() for k in _CHEM_KW_GH):
            continue  # off-topic repo (strict filter drops molecule-named non-chem software)
        stars = int(it.get("stargazers_count", 0) or 0)
        out.append(Candidate(model_id=name, downloads=stars, library="github",
                             family=_classify_family(name, [desc]),
                             score=2.0 + math.log10(stars + 10)))  # github bonus + star popularity
    return out


def _zenodo_search(query: str, limit: int) -> list[Candidate]:
    """Search Zenodo — where the weights live (CheMeleon, Uni-Mol, MolE). Noisy → filter hard."""
    url = ZENODO_API + "?" + urllib.parse.urlencode({"q": query, "size": limit})
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            hits = json.load(r).get("hits", {}).get("hits", [])
    except Exception as exc:
        print(f"[hf_retrieval] zenodo {query!r} failed: {exc}")
        return []
    out = []
    for h in hits:
        md = h.get("metadata", {})
        rtype = md.get("resource_type", {}).get("type", "")
        title = md.get("title", "")
        if rtype not in ("model", "software"):           # drop publications/images/datasets noise
            continue
        if not any(k in title.lower() for k in _CHEM_KW):
            continue
        out.append(Candidate(model_id=title[:70], downloads=0, library="zenodo", tags=[rtype],
                             family=_classify_family(title, []), score=1.8))
    return out


def discover_models(
    queries: list[str] | None = None,
    limit_each: int = 25,
    top_k: int = 20,
    min_score: float = 0.0,
    sources: tuple[str, ...] = ("hf", "github", "zenodo"),
    per_family: int = 4,
) -> list[Candidate]:
    """Multi-source live search (HF + GitHub + Zenodo); deduped, family-tagged, ranked candidates.

    HF surfaces mostly weak models; GitHub/Zenodo are where the strong foundation models live
    (CheMeleon, Uni-Mol, MolE). github/zenodo only queried for the first few queries to respect
    GitHub's unauthenticated rate limit.
    """
    queries = queries or DEFAULT_QUERIES
    by_id: dict[str, Candidate] = {}
    for qi, q in enumerate(queries):
        if "hf" in sources:
            for m in _hf_get(q, limit_each):
                mid = m.get("id")
                if not mid or mid in by_id:
                    continue
                tags = m.get("tags", []) or []
                tset = set(tags)
                if tset & _BAD_TAGS and not (tset & _GOOD_TAGS):
                    continue  # pure generation/chat junk
                hay = (mid + " " + " ".join(tags)).lower()
                if any(c in hay for c in _LLM_CUES):
                    continue  # general chat LLM mis-named for "drug discovery" (qwen/gpt/-14b/gguf...)
                if not (tset & _GOOD_TAGS or any(k in hay for k in _CHEM_KW_GH)):
                    continue  # no real chemistry signal -> skip (was: accept everything but chat junk)
                by_id[mid] = Candidate(
                    model_id=mid, downloads=int(m.get("downloads", 0) or 0),
                    likes=int(m.get("likes", 0) or 0), tags=tags,
                    library=m.get("library_name", "") or "",
                    family=_classify_family(mid, tags),
                    score=_score(int(m.get("downloads", 0) or 0), int(m.get("likes", 0) or 0), tset),
                )
        # Zenodo has NO rate limit and is where strong weights live (Uni-Mol, CheMeleon), so search it
        # for EVERY query — otherwise a model-name query that isn't in the first few (e.g. "Uni-Mol")
        # never gets looked up there. GitHub IS rate-limited (10/min unauth) → only the first few queries.
        extra = []
        if "zenodo" in sources:
            extra += _zenodo_search(q, 5)
        if "github" in sources and qi < 5:
            extra += _github_search(q, 5)
        for c in extra:
            if c.model_id not in by_id:
                by_id[c.model_id] = c
    # Boost LLM-NAMED matches: the planner deliberately named these models (e.g. "Uni-Mol"), so a
    # candidate whose id contains a query name should rank UP, not be buried by popularity or Zenodo's
    # low fixed score. This is how an LLM-named backbone reliably surfaces instead of being cut by top_k.
    def _norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())
    qnorms = [n for n in (_norm(q) for q in queries) if len(n) >= 4]   # skip tiny/ambiguous terms
    for c in by_id.values():
        cid = _norm(c.model_id)
        if any(qn in cid for qn in qnorms):
            c.score += 6.0                                             # named by the LLM -> honor it

    ranked = [c for c in sorted(by_id.values(), key=lambda c: -c.score) if c.score >= min_score]
    if not per_family:
        return ranked[:top_k]
    # Per-family TOP-N: take the top `per_family` of EACH family independently (NOT a global top_k
    # cut). This GUARANTEES every represented family contributes its best `per_family` — a strong but
    # globally low-ranked family (e.g. 3D / Uni-Mol, only on Zenodo with a low fixed score) can never
    # be squeezed to zero by a popular family (e.g. many CheMeleon variants). No global top_k squeeze.
    import collections
    by_fam: dict[str, list] = collections.defaultdict(list)
    for c in ranked:
        by_fam[c.family].append(c)                       # each family's list is already score-sorted
    result = []
    for cands in by_fam.values():
        result.extend(cands[:per_family])                # top `per_family` of THIS family
    result.sort(key=lambda c: -c.score)                  # present best-first
    return result


if __name__ == "__main__":
    for c in discover_models(top_k=15):
        print(f"  [{c.family:<9}] {c.model_id:<55} score={c.score:5.2f} dl={c.downloads}")
