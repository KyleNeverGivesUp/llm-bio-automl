"""
Bio Model Collector
1. Search biological foundation models on Hugging Face
2. Filter them with Claude scoring
3. Generate SKILL.md files for approved models
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic, AuthenticationError
from dotenv import load_dotenv

# Load environment variables from .env.
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────
TOP_N = 20
QUALITY_THRESHOLD = 7
MIN_DOWNLOADS = 1000
SEARCH_LIMIT = 10
OUTPUT_DIR = Path("./skills/models")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

BIO_SEARCH_TERMS = [
    # "biomedical",
    # "bioinformatics",
    # "protein",
    # "genomics",
    # "DNA",
    # "RNA",
    # "drug discovery",
    # "medical imaging",
    "small molecule property prediction",
    "SMILES regression",
    "molecular property prediction",
    "ligand activity prediction",
    "chemberta",
    "small molecule drug discovery",
    "molecular transformer",
    "chemistry foundation model",
]

client_kwargs = {}
if ANTHROPIC_API_KEY:
    client_kwargs["api_key"] = ANTHROPIC_API_KEY
if ANTHROPIC_BASE_URL:
    client_kwargs["base_url"] = ANTHROPIC_BASE_URL

client = Anthropic(**client_kwargs)


def anthropic_target() -> str:
    """Return the active Anthropic API target for debugging."""
    return ANTHROPIC_BASE_URL or "https://api.anthropic.com"


def hf_cli_command() -> list[str]:
    """Return the available Hugging Face CLI command."""
    hf = shutil.which("hf")
    if hf:
        return [hf]

    legacy = shutil.which("huggingface-cli")
    if legacy:
        return [legacy]

    raise RuntimeError("Hugging Face CLI not found. Run `uv sync` or install `huggingface_hub[cli]`.")


def run_hf_json(args: list[str], timeout: int = 30):
    """Run the Hugging Face CLI and parse JSON output."""
    command = hf_cli_command() + args + ["--format", "json"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(stderr or stdout or "hf CLI call failed")

    stdout = result.stdout.strip()
    if not stdout:
        return None

    return json.loads(stdout)


def response_text(resp) -> str:
    """Join Anthropic text blocks into a single string."""
    parts = []
    for block in resp.content:
        text = getattr(block, "text", "")
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def parse_json_response(raw: str) -> dict:
    """Extract JSON from model output, including fenced blocks or extra prose."""
    text = raw.strip()
    if not text:
        raise ValueError("Model returned empty text")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(text[start : end + 1])

    return json.loads(text)


def normalize_hf_model(row: dict) -> dict:
    """Map Hugging Face CLI fields into the script's internal structure."""
    ref = row.get("id", "")
    downloads = row.get("downloads", 0) or 0
    likes = row.get("likes", 0) or 0
    trending_score = row.get("trendingScore", 0) or 0
    spaces = row.get("spaces") or []
    eval_results = row.get("evalResults") or []
    card_data = row.get("cardData") or {}
    tags = row.get("tags") or []
    pipeline_tag = row.get("pipeline_tag") or ""

    if not isinstance(tags, list):
        tags = [str(tags)]

    # Hugging Face quality-related fields:
    # - downloads: usually recent download volume and the strongest adoption signal
    # - likes: lightweight community approval signal
    # - trendingScore: short-term momentum / current popularity signal
    # - spaces: public demos; useful as a quick usability signal
    # - evalResults: explicit benchmark results; often the strongest quality metadata
    # - cardData: structured model card metadata; indicates documentation quality
    #
    # Model name suffixes such as "-335M", "-1B", or "-7B" usually indicate
    # parameter count (roughly 335 million, 1 billion, 7 billion parameters).
    return {
        "ref": ref,
        "id": ref,
        "title": ref.split("/")[-1] if ref else "",
        "subtitle": pipeline_tag or f"likes={likes}",
        "downloadCount": int(downloads),
        "likes": int(likes),
        "trendingScore": int(trending_score),
        "spaces": spaces,
        "spaceCount": len(spaces),
        "evalResults": eval_results,
        "evalResultCount": len(eval_results),
        "hasCardData": bool(card_data),
        "cardDataKeys": sorted(card_data.keys()) if isinstance(card_data, dict) else [],
        "tags": tags,
        "pipeline_tag": pipeline_tag,
        "url": f"https://huggingface.co/{ref}" if ref else "",
    }


# ── Step 1: Search Hugging Face ────────────────────────────────────────────

def search_huggingface(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    """Search models with the Hugging Face CLI and return normalized rows."""
    try:
        rows = run_hf_json(
            [
                "models",
                "list",
                "--search",
                query,
                "--sort",
                "downloads",
                "--limit",
                str(limit),
                "--expand",
                "downloads,likes,trendingScore,tags,pipeline_tag,spaces,evalResults,cardData",
            ]
        )
        if not rows:
            return []
        if not isinstance(rows, list):
            raise ValueError("Hugging Face CLI did not return a model list")
        return [normalize_hf_model(row) for row in rows if row.get("id")]
    except RuntimeError as e:
        print(f"  [WARN] hf error for '{query}': {e}")
        return []
    except Exception as e:
        print(f"  [WARN] Search failed for '{query}': {e}")
        return []


def collect_candidates() -> list[dict]:
    """Search all bio keywords, deduplicate results, and return candidates."""
    seen = set()
    candidates = []

    for term in BIO_SEARCH_TERMS:
        print(f"  Search: '{term}'")
        results = search_huggingface(term)
        for model in results:
            ref = model.get("ref", "")
            if not ref or ref in seen:
                continue
            seen.add(ref)
            candidates.append(model)
        time.sleep(0.5)

    print(f"\n  Found {len(candidates)} candidate models\n")
    return candidates


# ── Step 2: Score And Filter ───────────────────────────────────────────────

def score_model(model: dict) -> dict:
    """Score a model with Claude and return the verdict."""
    ref = model.get("ref", "unknown")
    title = model.get("title", "")
    subtitle = model.get("subtitle", "")
    downloads = model.get("downloadCount", 0) or 0
    likes = model.get("likes", 0) or 0
    trending_score = model.get("trendingScore", 0) or 0
    tags = model.get("tags", [])
    url = model.get("url", "")
    space_count = model.get("spaceCount", 0) or 0
    eval_result_count = model.get("evalResultCount", 0) or 0
    card_data_keys = model.get("cardDataKeys", [])

    prompt = f"""Evaluate the following Hugging Face model and determine whether it is a high-quality biological foundation model.

Model ref: {ref}
Title: {title}
Subtitle: {subtitle}
Downloads: {downloads}
Likes: {likes}
Trending Score: {trending_score}
Tags: {tags}
URL: {url}
Space count: {space_count}
Eval results count: {eval_result_count}
Card data fields: {card_data_keys}

Evaluation criteria:
1. Is it a genuine biology or biomedical foundation model?
2. Does it have strong public quality signals? Prefer evalResults, Spaces, and a clear model card.
3. Does it have practical value for downstream bio-AI tasks such as protein prediction, genomics, or drug discovery?

Return only the following JSON and nothing else:
{{
  "bio_score": <integer from 1 to 10>,
  "is_bio_model": <true or false>,
  "domain": "<protein|genomics|drug_discovery|medical_imaging|other>",
  "reason": "<one sentence explaining why it is or is not a high-value biological model; do not repeat download counts>"
}}"""

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response_text(resp)
        result = parse_json_response(raw)
        bio_score = int(result.get("bio_score", result.get("score", 0)) or 0)
        usage_ok = downloads >= MIN_DOWNLOADS
        final_score = bio_score + (1 if usage_ok else 0)
        final_score = max(1, min(10, final_score))

        result["bio_score"] = bio_score
        result["downloads"] = downloads
        result["usage_ok"] = usage_ok
        result["score"] = final_score
        result["create_skill"] = (
            result.get("is_bio_model", False)
            and usage_ok
            and final_score >= QUALITY_THRESHOLD
        )
        result["ref"] = ref
        return result
    except AuthenticationError as e:
        raise RuntimeError(
            f"Anthropic authentication failed. Check ANTHROPIC_API_KEY. Current target: {anthropic_target()}"
        ) from e
    except Exception as e:
        raw_preview = locals().get("raw", "")
        if raw_preview:
            raw_preview = raw_preview.replace("\n", "\\n")[:240]
            print(f"  [WARN] Raw response for {ref}: {raw_preview}")
        print(f"  [WARN] Scoring failed for {ref}: {e}")
        return {"ref": ref, "score": 0, "create_skill": False, "reason": str(e)}


def filter_models(candidates: list[dict]) -> list[tuple]:
    """Score all candidates and return approved (model, verdict) tuples."""
    approved = []

    for i, model in enumerate(candidates):
        ref = model.get("ref", "unknown")
        print(f"  [{i+1}/{len(candidates)}] Evaluate: {ref}")

        verdict = score_model(model)
        score = verdict.get("score", 0)
        reason = verdict.get("reason", "")
        downloads = verdict.get("downloads", 0)
        usage_note = f"downloads={downloads}"
        if not verdict.get("usage_ok", False):
            usage_note += f" < {MIN_DOWNLOADS}"

        if verdict.get("create_skill"):
            print(f"    ✓ {score}/10 | {verdict.get('domain')} | {usage_note} | {reason}")
            approved.append((model, verdict))
        else:
            print(f"    ✗ {score}/10 | {usage_note} | {reason}")

        if len(approved) >= TOP_N:
            print(f"\n  Reached TOP_N={TOP_N}, stopping early")
            break

        time.sleep(0.3)

    print(f"\n  Approved: {len(approved)} models\n")
    return approved


# ── Step 3: Generate SKILL.md ──────────────────────────────────────────────

def generate_skill_md(model: dict, verdict: dict) -> str:
    """Generate SKILL.md content for one approved model."""
    ref = model.get("ref", "unknown")
    title = model.get("title", "")
    subtitle = model.get("subtitle", "")
    domain = verdict.get("domain", "other")
    tags = model.get("tags", [])
    model_url = model.get("url", f"https://huggingface.co/{ref}")

    prompt = f"""Generate a SKILL.md file for the following biological foundation model on Hugging Face.

Model ref: {ref}
Title: {title}
Subtitle: {subtitle}
Domain: {domain}
Tags: {tags}
Hugging Face URL: {model_url}

Follow the exact format below and do not add any extra text:

---
name: <lowercase slug separated by hyphens>
description: <1-2 sentences explaining what the model does and when to use it, with specific biological domain and task details>
---

# {title}

## Overview
Explain what the model does and which biological problem it solves.

## When to Use
List the specific biological tasks this model is best suited for.

## How to Use
Explain how to load and run the model, including a minimal Python example:

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="{ref}")
# Minimal working example
```

## Input Format
Describe the expected input format in concrete terms.

## Output Format
Describe the output format and how to interpret it.

## Example
A short, concrete usage example.

## Notes
Important limitations, dependencies, or caveats."""

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response_text(resp)
    except AuthenticationError as e:
        raise RuntimeError(
            f"Anthropic authentication failed. Check ANTHROPIC_API_KEY. Current target: {anthropic_target()}"
        ) from e
    except Exception as e:
        print(f"  [WARN] Generation failed for {ref}: {e}")
        return None


def save_skill(ref: str, content: str, domain: str) -> Path:
    """Save SKILL.md to ./skills/models/<domain>/<slug>/SKILL.md."""
    slug = ref.replace("/", "--")
    skill_dir = OUTPUT_DIR / domain / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


# ── Main Flow ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Bio Model Collector")
    print("=" * 60)
    print(f"Anthropic model: {MODEL}")
    print(f"Anthropic target: {anthropic_target()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Search Hugging Face bio models...")
    try:
        candidates = collect_candidates()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    if not candidates:
        print("No candidate models found. Check the Hugging Face CLI, network, or search terms.")
        return

    print("[2/3] Score and filter...")
    try:
        approved = filter_models(candidates)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return
    if not approved:
        print("No models passed the filter.")
        return

    print("[3/3] Generate SKILL.md...")
    created = []
    for model, verdict in approved:
        ref = model.get("ref", "unknown")
        domain = verdict.get("domain", "other")
        print(f"  Generate: {ref}")

        try:
            content = generate_skill_md(model, verdict)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return
        if content:
            path = save_skill(ref, content, domain)
            created.append({"ref": ref, "domain": domain, "path": str(path)})
            print(f"    ✓ Saved to {path}")
        else:
            print("    ✗ Generation failed")

        time.sleep(0.3)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(created),
        "skills": created,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(f"Done. Generated {len(created)} skills -> {OUTPUT_DIR}/")
    print(f"Manifest -> {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
