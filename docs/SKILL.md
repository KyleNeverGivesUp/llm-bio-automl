---
name: bio-model-collector
description: Use this skill to automatically discover, evaluate, and create skills for high-quality biological foundation models from Kaggle. Trigger this skill whenever the user wants to build a library of bio model skills, find the best biological foundation models on Kaggle, or prepare skills for a downstream biological AI model-building agent. Also trigger when the user says things like "collect bio models", "find protein models on Kaggle", or "generate skills for biological foundation models".
---

# Bio Model Collector

Your goal is to search Kaggle for high-quality biological foundation models, filter them by quality, and generate a reusable SKILL.md for each approved model. This builds a library of skills that downstream agents can use to automatically develop biological AI models.

Think of this as a pipeline: search → filter → generate. Each stage feeds into the next. The end result is a folder of SKILL.md files, one per model, ready to be used by other agents.

## Step 1: Search Kaggle for Biological Foundation Models

Write and run a Python script to search Kaggle using the `kaggle` CLI across these biological domains:

- protein structure / protein function prediction
- genomics / DNA sequence modeling
- RNA structure and function
- drug discovery / molecular property prediction
- medical imaging / pathology
- bioinformatics / gene expression

For each domain, fetch models sorted by download count. Deduplicate by model ref across domains. Aim to collect at least 50 candidates before filtering.

Example kaggle CLI command:
```bash
kaggle models list --search "protein structure" --sort-by downloadCount --page-size 20 --output json
```

Collect the following fields for each model: `ref`, `title`, `subtitle`, `downloadCount`, `tags`.

## Step 2: Filter for Quality

For each candidate model, evaluate it against these criteria:

1. **Biological relevance** — Is this genuinely a biological or biomedical foundation model, not just a generic model that happens to appear in results?
2. **Adoption** — Does it have meaningful downloads or usage?
3. **Downstream utility** — Would it be useful for biological AI tasks like protein function prediction, genomics analysis, or drug discovery?

Score each model 1–10. Keep only models scoring 6 or above. Stop once you have 20 approved models — that's the target library size.

When in doubt about a model's relevance, lean toward excluding it. A smaller set of genuinely useful skills is better than a large set of noisy ones.

## Step 3: Generate a SKILL.md for Each Approved Model

For each approved model, generate a SKILL.md that tells a future Claude agent exactly how to use this model. The skill should be practical and specific — not a generic description, but a real how-to guide.

Use this exact structure:

```
---
name: <short-lowercase-slug>
description: <1-2 sentences: what biological tasks this model handles and when a downstream agent should use it. Be specific about the domain — e.g., "predicts protein 3D structure from amino acid sequences" not just "a protein model">
---

# <Model Title>

## Overview
What this model does, what biological problem it solves, and who created it.

## When to Use
List the specific biological tasks where this model excels. Be concrete:
- "Predicting whether a protein sequence will fold into a specific structure"
- "Classifying genomic variants as pathogenic or benign"

## How to Use
Step-by-step instructions for loading and running this model.
Include a minimal working Python example using kagglehub or transformers.

```python
import kagglehub
path = kagglehub.model_download("<model-ref>")
# ... minimal working example
```

## Input Format
What the model expects as input. Be specific about data types, shapes, formats (e.g., "amino acid sequence as a string of single-letter codes, max 1024 residues").

## Output Format
What the model returns and how to interpret it (e.g., "confidence scores per residue, higher = more structured").

## Example
A short concrete use case showing real input and expected output.

## Notes
Key limitations, required dependencies, known gotchas, or performance characteristics worth knowing.
```

## Step 4: Save Everything

Save each skill to:
```
./skills/<domain>/<model-slug>/SKILL.md
```

Where `<domain>` is one of: `protein`, `genomics`, `drug_discovery`, `medical_imaging`, `other`.

After all skills are saved, write a manifest:
```
./skills/manifest.json
```

The manifest should list each skill with its path, domain, Kaggle ref, and quality score.

## What Good Looks Like

A good run of this skill produces 20 SKILL.md files where each one:
- Has a description specific enough that a downstream agent can decide whether to use it without reading the full file
- Has a working code example that actually loads the model
- Clearly states input/output formats in biological terms

A bad run produces skills that are vague, generic, or copy-paste the Kaggle description without adding useful how-to information.

## Requirements

- `~/.kaggle/kaggle.json` with valid Kaggle credentials
- `ANTHROPIC_API_KEY` environment variable set
- Python packages: `anthropic`, `kaggle`, `kagglehub`

```bash
pip install anthropic kaggle kagglehub
```