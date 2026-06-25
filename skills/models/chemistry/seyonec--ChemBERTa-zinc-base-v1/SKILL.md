---
name: chemberta-zinc-base-v1
description: A RoBERTa-based language model pre-trained on chemical SMILES strings for molecular property prediction and chemical compound understanding. Use this model for fill-mask tasks in drug discovery to predict missing tokens in chemical sequences and understand molecular representations.
---

# ChemBERTa-zinc-base-v1

## Overview
ChemBERTa-zinc-base-v1 is a transformer-based foundation model specifically trained on chemical compound data from the ZINC database. It learns to understand chemical structures represented as SMILES (Simplified Molecular Input Line Entry System) strings, enabling downstream applications in drug discovery and molecular analysis. The model uses a RoBERTa architecture adapted for chemical language, allowing it to capture relationships between molecular substructures and chemical properties.

## When to Use
- **Molecular property prediction**: Transfer learning for predicting bioactivity, toxicity, or other molecular properties
- **Fill-mask tasks**: Predicting masked tokens in chemical SMILES sequences
- **Chemical similarity**: Computing embeddings for chemical compounds to identify similar molecules
- **Drug discovery workflows**: Identifying promising candidate compounds or understanding chemical structure-activity relationships
- **Molecular representation learning**: Feature extraction for chemical compounds in downstream ML pipelines

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: Fill-mask prediction
smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"  # Ibuprofen
masked_smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)[MASK]"

inputs = tokenizer(masked_smiles, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get predictions for masked token
predictions = torch.softmax(outputs.logits[0, inputs.input_ids[0] == tokenizer.mask_token_id], dim=-1)
top_tokens = torch.topk(predictions, 5)
```

## Input Format
- **SMILES strings**: Chemical compounds represented as SMILES notation (text format)
- **Tokenization**: Uses a custom tokenizer trained on chemical vocabulary; automatically handles SMILES substructure tokenization
- **Masking**: Tokens can be masked with `[MASK]` token for fill-mask tasks
- **Sequence length**: Supports variable-length sequences up to the model's context window

## Output Format
- **Hidden states**: Dense vector representations (768 dimensions) for each token and the full sequence
- **Logits**: Unnormalized prediction scores for the masked language modeling task
- **Embeddings**: Token and sequence embeddings useful for downstream tasks (classification, regression, clustering)

## Example
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Predict missing functional group
smiles_masked = "c1ccc(cc1)C(=O)[MASK]"  # Benzene with masked carbonyl group
inputs = tokenizer(smiles_masked, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get top 5 predictions for masked position
mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
logits = outputs.logits[0, mask_token_index]
top_tokens = torch.topk(logits, 5)

for token_id in top_tokens.indices[0]:
    print(tokenizer.decode([token_id]))
```

## Notes
- **Chemical vocabulary**: Trained specifically on ZINC database; best performance on drug-like molecules
- **SMILES format**: Input must be valid SMILES strings; use RDKit or similar tools for validation
- **Fine-tuning**: Model is designed for transfer learning; fine-tune on task-specific labeled data for optimal performance
- **Deployment**: Compatible with Hugging Face Inference Endpoints and Azure deployments
- **Dependencies**: Requires `transformers`, `torch` (or `jax`), and optionally `RDKit` for SMILES manipulation
- **Computational cost**: Base model is relatively lightweight; inference is fast on CPU for single molecules