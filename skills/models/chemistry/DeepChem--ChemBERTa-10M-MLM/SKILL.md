---
name: chemberta-10m-mlm
description: A RoBERTa-based masked language model pretrained on 10M chemical SMILES strings for predicting masked tokens in molecular structures. Use this model for molecular representation learning, chemical language understanding, and as a foundation for drug discovery tasks.
---

# ChemBERTa-10M-MLM

## Overview
ChemBERTa-10M-MLM is a masked language model based on RoBERTa architecture, pretrained on 10 million chemical SMILES (Simplified Molecular Input Line Entry System) strings. It learns chemical language patterns and molecular representations by predicting masked tokens in SMILES sequences. This model serves as a foundational representation for downstream drug discovery and molecular property prediction tasks.

## When to Use
- Fine-tuning for molecular property prediction (toxicity, solubility, bioactivity)
- Chemical similarity and clustering tasks
- Drug-likeness assessment and lead optimization
- Transfer learning for small-data chemical datasets
- Feature extraction for molecular machine learning pipelines
- Pretraining foundation for specialized chemistry models

## How to Use
Load and use the model for fill-mask predictions on chemical SMILES:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "DeepChem/ChemBERTa-10M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: predict masked token in SMILES
smiles = "CC(C)C[C@H](NC(=O)[C@H](CC(=O)N)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(=O)N)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H]1CCCN1C(=O))[C@@H](O)CC(=O)N"
inputs = tokenizer(smiles, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits
```

## Input Format
- SMILES strings (chemical notation representing molecular structure)
- Input should be a valid SMILES representation of a molecule
- Tokens are separated by the model's tokenizer trained on chemical data
- Masked positions denoted by `<mask>` token

## Output Format
- **Logits**: Raw prediction scores for each token in vocabulary
- **Top predictions**: List of most likely tokens for masked positions with probabilities
- Output shape: `(batch_size, sequence_length, vocabulary_size)`
- Use softmax to convert logits to probabilities

## Example
```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="DeepChem/ChemBERTa-10M-MLM")

# Predict missing atom/bond in SMILES
result = fill_mask("CC(C)C<mask>C(=O)O")
for pred in result:
    print(f"Token: {pred['token_str']}, Score: {pred['score']:.4f}")
```

## Notes
- Trained exclusively on SMILES representations; input must be valid chemical notation
- Model vocabulary is specialized for chemistry; standard English text will not tokenize properly
- Best performance on drug-like molecules (MW < 500, similar to training distribution)
- Requires `transformers` and `torch` libraries
- Model size: ~125M parameters; GPU recommended for batch processing
- Fine-tuning on task-specific data typically improves downstream performance
- No built-in SMILES validation; invalid SMILES may produce unreliable predictions