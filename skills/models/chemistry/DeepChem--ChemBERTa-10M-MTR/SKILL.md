---
name: chemberta-10m-mtr
description: A RoBERTa-based transformer model pre-trained on 10M chemical compounds for molecular property prediction and drug discovery tasks. Use this model for predicting molecular characteristics and screening compounds in early-stage drug discovery workflows.
---

# ChemBERTa-10M-MTR

## Overview
ChemBERTa-10M-MTR is a transformer-based language model adapted from RoBERTa and pre-trained on 10 million chemical compounds from ChEMBL. It learns molecular representations by predicting chemical structures and properties, enabling downstream prediction of drug-like molecular characteristics. The model solves the problem of limited labeled molecular datasets by providing transfer learning capabilities for drug discovery and computational chemistry tasks.

## When to Use
- Predicting molecular properties (toxicity, activity, binding affinity) from chemical structures
- Molecular representation learning for similarity-based compound screening
- Transfer learning for drug discovery projects with limited training data
- Chemical compound classification and regression tasks
- Multi-task molecular property prediction

## How to Use
```python
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

# Download model weights
local_dir = snapshot_download(repo_id="DeepChem/ChemBERTa-10M-MTR")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-10M-MTR")

# Tokenize SMILES notation chemical representation
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
inputs = tokenizer(smiles, return_tensors="pt")

# Get molecular embeddings
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

## Input Format
SMILES strings (Simplified Molecular Input Line Entry System) representing chemical structures. Each SMILES string is tokenized into chemical tokens recognized during pre-training on 10M compounds.

## Output Format
Contextualized token embeddings of dimension 384 for each token in the input SMILES. The `[CLS]` token embedding represents the full molecule and can be used for downstream prediction tasks.

## Example
```python
# Predict molecular property using fine-tuned model
smiles_list = ["CCO", "CC(C)O", "c1ccccc1"]  # Ethanol, Isopropanol, Benzene
batch = tokenizer(smiles_list, padding=True, return_tensors="pt")
outputs = model(**batch)
# Use embeddings for property prediction downstream model
mol_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] tokens
```

## Notes
- Requires valid SMILES notation as input; invalid SMILES may produce poor results
- Pre-trained on ChEMBL compounds; may have biased representation toward drug-like molecules
- Best used as a feature extractor or fine-tuned on domain-specific tasks rather than as a generative model
- Dependencies: `transformers`, `torch`, `huggingface-hub`
- The MTR suffix indicates multi-task representation learning from molecular properties
- Model is endpoint-compatible and can be deployed on Hugging Face Inference API