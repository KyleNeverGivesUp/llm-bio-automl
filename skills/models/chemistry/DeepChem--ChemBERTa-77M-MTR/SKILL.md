---
name: chemberta-77m-mtr
description: ChemBERTa-77M-MTR is a RoBERTa-based transformer model pre-trained on chemical SMILES data for molecular property prediction and drug discovery tasks. Use this model for transfer learning on molecular representation tasks, chemical property prediction, and computational drug discovery applications.
---

# ChemBERTa-77M-MTR

## Overview
ChemBERTa-77M-MTR is a 77 million parameter RoBERTa transformer model pre-trained on large-scale chemical structure data (SMILES notation). It learns contextual molecular representations useful for predicting chemical and biological properties of compounds, making it suitable for downstream drug discovery and molecular screening tasks.

## When to Use
This model is best suited for:
- Molecular property prediction (solubility, toxicity, binding affinity)
- Chemical similarity and clustering tasks
- Transfer learning for drug candidate screening
- Fine-tuning on small molecular datasets for bioactivity prediction
- Computational chemistry feature extraction

## How to Use
```python
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

# Download model
local_dir = snapshot_download(repo_id="DeepChem/ChemBERTa-77M-MTR")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

# Tokenize SMILES string
smiles = "CCO"  # ethanol
inputs = tokenizer(smiles, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

## Input Format
Expects SMILES (Simplified Molecular Input Line Entry System) strings as input. Each SMILES string represents a chemical structure. Input should be tokenized using the ChemBERTa tokenizer, which handles chemical tokens appropriately.

## Output Format
Returns contextual token embeddings (hidden states) with shape [batch_size, sequence_length, 768]. The pooled representation can be extracted from the [CLS] token for whole-molecule predictions.

## Example
```python
# Predict molecular embeddings for drug candidates
smiles_list = ["CCO", "c1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
inputs = tokenizer(smiles_list, padding=True, return_tensors="pt")
outputs = model(**inputs)

# Use [CLS] token embedding for downstream tasks
cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [3, 768]
```

## Notes
- Requires `transformers` library and PyTorch
- Model is optimized for chemical SMILES input; use canonical SMILES for best results
- Pre-trained weights are frozen; fine-tune on task-specific data for optimal performance
- Compatible with Hugging Face Model Hub endpoints
- MTR likely indicates Multi-Task learning or Molecular Training objectives