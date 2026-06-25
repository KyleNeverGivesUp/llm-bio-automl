---
name: chemberta-augmented-pubchem-13m
description: A RoBERTa-based language model pre-trained on 13 million augmented chemical SMILES from PubChem for masked language modeling tasks in drug discovery and cheminformatics applications.
---

# ChemBERTa_augmented_pubchem_13m

## Overview
ChemBERTa_augmented_pubchem_13m is a transformer-based model built on RoBERTa architecture, specifically trained for understanding chemical structures and molecular representations. It performs masked language modeling on SMILES (Simplified Molecular Input Line Entry System) strings derived from PubChem data. The model learns to predict masked tokens in chemical sequences, enabling it to capture meaningful chemical and structural patterns useful for downstream drug discovery tasks.

## When to Use
This model is best suited for:
- Molecular representation learning and chemical compound analysis
- Drug discovery and virtual screening applications
- Chemical property prediction tasks
- Cheminformatic feature extraction from SMILES strings
- Transfer learning for downstream chemistry-related classification or regression tasks
- Masked token prediction in chemical sequences for data augmentation and analysis

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
model_name = "Derify/ChemBERTa_augmented_pubchem_13m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example SMILES string with masked token
smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
masked_smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)[MASK]O"

# Tokenize and predict
inputs = tokenizer(masked_smiles, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Get the mask token position and top predictions
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode([predicted_token_id])
```

## Input Format
The model expects SMILES strings (chemical molecular representations) tokenized using the model's associated tokenizer. Input can be:
- Single SMILES strings
- Multiple SMILES strings in a batch
- SMILES with `[MASK]` tokens indicating positions to be predicted
- Maximum sequence length: 512 tokens (standard for RoBERTa)

## Output Format
The output is a tensor of logits with shape `(batch_size, sequence_length, vocab_size)` representing prediction scores for each token in the vocabulary at each position. For masked language modeling, the logits at masked token positions can be used to:
- Identify the most likely token (argmax)
- Obtain probability distributions over possible chemical tokens
- Generate alternative chemical structures

## Example
```python
from transformers import pipeline

# Use pipeline for convenience
unmasker = pipeline('fill-mask', model='Derify/ChemBERTa_augmented_pubchem_13m')

# Predict masked token in a drug-like molecule
results = unmasker("CC(C)Cc1ccc(cc1)[C@@H](C)[MASK]O")
for result in results:
    print(f"Token: {result['token_str']}, Score: {result['score']:.4f}")
```

## Notes
- The model is trained specifically on augmented PubChem SMILES data, making it optimized for drug-like molecules and common chemical motifs
- Performance is best for molecules similar to those in PubChem
- Requires `transformers` library version compatible with RoBERTa models
- Model outputs raw logits; softmax normalization is recommended for probability interpretations
- The tokenizer must be used consistently with the model for proper token alignment
- See associated papers (arxiv:2010.09885, arxiv:2209.01712) for detailed training procedures and benchmarks
- Licensed under Apache 2.0