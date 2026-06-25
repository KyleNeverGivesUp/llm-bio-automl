---
name: chemberta-100m-mlm
description: A 100M parameter RoBERTa model pre-trained on chemical SMILES data for masked language modeling. Use this model for molecular property prediction, chemical compound analysis, and drug discovery tasks that require understanding of chemical structure representations.
---

# ChemBERTa-100M-MLM

## Overview
ChemBERTa-100M-MLM is a masked language model based on RoBERTa architecture, pre-trained on large-scale chemical SMILES (Simplified Molecular Input Line Entry System) data. The model learns chemical structure patterns and molecular representations through masked token prediction, making it suitable for downstream drug discovery and cheminformatics applications. It encodes chemical knowledge directly into its learned embeddings.

## When to Use
This model is best suited for:
- Predicting masked tokens in chemical SMILES strings
- Feature extraction for molecular property prediction
- Chemical compound similarity and clustering tasks
- Transfer learning for drug discovery applications
- Molecular fingerprint generation
- Fine-tuning on downstream cheminformatics tasks with limited labeled data

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load model and tokenizer
model_name = "DeepChem/ChemBERTa-100M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: Fill masked token in SMILES string
smiles = "CC(=O)O[MASK]"
inputs = tokenizer(smiles, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Get predictions for masked token
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode([predicted_token_id])
print(f"Predicted token: {predicted_token}")
```

## Input Format
Inputs are SMILES strings (chemical notation) tokenized into subword tokens. The model expects:
- Text input: Valid SMILES notation with optional [MASK] tokens for prediction
- Token IDs: Integers corresponding to the tokenizer's vocabulary
- Masked positions: Indicated with [MASK] token for fill-mask tasks
- Max sequence length: Typically 512 tokens

## Output Format
For masked language modeling:
- **Logits**: Shape `(batch_size, sequence_length, vocab_size)` containing prediction scores for each token position
- **Hidden states** (optional): Contextualized embeddings of shape `(batch_size, sequence_length, 768)` for downstream tasks
- Predictions can be obtained by taking the argmax across the vocabulary dimension at masked positions

## Example
```python
from transformers import pipeline

# Use fill-mask pipeline for convenience
unmasker = pipeline('fill-mask', model='DeepChem/ChemBERTa-100M-MLM')

# Predict missing chemical group in aspirin-like compound
results = unmasker("CC(=O)O[MASK]")
for result in results:
    print(f"Token: {result['token_str']}, Score: {result['score']:.4f}")
```

## Notes
- This is a pre-trained masked language model, not a classification or regression model. Fine-tuning is recommended for specific prediction tasks.
- SMILES tokenization requires the specific tokenizer from the model repository; standard BERT tokenizers will not work correctly.
- Model performance depends on the quality and relevance of SMILES strings in your domain.
- Requires `transformers`, `torch`, and `tokenizers` libraries.
- MIT licensed; suitable for commercial and research applications.
- For optimal results on drug discovery tasks, consider fine-tuning on task-specific chemical datasets.