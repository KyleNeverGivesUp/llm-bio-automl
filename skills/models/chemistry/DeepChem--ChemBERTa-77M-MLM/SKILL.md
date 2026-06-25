```markdown
---
name: chemberta-77m-mlm
description: ChemBERTa-77M-MLM is a RoBERTa-based masked language model pre-trained on chemical SMILES data for molecular property prediction and chemical representation learning. Use it for fill-mask tasks to predict missing tokens in chemical sequences, enabling applications in drug discovery and molecular design.
---

# ChemBERTa-77M-MLM

## Overview
ChemBERTa-77M-MLM is a 77-million parameter RoBERTa model pre-trained on chemical SMILES (Simplified Molecular Input Line Entry System) representations. It learns deep chemical representations through masked language modeling, enabling it to understand molecular structures and predict missing tokens in chemical sequences. This model serves as a foundational representation for downstream drug discovery and molecular property prediction tasks.

## When to Use
This model is best suited for:
- **Masked token prediction**: Predicting missing or masked atoms/bonds in chemical structures
- **Molecular representation learning**: Extracting meaningful embeddings for chemical compounds
- **Drug discovery pipelines**: As a pre-trained feature extractor for molecular property prediction
- **Chemical similarity analysis**: Computing semantic similarity between molecular structures
- **Fine-tuning for downstream tasks**: Transfer learning for tasks like toxicity prediction, solubility estimation, or binding affinity prediction

## How to Use
Load and use the model with the Hugging Face `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example: predict masked token in SMILES
smiles = "CC(=O)O[MASK]"
inputs = tokenizer(smiles, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits
```

## Input Format
The model accepts SMILES strings (molecular notation) as input. Tokens can be individual characters representing atoms and bonds, or special tokens like `[MASK]` (typically `<mask>`) for fill-mask tasks. Inputs are tokenized using the ChemBERTa tokenizer which has been trained on chemical vocabulary.

Example valid inputs:
- `"CC(=O)O"` (acetic acid)
- `"CC(=O)O[MASK]"` (with masked position)
- `"c1ccccc1"` (benzene)

## Output Format
For masked language modeling tasks, the output is a tensor of shape `(batch_size, sequence_length, vocab_size)` containing logit scores for each token position. The highest logit values indicate the most likely token predictions for masked positions. For embeddings extraction, intermediate layer outputs provide rich chemical representations.

## Example
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "DeepChem/ChemBERTa-77M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Predict masked atom in aspirin-like molecule
smiles = "CC(=O)Oc1ccccc1[MASK](=O)O"
inputs = tokenizer(smiles, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get predictions for the masked position
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
predictions = outputs.logits[0, mask_token_index]
predicted_token = predictions.argmax(axis=-1)
print(tokenizer.decode(predicted_token))
```

## Notes
- **SMILES vocabulary**: The model is specifically trained on SMILES notation; inputs should be valid SMILES strings
- **Dependencies**: Requires `transformers>=4.0` and `torch`
- **Computational requirements**: 77M parameters; inference is CPU-compatible but GPU recommended for batch processing
- **Fine-tuning**: Recommended for best results on specific downstream tasks (property prediction, binding affinity, etc.)
- **Tokenization**: Must use the provided ChemBERTa tokenizer to maintain compatibility with pre-trained weights
```
</markdown>