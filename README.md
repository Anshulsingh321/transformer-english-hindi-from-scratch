# Transformer-based English → Hindi Machine Translation (From Scratch)

## Overview
This project implements the **Transformer architecture from scratch in PyTorch**, based on the paper  
**“Attention Is All You Need” (Vaswani et al., 2017)**.

The model is trained for **English → Hindi machine translation** using a real parallel corpus.  
All major components — tokenization, attention, encoder, decoder, masking, training, and inference — are implemented manually **without using `nn.Transformer` or HuggingFace**.

---

## Key Features
- Transformer implemented from scratch
- Multi-head self-attention
- Sinusoidal positional encoding
- Encoder–decoder architecture
- Proper masking (padding + causal masks)
- Teacher forcing during training
- Real training on English–Hindi dataset
- Greedy decoding for inference

---

## Architecture

### Encoder
- Token embedding + positional encoding
- 6 stacked encoder layers
- Multi-head self-attention
- Feed-forward networks

### Decoder
- Masked self-attention
- Encoder–decoder attention
- Feed-forward networks

### Output
- Linear projection to Hindi vocabulary

---

## Project Structure
transformer-english-hindi-from-scratch/
├── data/
│   └── dataset.py
├── utils/
│   ├── tokenizer.py
│   └── masking.py
├── model/
│   ├── embeddings.py
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
├── training/
│   └── train.py
├── inference/
│   ├── init.py
│   └── translate.py
├── README.md
└── requirements.txt
---

## Dataset
- English–Hindi parallel corpus
- ~125,000 sentence pairs after cleaning
- Source: Kaggle (English–Hindi Translation Dataset)
- Word-level tokenization
- Vocabulary sizes:
  - English: 15,000
  - Hindi: 20,000

> Dataset files are not included in this repository due to size and licensing constraints.

---

## Training Details
- Model: Transformer (Encoder–Decoder)
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: CrossEntropyLoss (padding ignored)
- Training Strategy: Teacher Forcing
- Epochs: 5 (controlled training)
- Hardware: Google Colab (Tesla T4 GPU)

### Training Loss Progression
Epoch 1 → 5.85
Epoch 2 → 5.35
Epoch 3 → 5.05
Epoch 4 → 4.79
Epoch 5 → 4.57
---

## Inference
After training, the model generates Hindi translations autoregressively using **greedy decoding**.

### Example
EN: how are you
HI: आप कैसे
---

## References
- Vaswani et al., *Attention Is All You Need*, 2017  
  https://arxiv.org/abs/1706.03762
