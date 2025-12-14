Transformer-based English → Hindi Machine Translation (From Scratch)

1. Overview -
This project implements the Transformer architecture from scratch in PyTorch, based on the paper
“Attention Is All You Need” (Vaswani et al., 2017).

The model is trained for English → Hindi machine translation using a real parallel corpus.
All major components — tokenization, attention, encoder, decoder, masking, training, and inference — are implemented manually without using nn.Transformer or HuggingFace.
⸻

2. Key Features -
	•	Transformer implemented from scratch
	•	Multi-head self-attention
	•	Sinusoidal positional encoding
	•	Encoder–decoder architecture
	•	Proper masking (padding + causal masks)
	•	Teacher forcing during training
	•	Real training on English–Hindi dataset
	•	Greedy decoding for inference
⸻

3. Architecture -
	•	Encoder
	•	Token embedding + positional encoding
	•	6 stacked encoder layers
	•	Multi-head self-attention
	•	Feed-forward networks
	•	Decoder
	•	Masked self-attention
	•	Encoder–decoder attention
	•	Feed-forward networks
	•	Output
	•	Linear projection to Hindi vocabulary
⸻

4. Project Structure -
.
├── data/
│   └── dataset.py
│
├── utils/
│   ├── tokenizer.py
│   └── masking.py
│
├── model/
│   ├── embeddings.py
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
│
├── training/
│   └── train.py
│
├── inference/
│   ├── __init__.py
│   └── translate.py
│
├── transformer_en_hi_epoch*.pt
└── README.md
⸻

5. Dataset -
	•	English–Hindi parallel corpus
	•	~125,000 sentence pairs after cleaning
	•	Source: Kaggle (English–Hindi Translation Dataset)
	•	Word-level tokenization
	•	Vocabulary sizes:
	•	English: 15,000
	•	Hindi: 20,000
⸻

6. Training Details -
	•	Model: Transformer (Encoder–Decoder)
	•	Optimizer: Adam
	•	Learning Rate: 1e-4
	•	Loss: CrossEntropyLoss (padding ignored)
	•	Training Strategy: Teacher Forcing
	•	Epochs: 5 (controlled training)
	•	Hardware: Google Colab (Tesla T4 GPU)

Training Loss Progression
Epoch 1 → 5.85
Epoch 2 → 5.35
Epoch 3 → 5.05
Epoch 4 → 4.79
Epoch 5 → 4.57
⸻

7. Inference (Greedy Decoding) -
After training, the model generates Hindi translations autoregressively using greedy decoding.

Example Outputs
EN: how are you
HI: आप कैसे <unk>

EN: i am a student
HI: मैं एक <unk> <unk>

EN: what is your name
HI: क्या आप को क्या कर रहे हैं
Note: Due to word-level tokenization, limited epochs, and greedy decoding, some rare words appear as <unk>.
Despite this, the model learns Hindi sentence structure and syntax, demonstrating correct Transformer behavior.
⸻

8. How to Run Inference -

from inference.translate import greedy_decode
translation = greedy_decode(
    model,
    "i am a student",
    eng_vocab,
    hin_vocab,
    device="cuda"
)
print(translation)
⸻

9. Future Improvements -
	•	Subword tokenization (BPE / SentencePiece)
	•	Beam search decoding
	•	Longer training with learning rate scheduling
	•	BLEU score evaluation
	•	Pretrained embeddings
⸻

10. Key Learnings -
	•	Deep understanding of Transformer internals
	•	Attention mechanisms and masking
	•	Training sequence-to-sequence models
	•	Handling real NLP datasets
	•	Building end-to-end ML systems from scratch
⸻

11. References -
	•	Vaswani et al., Attention Is All You Need, 2017
	•	https://arxiv.org/abs/1706.03762
⸻

12. Author-
Anshul Singh
B.Tech Computer Science
Interests: NLP, Deep Learning, Transformers, Machine Translation
