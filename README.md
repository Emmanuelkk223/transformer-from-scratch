# Transformer from Scratch: Complete PyTorch Guide & Implementation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end, zero-abstraction implementation of the Transformer sequence-to-sequence model built entirely from scratch in PyTorch, strictly following the foundational paper **["Attention Is All You Need" (Vaswani et al., NIPS 2017)](https://arxiv.org/abs/1706.03762)**.

This repository avoids high-level wrapper abstractions (such as `torch.nn.Transformer` or Hugging Face `transformers`) to expose the low-level mathematical primitives, tensor transformations, masking mechanisms, decoding algorithms, and optimization mechanics behind modern Transformer architectures.

---

## 📋 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [Mathematical Mechanics](#-mathematical-mechanics)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Complete Usage Guide](#-complete-usage-guide)
  - [1. Subword Tokenizer Training](#1-subword-tokenizer-training)
  - [2. Model Training & Optimization](#2-model-training--optimization)
  - [3. Single-Sentence Generation (`generate.py`)](#3-single-sentence-generation-generatepy)
  - [4. Interactive CLI Evaluation (`scripts/evaluate.py`)](#4-interactive-cli-evaluation-scriptsevaluatepy)
  - [5. Visualizing Attention Heatmaps](#5-visualizing-attention-heatmaps)
- [Hyperparameters & Configuration](#-hyperparameters--configuration)
- [Automated Testing (`pytest`)](#-automated-testing-pytest)
- [Module Code Blueprint](#-module-code-blueprint)
- [References & Citation](#-references--citation)
- [License](#-license)

---

## 🏛️ Architectural Overview

The model implements a full **Encoder-Decoder Architecture** designed for autoregressive sequence-to-sequence tasks (e.g., Neural Machine Translation).

```
                 [ Output Target Sequence ]
                             │
                             ▼
                    Output Embeddings
                             │
                             ▼
                   Positional Encoding
                             │
                             ▼
                     ┌───────────────┐
                     │ Decoder Block │ ◄──┐
                     │  (Pre-LN x N) │    │
                     └───────┬───────┘    │
                             │            │ Masked Cross-Attention
                             ▼            │
                 ┌───────────────────────┴──┐
                 │       Encoder Stack       │
                 │       (Pre-LN x N)        │
                 └───────────▲──────────────┘
                             │
                    Positional Encoding
                             │
                             ▼
                     Input Embeddings
                             │
                             ▼
                  [ Input Source Sequence ]
```

---

## 🧮 Mathematical Mechanics

### 1. Scaled Dot-Product Attention
Calculated across Query ($Q$), Key ($K$), and Value ($V$) projections with scaling factor $\sqrt{d_k}$ to prevent gradient vanishing in the softmax function at high dimension scales:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

*where $M$ is the dynamic masking tensor ($-\infty$ for masked positions, $0$ otherwise).*

### 2. Multi-Head Attention (MHA)
Splits $d_{\text{model}}$ dimensional vectors across $h$ parallel attention heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 3. Sinusoidal Positional Encoding
Injects spatial ordering directly into sequence embeddings without adding learnable parameters:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

### 4. Position-Wise Feed-Forward Network (FFN)
$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

---

## 📁 Repository Structure

```text
transformer-from-scratch/
├── config/
│   └── default_config.yaml       # Hyperparameter & path configurations
├── data/
│   ├── dataset.py                # PyTorch Dataset & DataLoader with dynamic masking
│   └── tokenizer.py              # BPE Subword Tokenizer pipeline
├── modules/
│   ├── attention.py              # Scaled Dot-Product & Multi-Head Attention
│   ├── embeddings.py             # Token Embedding & Sinusoidal Positional Encoding
│   ├── feed_forward.py           # Position-wise Feed-Forward Networks
│   └── layer_norm.py             # Custom Layer Normalization (Pre-LN)
├── models/
│   ├── encoder.py                # Encoder Layer & Encoder Stack
│   ├── decoder.py                # Decoder Layer & Decoder Stack
│   └── transformer.py            # End-to-End Seq2Seq Transformer model
├── utils/
│   ├── decoding.py               # Greedy Search & Beam Search inference algorithms
│   ├── scheduler.py              # Noam Learning Rate Scheduler
│   └── visualization.py          # Multi-Head Attention Heatmap utilities
├── scripts/
│   ├── train.py                  # Main training execution pipeline
│   └── evaluate.py               # Interactive CLI translation & evaluation script
├── tests/
│   └── test_modules.py           # Comprehensive PyTorch unit testing suite
├── generate.py                   # Single-sentence generation CLI tool
├── requirements.txt              # Core dependencies
├── .gitignore                    # Version control ignore rules
├── LICENSE                       # MIT License
└── README.md                     # Documentation
```

---

## 💻 Installation & Setup

### 1. Clone & Navigate
```bash
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch
```

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Complete Usage Guide

### 1. Subword Tokenizer Training
Train Byte-Pair Encoding (BPE) subword tokenizers for source and target languages:
```bash
python data/tokenizer.py \
  --src_data data/raw/train.en \
  --tgt_data data/raw/train.de \
  --vocab_size 32000 \
  --save_dir checkpoints/
```

### 2. Model Training & Optimization
Train the model using mixed precision and Noam warmup scheduling:
```bash
python scripts/train.py --config config/default_config.yaml
```

### 3. Single-Sentence Generation (`generate.py`)
Translate an input sentence using Beam Search ($k=5$):
```bash
python generate.py \
  --text "The black cat sat on the mat." \
  --checkpoint checkpoints/best_model.pt \
  --src_tok checkpoints/src_tok.json \
  --tgt_tok checkpoints/tgt_tok.json \
  --beam_size 5
```

### 4. Interactive CLI Evaluation (`scripts/evaluate.py`)
Launch an interactive session to evaluate inputs in real-time:
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --src_tok checkpoints/src_tok.json \
  --tgt_tok checkpoints/tgt_tok.json \
  --beam_size 5
```

### 5. Visualizing Attention Heatmaps
Extract and render attention weight matrices across all heads:
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --src_tok checkpoints/src_tok.json \
  --tgt_tok checkpoints/tgt_tok.json \
  --visualize
```

---

## ⚙️ Hyperparameters & Configuration

Default settings in `config/default_config.yaml` follow the **Transformer-Base** specification:

| Hyperparameter | Paper Baseline | Our Config | Description |
| :--- | :---: | :---: | :--- |
| **Encoder Layers ($N$)** | 6 | 6 | Number of stacked encoder layers |
| **Decoder Layers ($N$)** | 6 | 6 | Number of stacked decoder layers |
| **Model Dimension ($d_{\text{model}}$)** | 512 | 512 | Hidden representation dimensionality |
| **Feed-Forward Dimension ($d_{ff}$)** | 2048 | 2048 | Inner dimension of position-wise FFN |
| **Attention Heads ($h$)** | 8 | 8 | Number of parallel attention heads |
| **Head Dimension ($d_k = d_v$)** | 64 | 64 | Dimension per attention head ($d_{\text{model}} / h$) |
| **Dropout ($P_{drop}$)** | 0.1 | 0.1 | Residual and attention dropout probability |
| **Warmup Steps** | 4000 | 4000 | Linear warmup steps for Noam scheduler |
| **Label Smoothing ($\epsilon$)** | 0.1 | 0.1 | Cross-entropy target smoothing factor |

---

## 🧪 Automated Testing (`pytest`)

Verify tensor operations, masking logic, and shape propagation:
```bash
pytest tests/ -v
```

---

## 🧩 Module Code Blueprint

### Core Attention Layer (`modules/attention.py`)
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, value), attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        query = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, attn_weights = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x), attn_weights
```

### Positional Embedding Layer (`modules/embeddings.py`)
```python
import math
import torch
import torch.nn as nn

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        embeddings = self.lut(x) * math.sqrt(self.d_model)
        x = embeddings + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

---

## 📚 References & Citation

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5998--6008},
  year={2017}
}
```

---

## 📄 License

This project is open-source software licensed under the [MIT License](LICENSE).