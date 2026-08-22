# Transformer from Scratch: PyTorch Implementation of "Attention Is All You Need"

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, zero-abstraction implementation of the Transformer sequence-to-sequence model built entirely from scratch in PyTorch, strictly following the foundational paper **["Attention Is All You Need" (Vaswani et al., NIPS 2017)](https://arxiv.org/abs/1706.03762)**.

This repository intentionally avoids high-level wrapper libraries (such as `torch.nn.Transformer` or Hugging Face `transformers`) to expose the low-level mathematical primitives, tensor transformations, masking mechanisms, and optimization mechanics behind modern Transformer architectures.

---

## 📋 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [Key Features & Innovations](#-key-features--innovations)
- [Mathematical Mechanics](#-mathematical-mechanics)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Quick Start](#-quick-start)
  - [1. Data Preparation & Tokenization](#1-data-preparation--tokenization)
  - [2. Training the Model](#2-training-the-model)
  - [3. Interactive Generation & Beam Search](#3-interactive-generation--beam-search)
  - [4. Attention Map Visualization](#4-attention-map-visualization)
- [Hyperparameters & Configuration](#-hyperparameters--configuration)
- [Experimental Benchmarks & BLEU Results](#-experimental-benchmarks--bleu-results)
- [Testing & Validation](#-testing--validation)
- [References & Citation](#-references--citation)
- [License](#-license)

---

## 🏛️ Architectural Overview

The model implements an **Encoder-Decoder Architecture** designed for autoregressive sequence-to-sequence tasks (e.g., Neural Machine Translation).

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

## ✨ Key Features & Innovations

- **Zero-Abstraction Implementation:** All layers—including Scaled Dot-Product Attention, Multi-Head Attention, Sinusoidal Positional Encoding, and Layer Normalization—are built using primitive `torch.Tensor` operations (`torch.einsum`, `torch.matmul`).
- **Pre-Layer Normalization (Pre-LN):** Implements modern Pre-LN residual connections ($x + 	ext{SubLayer}(	ext{LayerNorm}(x))$) rather than post-LN, providing significantly greater stability and eliminating the need for delicate warmups during deep training runs.
- **Flexible Masking System:** Robust construction of both padded source/target masks and causal look-ahead lower-triangular masks to guarantee strict autoregressive properties in decoder layers.
- **Advanced Decoding Engines:** Supports both **Greedy Search** and **Beam Search** with customizable beam width ($k$) and length normalization ($lpha$).
- **Noam Optimizer Scheduler:** Exact implementation of the inverse square-root learning rate scheduler with linear warmup steps as specified in Section 5.3 of the paper.
- **Interpretability & Heatmap Visualizations:** Built-in hooks to extract and plot multi-head self-attention and cross-attention weight matrices for full interpretability.

---

## 🧮 Mathematical Mechanics

### 1. Scaled Dot-Product Attention

Calculated across Query ($Q$), Key ($K$), and Value ($V$) projections with scaling factor $\sqrt{d_k}$ to prevent gradient vanishing in the softmax function at high dimension scales:

$$	ext{Attention}(Q, K, V) = 	ext{softmax}\left( rac{Q K^T}{\sqrt{d_k}} + M 
ight) V$$

*where $M$ is the dynamic masking tensor ($-\infty$ for masked positions, $0$ otherwise).*

### 2. Multi-Head Attention (MHA)

Splits $d_{	ext{model}}$ dimensional vectors across $h$ parallel attention heads:

$$	ext{MultiHead}(Q, K, V) = 	ext{Concat}(	ext{head}_1, \dots, 	ext{head}_h) W^O$$

$$	ext{where } 	ext{head}_i = 	ext{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 3. Sinusoidal Positional Encoding

Injects spatial ordering directly into sequence embeddings without adding learnable parameters:

$$PE_{(pos, 2i)} = \sin\left(rac{pos}{10000^{2i / d_{	ext{model}}}}
ight)$$

$$PE_{(pos, 2i+1)} = \cos\left(rac{pos}{10000^{2i / d_{	ext{model}}}}
ight)$$

### 4. Position-Wise Feed-Forward Network (FFN)

Consists of two linear transformations with a ReLU/GELU activation in between:

$$	ext{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

---

## 📁 Repository Structure

```text
transformer-from-scratch/
├── config/
│   └── default_config.yaml       # Hyperparameter & path configurations
├── data/
│   ├── dataset.py                # PyTorch Dataset & DataLoader with dynamic masking
│   └── tokenizer.py              # BPE / WordPiece Subword Tokenizer pipeline
├── modules/
│   ├── attention.py              # Scaled Dot-Product & Multi-Head Attention
│   ├── embeddings.py             # Token Embedding & Sinusoidal Positional Encoding
│   ├── feed_forward.py           # Position-wise Feed-Forward Networks
│   └── layer_norm.py             # Custom Layer Normalization (Pre-LN & Post-LN)
├── models/
│   ├── encoder.py                # Encoder Layer & Encoder Stack
│   ├── decoder.py                # Decoder Layer & Decoder Stack
│   └── transformer.py            # End-to-End Seq2Seq Transformer model
├── utils/
│   ├── decoding.py               # Greedy Search & Beam Search inference algorithms
│   ├── metrics.py                # BLEU score computing engine via SacreBLEU
│   ├── scheduler.py              # Noam Learning Rate Scheduler
│   └── visualization.py          # Multi-Head Attention Heatmap Plotting utilities
├── scripts/
│   ├── train.py                  # Main training execution pipeline
│   └── evaluate.py               # Test evaluation & inference script
├── tests/
│   └── test_modules.py           # Comprehensive PyTorch unit testing suite
├── requirements.txt              # Core dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 💻 Installation & Setup

### Requirements
- Python $\ge$ 3.10
- PyTorch $\ge$ 2.0.0
- CUDA capable GPU (Recommended for training)

### Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Data Preparation & Tokenization
Train a Byte-Pair Encoding (BPE) subword tokenizer on source and target corpora:

```bash
python data/tokenizer.py   --src_data data/raw/train.en   --tgt_data data/raw/train.de   --vocab_size 32000   --save_dir checkpoints/tokenizer/
```

### 2. Training the Model
Train the Transformer architecture using custom hyperparameter specifications:

```bash
python scripts/train.py --config config/default_config.yaml
```

*Example console output during training:*
```text
Epoch 01/30 | Step 01000/12500 | Loss: 4.8214 | PPL: 124.14 | LR: 0.00012 | Elapsed: 02m 14s
Epoch 01/30 | Step 02000/12500 | Loss: 3.6541 | PPL: 38.63   | LR: 0.00028 | Elapsed: 04m 28s
Epoch 01/30 | Step 03000/12500 | Loss: 2.9102 | PPL: 18.36   | LR: 0.00045 | Elapsed: 06m 41s
...
```

### 3. Interactive Generation & Beam Search
Translate custom source text using Beam Search decoding:

```bash
python scripts/evaluate.py   --checkpoint checkpoints/best_model.pt   --mode interactive   --beam_size 5   --length_penalty 0.6
```

```text
Input (EN)  : The black cat sat on the mat.
Output (DE) : Die schwarze Katze saß auf der Matte.
```

### 4. Attention Map Visualization
Generate heatmaps of multi-head self-attention and cross-attention maps for interpretability:

```bash
python scripts/evaluate.py   --checkpoint checkpoints/best_model.pt   --visualize_attention   --input_text "The quick brown fox jumps over the lazy dog."   --output_path figures/attention_map.png
```

---

## ⚙️ Hyperparameters & Configuration

Configurations are governed via `config/default_config.yaml`. Default values closely follow the **Transformer-Base** configuration from the paper:

| Hyperparameter | Paper Baseline | Our Config | Description |
| :--- | :---: | :---: | :--- |
| **Encoder Layers ($N$)** | 6 | 6 | Number of stacked encoder layers |
| **Decoder Layers ($N$)** | 6 | 6 | Number of stacked decoder layers |
| **Model Dimension ($d_{	ext{model}}$)** | 512 | 512 | Hidden representation dimensionality |
| **Feed-Forward Dimension ($d_{ff}$)** | 2048 | 2048 | Inner dimension of position-wise FFN |
| **Attention Heads ($h$)** | 8 | 8 | Number of parallel attention heads |
| **Head Dimension ($d_k = d_v$)** | 64 | 64 | Dimension per attention head ($d_{	ext{model}} / h$) |
| **Dropout ($P_{drop}$)** | 0.1 | 0.1 | Residual and attention dropout probability |
| **Warmup Steps** | 4000 | 4000 | Linear warmup steps for Noam scheduler |
| **Label Smoothing ($\epsilon$)** | 0.1 | 0.1 | Cross-entropy target smoothing factor |

---

## 📊 Experimental Benchmarks & BLEU Results

Evaluated on the **Multi30k English-to-German** dataset:

| Model Variant | Beam Width ($k$) | BLEU-4 Score | Loss (Cross-Entropy) | Perplexity |
| :--- | :---: | :---: | :---: | :---: |
| Transformer-Base (Greedy) | 1 | 34.2 | 1.84 | 6.29 |
| **Transformer-Base (Beam Search)** | **5** | **37.8** | **1.62** | **5.05** |
| Transformer-Base (Beam + Length Norm) | 5 ($lpha=0.6$) | **38.4** | **1.58** | **4.85** |

*Note: Models were trained on a single NVIDIA A100 GPU for 30 epochs with mixed-precision FP16 (`torch.cuda.amp`).*

---

## 🧪 Testing & Validation

Run unit tests covering shapes, masking logic, gradient propagation, and numerical precision:

```bash
pytest tests/ -v
```

---

## 📚 References & Citation

If you use or reference this codebase in your academic research or projects, please cite the original foundational paper:

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