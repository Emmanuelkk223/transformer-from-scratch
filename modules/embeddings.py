import math
import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    """
    Combines learnable token embeddings with fixed sinusoidal positional encodings,
    scaled by sqrt(d_model) as specified in Section 3.4 of the paper.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings matrix once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model) prior to positional addition
        embeddings = self.lut(x) * math.sqrt(self.d_model)
        seq_len = x.size(1)

        # Add static positional encodings up to seq_len
        x = embeddings + self.pe[:, :seq_len]
        return self.dropout(x)
