import math
import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    """
    Scaled Token Embedding layer with sinusoidal positional fallbacks.
    Multiplies lookup tables by sqrt(d_model) for semantic vector domination.
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

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.lut(x) * math.sqrt(self.d_model)
        x = embeddings + self.pe[:, : x.size(1)]
        return self.dropout(x)
