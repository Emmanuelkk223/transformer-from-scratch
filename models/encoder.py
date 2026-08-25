import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.feed_forward import PositionwiseFeedForward
from modules.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """
    Single Encoder layer utilizing Pre-LN residual connections.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 1. Multi-Head Self-Attention Sub-Layer (Pre-LN)
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(q=norm_x, k=norm_x, v=norm_x, mask=mask)
        x = x + self.dropout(attn_out)

        # 2. Feed-Forward Sub-Layer (Pre-LN)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Encoder(nn.Module):
    """
    Stack of N identical EncoderLayers followed by a final Layer Normalization.
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)
