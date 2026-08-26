import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.feed_forward import SwiGLUFFN
from modules.layer_norm import RMSNorm


class EncoderLayer(nn.Module):
    """Modern Pre-RMSNorm Encoder Layer with GQA & SwiGLU."""

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
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(q=norm_x, k=norm_x, v=norm_x, mask=mask)
        x = x + self.dropout(attn_out)

        norm_ffn = self.norm2(x)
        ffn_out = self.ffn(norm_ffn)
        x = x + self.dropout(ffn_out)
        return x


class Encoder(nn.Module):
    """Stacked Transformer Encoder."""

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
        self.norm = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)
