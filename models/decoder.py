import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.feed_forward import PositionwiseFeedForward
from modules.layer_norm import LayerNorm
from models.encoder import Encoder


class DecoderLayer(nn.Module):
    """
    Single Decoder layer with Masked Self-Attention, Cross-Attention, and FFN.
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
        self.cross_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Masked Self-Attention Sub-Layer (Pre-LN)
        norm_x = self.norm1(x)
        self_attn_out, _ = self.self_attn(q=norm_x, k=norm_x, v=norm_x, mask=tgt_mask)
        x = x + self.dropout(self_attn_out)

        # 2. Encoder-Decoder Cross-Attention Sub-Layer (Pre-LN)
        norm_x = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            q=norm_x, k=enc_output, v=enc_output, mask=src_mask
        )
        x = x + self.dropout(cross_attn_out)

        # 3. Feed-Forward Sub-Layer (Pre-LN)
        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x, cross_attn_weights


class Decoder(nn.Module):
    """
    Stack of N identical DecoderLayers followed by a final Layer Normalization.
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
                DecoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last_attn_weights = None
        for layer in self.layers:
            x, last_attn_weights = layer(
                x, enc_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask
            )
        return self.norm(x), last_attn_weights


if __name__ == "__main__":
    batch_size, src_len, tgt_len, d_model = 2, 12, 10, 512

    # Mock representations
    src_rep = torch.randn(batch_size, src_len, d_model)
    tgt_rep = torch.randn(batch_size, tgt_len, d_model)

    # Instantiate Encoder & Decoder
    encoder = Encoder(num_layers=6, d_model=d_model)
    decoder = Decoder(num_layers=6, d_model=d_model)

    enc_out = encoder(src_rep)
    dec_out, attn_map = decoder(tgt_rep, enc_output=enc_out)

    assert enc_out.shape == (batch_size, src_len, d_model)
    assert dec_out.shape == (batch_size, tgt_len, d_model)
    print(" Encoder and Decoder stacks executed successfully!")
