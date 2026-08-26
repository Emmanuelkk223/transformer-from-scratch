import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.feed_forward import SwiGLUFFN
from modules.layer_norm import RMSNorm


class DecoderLayer(nn.Module):
    """Pre-RMSNorm Decoder Layer with RoPE in Self-Attn and Standard Cross-Attn."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Enable RoPE for Target Self-Attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, use_rope=True
        )
        # CRITICAL FIX: Disable RoPE for Cross-Attention
        self.cross_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, use_rope=False
        )
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        layer_paste_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        norm_x = self.norm1(x)
        if use_cache:
            self_out, _, current_kv = self.self_attn(
                q=norm_x,
                k=norm_x,
                v=norm_x,
                mask=tgt_mask,
                layer_paste_kv=layer_paste_kv,
                use_cache=True,
            )
        else:
            self_out, _ = self.self_attn(q=norm_x, k=norm_x, v=norm_x, mask=tgt_mask)
            current_kv = None

        x = x + self.dropout(self_out)

        norm_cross = self.norm2(x)
        cross_out, attn_weights = self.cross_attn(
            q=norm_cross, k=memory, v=memory, mask=src_mask
        )
        x = x + self.dropout(cross_out)

        norm_ffn = self.norm3(x)
        ffn_out = self.ffn(norm_ffn)
        x = x + self.dropout(ffn_out)

        if use_cache:
            return x, attn_weights, current_kv
        return x, attn_weights


class Decoder(nn.Module):
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
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ):
        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x), last_attn
