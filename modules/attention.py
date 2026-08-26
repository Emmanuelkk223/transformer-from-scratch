import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_emb(x: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embeddings (RoPE) to Q/K tensors."""
    seq_len = x.size(2)
    dim = x.size(-1)
    device = x.device

    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().view(1, 1, seq_len, dim)
    sin = emb.sin().view(1, 1, seq_len, dim)

    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)

    return (x * cos) + (x_rotated * sin)


class ScaledDotProductAttention(nn.Module):
    """Attention core with FP16-safe masking and FlashAttention support."""

    def __init__(self, dropout: float = 0.1, use_flash_attn: bool = False):
        super().__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        self.use_flash_attn = use_flash_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_flash_attn:
            p_drop = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=p_drop
            )
            return output, None

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            fill_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, fill_val)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Modern Multi-Head & Grouped-Query Attention (GQA) with RoPE and KV Caching.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            dropout=dropout, use_flash_attn=use_flash_attn
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        layer_paste_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        batch_size = q.size(0)

        query = (
            self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )
        key = (
            self.w_k(k)
            .view(batch_size, -1, self.num_kv_heads, self.d_k)
            .transpose(1, 2)
        )
        value = (
            self.w_v(v)
            .view(batch_size, -1, self.num_kv_heads, self.d_k)
            .transpose(1, 2)
        )

        if self.use_rope:
            query = apply_rotary_emb(query)
            key = apply_rotary_emb(key)

        if layer_paste_kv is not None:
            past_k, past_v = layer_paste_kv
            key = torch.cat([past_k, key], dim=-2)
            value = torch.cat([past_v, value], dim=-2)

        current_kv = (key, value) if use_cache else None

        # Repeat KV heads for GQA alignment if num_kv_heads < num_heads
        if self.num_queries_per_kv > 1:
            key = torch.repeat_interleave(key, dim=1, repeats=self.num_queries_per_kv)
            value = torch.repeat_interleave(
                value, dim=1, repeats=self.num_queries_per_kv
            )

        x, attn_weights = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(x)

        if use_cache:
            return output, attn_weights, current_kv
        return output, attn_weights
