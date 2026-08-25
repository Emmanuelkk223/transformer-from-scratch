import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention supporting both primitive tensor math
    and PyTorch 2.0+ Fused SDPA (FlashAttention / Memory-Efficient).
    """

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

        # 1. High-Performance Fused SDPA Kernel
        if self.use_flash_attn:
            p_drop = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=p_drop
            )
            return output, None

        # 2. Primitive Tensor Implementation
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention supporting optional KV Caching and FlashAttention.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
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

        # Linear projections & split heads
        query = (
            self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )
        key = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = (
            self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )

        # Retrieve and concatenate KV cache
        if layer_paste_kv is not None:
            past_k, past_v = layer_paste_kv
            key = torch.cat([past_k, key], dim=-2)
            value = torch.cat([past_v, value], dim=-2)

        current_kv = (key, value) if use_cache else None

        # Calculate attention
        x, attn_weights = self.attention(query, key, value, mask=mask)

        # Concatenate heads and project output
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(x)

        if use_cache:
            return output, attn_weights, current_kv
        return output, attn_weights
