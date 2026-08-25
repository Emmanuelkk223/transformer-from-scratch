import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention as defined in Section 3.2.1 of
    'Attention Is All You Need'.

    Formula: Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + M ) @ V
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Tensor of shape (batch_size, num_heads, seq_len_q, d_k)
            key: Tensor of shape (batch_size, num_heads, seq_len_k, d_k)
            value: Tensor of shape (batch_size, num_heads, seq_len_v, d_v) where seq_len_k == seq_len_v
            mask: Optional Tensor (batch_size, 1, seq_len_q, seq_len_k) or broadcastable

        Returns:
            output: Tensor of shape (batch_size, num_heads, seq_len_q, d_v)
            attn_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)

        # 1. Compute scaled dot products: (Q @ K^T) / sqrt(d_k)
        # Tensor shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply attention mask (if provided)
        # Mask out positions where value is 0 (False) with large negative value
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax over the key sequence length dimension
        attn_weights = F.softmax(scores, dim=-1)

        # 4. Apply dropout to attention probabilities
        attn_weights = self.dropout(attn_weights)

        # 5. Multiply attention weights by values: Attention_Weights @ V
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as defined in Section 3.2.2 of
    'Attention Is All You Need'.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key, Value vectors
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection layer
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Attention block
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Tensor of shape (batch_size, seq_len_q, d_model)
            k: Tensor of shape (batch_size, seq_len_k, d_model)
            v: Tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional Tensor (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            output: Tensor of shape (batch_size, seq_len_q, d_model)
            attn_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = q.size(0)

        # 1. Project inputs linearly and split across num_heads
        # Reshape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # Permute:  (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        query = (
            self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )
        key = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = (
            self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        )

        # 2. Compute Scaled Dot-Product Attention across all heads concurrently
        x, attn_weights = self.attention(query, key, value, mask=mask)

        # 3. Concatenate attention heads back together
        # Permute:  (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, num_heads, d_k)
        # Contiguous view: (batch_size, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final output linear projection
        output = self.w_o(x)

        return output, attn_weights


if __name__ == "__main__":
    # Unit Test Verification
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    dummy_input = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    out, weights = mha(q=dummy_input, k=dummy_input, v=dummy_input, mask=None)

    assert out.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"
    assert weights.shape == (
        batch_size,
        num_heads,
        seq_len,
        seq_len,
    ), f"Expected weights shape {(batch_size, num_heads, seq_len, seq_len)}, got {weights.shape}"

    print(
        "✅ MultiHeadAttention module executed and passed shape assertions successfully!"
    )
